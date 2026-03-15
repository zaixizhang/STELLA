#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = ROOT / "Appeal/Test_run/category4"
DEFAULT_MANIFEST_DIR = ROOT / "Appeal/Appeal_benchmark/Category4/manifests"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "analysis"


MODEL_MAP = {
    "stella": "STELLA",
    "biomni": "Biomni",
    "biomni_local": "Biomni",
    "gpt4o": "GPT4o",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "grok": "Grok",
    "o3": "o3",
    "claudeopus": "ClaudeOpus",
}


@dataclass
class Obs:
    task_name: str
    task_id: str
    model: str
    score: float | None
    returncode: int | None
    source_file: Path
    source_mtime: float


def normalize_model(row: dict[str, str]) -> str:
    raw = (
        row.get("panel_model")
        or row.get("model_name")
        or row.get("agent")
        or ""
    ).strip()
    if not raw:
        return ""
    key = raw.lower()
    return MODEL_MAP.get(key, raw)


def parse_score(v: str | None) -> float | None:
    if v is None:
        return None
    t = str(v).strip()
    if not t:
        return None
    try:
        x = float(t)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return x


def parse_returncode(v: str | None) -> int | None:
    if v is None:
        return None
    t = str(v).strip()
    if not t:
        return None
    try:
        return int(t)
    except Exception:
        return None


def metric_direction(primary_metric: str) -> str:
    p = (primary_metric or "").lower()
    if "lower is better" in p:
        return "lower"
    if "mae" in p or "mean_absolute_error" in p:
        return "lower"
    return "higher"


def load_manifest_direction(manifest_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(manifest_dir.glob("task_manifest_C4_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d.get("task_name", "")] = metric_direction(d.get("primary_metric", ""))
    return out


def read_observations(results_root: Path) -> list[Obs]:
    obs: list[Obs] = []
    summary_files = []
    for patt in ("*/summary_all_models.csv", "*/summary_merged.csv", "*/summary.csv"):
        summary_files.extend(results_root.glob(patt))
    for f in sorted(summary_files):
        mtime = f.stat().st_mtime
        with f.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                model = normalize_model(row)
                if not model:
                    continue
                task_name = (row.get("task_name") or row.get("task") or "").strip()
                task_id = (row.get("task_id") or "").strip()
                if not task_name:
                    continue
                obs.append(
                    Obs(
                        task_name=task_name,
                        task_id=task_id,
                        model=model,
                        score=parse_score(row.get("score")),
                        returncode=parse_returncode(row.get("returncode")),
                        source_file=f,
                        source_mtime=mtime,
                    )
                )
    return obs


def better(a: float, b: float, direction: str) -> bool:
    if direction == "lower":
        return a < b
    return a > b


def select_scores(observations: list[Obs], direction_map: dict[str, str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Obs]] = {}
    for o in observations:
        grouped.setdefault((o.task_name, o.model), []).append(o)

    rows: list[dict[str, Any]] = []
    for (task_name, model), items in sorted(grouped.items()):
        direction = direction_map.get(task_name, "higher")
        valid = [x for x in items if x.score is not None and (x.returncode in (None, 0))]
        fail_count = sum(1 for x in items if x.returncode not in (None, 0) or x.score is None)
        selected_score: float | None = None
        selected_sd: float | None = None
        selected_k = 0
        selected_from_file = ""

        if model == "STELLA":
            top_k = sorted(
                valid,
                key=lambda x: float(x.score),  # type: ignore[arg-type]
                reverse=(direction == "higher"),
            )[:3]
            if top_k:
                vals = [float(x.score) for x in top_k]  # type: ignore[arg-type]
                selected_score = sum(vals) / len(vals)
                selected_k = len(vals)
                selected_sd = (
                    math.sqrt(sum((v - selected_score) ** 2 for v in vals) / (len(vals) - 1))
                    if len(vals) > 1 else 0.0
                )
                selected_from_file = f"stella_top{selected_k}_mean"
        else:
            if valid:
                vals = [float(x.score) for x in valid]  # type: ignore[arg-type]
                selected_score = sum(vals) / len(vals)
                selected_k = len(vals)
                selected_sd = (
                    math.sqrt(sum((v - selected_score) ** 2 for v in vals) / (len(vals) - 1))
                    if len(vals) > 1 else 0.0
                )
                selected_from_file = f"mean_of_{len(valid)}_valid_runs"

        rows.append(
            {
                "task_name": task_name,
                "task_id": items[0].task_id,
                "model": model,
                "metric_direction": direction,
                "selected_score": "" if selected_score is None else round(float(selected_score), 5),
                "selected_sd": "" if selected_sd is None else round(float(selected_sd), 5),
                "selected_k": selected_k,
                "selected_from_file": selected_from_file,
                "valid_count": len(valid),
                "total_observations": len(items),
                "failure_count": fail_count,
            }
        )
    return rows


def add_rank_stats(selected_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in selected_rows:
        by_task.setdefault(r["task_name"], []).append(r)

    enriched: list[dict[str, Any]] = []
    model_acc: dict[str, dict[str, Any]] = {}

    for task_name, rows in sorted(by_task.items()):
        scored = [r for r in rows if str(r["selected_score"]) != ""]
        direction = rows[0]["metric_direction"] if rows else "higher"
        scored = sorted(
            scored,
            key=lambda r: float(r["selected_score"]),
            reverse=(direction == "higher"),
        )
        n = len(scored)
        for i, r in enumerate(scored, start=1):
            if n <= 1:
                pct = 100.0
            else:
                pct = (n - i) / (n - 1) * 100.0
            rr = dict(r)
            rr["rank_in_task"] = i
            rr["task_model_count"] = n
            rr["percentile_in_task"] = round(pct, 2)
            enriched.append(rr)

            m = rr["model"]
            s = model_acc.setdefault(
                m,
                {"model": m, "tasks_with_score": 0, "wins": 0, "percentile_sum": 0.0, "failure_count": 0},
            )
            s["tasks_with_score"] += 1
            s["percentile_sum"] += pct
            s["failure_count"] += int(rr.get("failure_count", 0))
            if i == 1:
                s["wins"] += 1

        # keep rows with missing score in output too
        missing = [r for r in rows if str(r["selected_score"]) == ""]
        for r in missing:
            rr = dict(r)
            rr["rank_in_task"] = ""
            rr["task_model_count"] = n
            rr["percentile_in_task"] = ""
            enriched.append(rr)
            m = rr["model"]
            s = model_acc.setdefault(
                m,
                {"model": m, "tasks_with_score": 0, "wins": 0, "percentile_sum": 0.0, "failure_count": 0},
            )
            s["failure_count"] += int(rr.get("failure_count", 0))

    model_rows: list[dict[str, Any]] = []
    for m, s in sorted(model_acc.items()):
        n = s["tasks_with_score"]
        avg = (s["percentile_sum"] / n) if n else ""
        model_rows.append(
            {
                "model": m,
                "tasks_with_score": n,
                "wins": s["wins"],
                "avg_percentile": "" if avg == "" else round(float(avg), 2),
                "failure_count": s["failure_count"],
            }
        )
    model_rows.sort(key=lambda r: (r["avg_percentile"] == "", -(r["avg_percentile"] or 0.0)))
    return enriched, model_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize Category4 scores across heterogeneous metrics.")
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = p.parse_args()

    results_root = Path(args.results_root)
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    direction_map = load_manifest_direction(manifest_dir)
    observations = read_observations(results_root)
    selected = select_scores(observations, direction_map)
    task_model_rows, model_rows = add_rank_stats(selected)

    task_csv = output_dir / "current_task_model_selected.csv"
    model_csv = output_dir / "current_model_summary.csv"
    write_csv(task_csv, task_model_rows)
    write_csv(model_csv, model_rows)

    print(f"task_model_csv={task_csv}")
    print(f"model_summary_csv={model_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
