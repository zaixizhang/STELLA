#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BIOML_CONFIG = ROOT / "Appeal/Appeal_benchmark/Category4/configs/benchmark_config_c4_cbx4_stella_only.json"
DEFAULT_OPENROUTER_CONFIG = ROOT / "Appeal/Appeal_benchmark/Category4/configs/benchmark_config_c4_cbx4_openrouter_pilot.json"

BIOML_SCRIPT = ROOT / "Appeal/Appeal_benchmark/Category4/src/run_category4_bioml.py"
OPENROUTER_SCRIPT = ROOT / "Appeal/Appeal_benchmark/Category4/src/run_category4_openrouter_pilot.py"
BIOMNI_LOCAL_SCRIPT = ROOT / "Appeal/Appeal_benchmark/Category4/src/run_category4_biomni_local.py"

MODEL_ORDER = ["STELLA", "Biomni", "GPT4o", "DeepSeek", "Gemini", "Grok", "o3", "ClaudeOpus"]
OPENROUTER_MODEL_MAP = {
    "GPT4o": ("GPT4o", "openai/gpt-4o"),
    "DeepSeek": ("DeepSeek", "deepseek/deepseek-chat"),
    "Gemini": ("Gemini", "google/gemini-2.5-flash"),
    "Grok": ("Grok", "x-ai/grok-4.1-fast"),
    "o3": ("o3", "openai/o3"),
    "ClaudeOpus": ("ClaudeOpus", "anthropic/claude-opus-4"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    all_rows = existing_rows + rows
    # Keep deterministic column order across mixed runners (bioml + openrouter)
    fieldnames: list[str] = []
    for r in all_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)


def _run_cmd(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return proc.returncode


def _run_biomni_local(base_bioml: dict[str, Any], run_id: str, repeats: int, out_dir: Path) -> list[dict[str, Any]]:
    task_dir = base_bioml.get("task_bundle_dir", "")
    task_id = str(base_bioml.get("task_id", ""))
    rows: list[dict[str, Any]] = []
    for repeat_idx in range(1, repeats + 1):
        run_dir = out_dir / f"{run_id}_Biomni" / "biomni_local" / f"repeat_{repeat_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(BIOMNI_LOCAL_SCRIPT),
            "--data-dir",
            str(base_bioml["data_dir"]),
            "--output-dir",
            str(run_dir),
            "--grade",
            "--task-id",
            task_id,
            "--biomlbench-root",
            str(base_bioml["biomlbench_root"]),
        ]
        if task_dir:
            cmd.extend(["--task-dir", str(task_dir)])
        rc = _run_cmd(cmd, ROOT)
        meta_path = run_dir / "run_meta.json"
        score = ""
        runtime_sec: float | str = ""
        notes = ""
        gr: dict[str, Any] = {}
        if meta_path.exists():
            meta = _load_json(meta_path)
            gr = meta.get("grade_report") or {}
            score = gr.get("score", "")
            s0, s1 = meta.get("started_at", ""), meta.get("ended_at", "")
            try:
                runtime_sec = round((datetime.fromisoformat(s1) - datetime.fromisoformat(s0)).total_seconds(), 2)
            except Exception:
                runtime_sec = ""
            notes = meta.get("error", "") or ""
        rows.append(
            {
                "task_id": base_bioml.get("task_id", ""),
                "task_name": base_bioml.get("task_name", ""),
                "agent": "biomni_local",
                "repeat": repeat_idx,
                "returncode": rc,
                "runtime_sec": runtime_sec,
                "score": score,
                "leaderboard_percentile": (gr.get("leaderboard_percentile", "") if meta_path.exists() else ""),
                "gold_medal": (gr.get("gold_medal", "") if meta_path.exists() else ""),
                "silver_medal": (gr.get("silver_medal", "") if meta_path.exists() else ""),
                "bronze_medal": (gr.get("bronze_medal", "") if meta_path.exists() else ""),
                "above_median": (gr.get("above_median", "") if meta_path.exists() else ""),
                "valid_submission": (gr.get("valid_submission", "") if meta_path.exists() else ""),
                "submission_jsonl": "",
                "grade_report_path": str(run_dir / "grade_report.json"),
                "run_dir": str(run_dir),
                "evaluation_json": "",
                "log_path": str(run_dir / "biomni_local.log"),
                "notes": notes if notes else ("ok" if rc == 0 else "failed"),
                "panel_model": "Biomni",
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Run Category4 mixed panel in one command.")
    p.add_argument("--bioml-config", default=str(DEFAULT_BIOML_CONFIG), help="Base config for biomlbench-native agents")
    p.add_argument("--openrouter-config", default=str(DEFAULT_OPENROUTER_CONFIG), help="Base config for OpenRouter pilot")
    p.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help="Comma-separated model names from: STELLA,Biomni,GPT4o,DeepSeek,Gemini,Grok,o3,ClaudeOpus",
    )
    p.add_argument("--stella-repeats", type=int, default=3)
    p.add_argument("--others-repeats", type=int, default=1)
    p.add_argument(
        "--run-id",
        default=f"category4_full_panel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Top-level run id for merged outputs",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "Appeal/Test_run/category4"),
        help="Output root directory",
    )
    p.add_argument(
        "--biomni-mode",
        choices=["local", "docker"],
        default="local",
        help="Run Biomni via local runner (recommended) or biomlbench docker agent",
    )
    args = p.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in selected if m not in MODEL_ORDER]
    if invalid:
        raise SystemExit(f"Unknown models: {invalid}")

    base_bioml = _load_json(Path(args.bioml_config))
    base_openrouter = _load_json(Path(args.openrouter_config)) if Path(args.openrouter_config).exists() else {}

    needs_openrouter = any(m in OPENROUTER_MODEL_MAP for m in selected)
    # OpenRouter pilot is ProteinGym-specific in current implementation.
    if needs_openrouter and base_openrouter and base_openrouter.get("task_id") != base_bioml.get("task_id"):
        raise SystemExit("bioml-config task_id and openrouter-config task_id mismatch")

    run_root = Path(args.out_dir) / args.run_id
    tmp_cfg_dir = run_root / "_tmp_configs"
    merged_csv = run_root / "summary_merged.csv"
    run_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for model in selected:
        if model == "Biomni" and args.biomni_mode == "local":
            print(f"[RUN] Biomni via local runner | repeats={args.others_repeats}")
            try:
                rows = _run_biomni_local(base_bioml, args.run_id, args.others_repeats, Path(args.out_dir))
                all_rows.extend(rows)
            except Exception as e:
                print(f"[WARN] Biomni local run failed: {e}")
            continue

        if model in ("STELLA", "Biomni"):
            cfg = dict(base_bioml)
            cfg["agents"] = [model.lower() if model == "STELLA" else "biomni"]
            cfg["repeats"] = args.stella_repeats if model == "STELLA" else args.others_repeats
            cfg["run_id"] = f"{args.run_id}_{model}"
            cfg_path = tmp_cfg_dir / f"{model}_bioml.json"
            _write_json(cfg_path, cfg)

            print(f"[RUN] {model} via biomlbench runner | repeats={cfg['repeats']}")
            rc = _run_cmd([sys.executable, str(BIOML_SCRIPT), "--config", str(cfg_path)], ROOT)
            if rc != 0:
                print(f"[WARN] {model} run failed (returncode={rc})")

            summary_csv = Path(cfg["out_dir"]) / cfg["run_id"] / "summary.csv"
            rows = _read_rows(summary_csv)
            for r in rows:
                r["panel_model"] = model
            all_rows.extend(rows)
            continue

        # OpenRouter models
        if not base_openrouter:
            raise SystemExit(f"OpenRouter config not found, cannot run model {model}")
        name, model_id = OPENROUTER_MODEL_MAP[model]
        cfg = dict(base_openrouter)
        cfg["models"] = [{"name": name, "model_id": model_id}]
        cfg["repeats"] = args.others_repeats
        cfg["run_id"] = f"{args.run_id}_{model}"
        cfg_path = tmp_cfg_dir / f"{model}_openrouter.json"
        _write_json(cfg_path, cfg)

        print(f"[RUN] {model} via OpenRouter pilot | repeats={cfg['repeats']}")
        rc = _run_cmd([sys.executable, str(OPENROUTER_SCRIPT), "--config", str(cfg_path)], ROOT)
        if rc != 0:
            print(f"[WARN] {model} run failed (returncode={rc})")

        summary_csv = Path(cfg["output_dir"]) / cfg["run_id"] / "summary.csv"
        rows = _read_rows(summary_csv)
        for r in rows:
            r["panel_model"] = model
        all_rows.extend(rows)

    _append_rows(merged_csv, all_rows)
    print(f"merged_summary={merged_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
