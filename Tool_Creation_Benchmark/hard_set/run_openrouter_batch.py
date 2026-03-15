#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def project_root_from_here() -> Path:
    # .../Appeal/submission/category4_repro_package/code -> repo root
    return Path(__file__).resolve().parents[4]


@dataclass
class Job:
    source_cfg: Path
    run_cfg: Path
    run_id: str
    log_file: Path


def resolve_path(root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (root / p)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_env_file(path: Path) -> int:
    """
    Load KEY=VALUE pairs from an env file into process environment.
    Returns number of variables loaded.
    """
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key:
            os.environ[key] = val
            loaded += 1
    return loaded


def read_task_ids(task_csv: Path, group: str) -> set[str]:
    prefixes = {
        "tdc": ("polarishub/",),
        "proteingym": ("proteingym-dms/",),
        "all": ("polarishub/", "proteingym-dms/"),
    }
    allow = prefixes[group]
    out: set[str] = set()
    with task_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            tid = (row.get("task_id") or "").strip()
            if tid and any(tid.startswith(pref) for pref in allow):
                out.add(tid)
    return out


def find_source_configs(configs_dir: Path, config_glob: str, allowed_task_ids: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(configs_dir.glob(config_glob)):
        try:
            d = load_json(p)
        except Exception:
            continue
        task_id = str(d.get("task_id", "")).strip()
        if task_id in allowed_task_ids:
            out.append(p)
    return out


def make_job(
    src_cfg: Path,
    run_cfg_dir: Path,
    out_batch_dir: Path,
    batch_tag: str,
    models: list[dict[str, str]],
    repeats: int,
) -> Job:
    source_name = src_cfg.stem
    run_id = f"{batch_tag}_{source_name}"
    run_cfg = run_cfg_dir / f"{run_id}.json"
    run_log = out_batch_dir / f"{run_id}.console.log"

    d = load_json(src_cfg)
    d["models"] = models
    d["repeats"] = repeats
    d["run_id"] = run_id
    d["output_dir"] = str(out_batch_dir)
    save_json(run_cfg, d)

    return Job(source_cfg=src_cfg, run_cfg=run_cfg, run_id=run_id, log_file=run_log)


def run_job(runner: Path, job: Job) -> int:
    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    with job.log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            ["python", str(runner), "--config", str(job.run_cfg)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel OpenRouter runner for reviewer package (non-STELLA/non-Biomni models)."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root_from_here()
    cfg_path = resolve_path(root, args.config)
    cfg = load_json(cfg_path)

    env_file_raw = str(cfg.get("env_file", "Appeal/submission/category4_repro_package/env/reviewer_api_keys.local.env"))
    env_file = resolve_path(root, env_file_raw)
    loaded_count = load_env_file(env_file)

    provider = str(cfg.get("api_provider", "openrouter")).strip().lower()
    if provider not in {"openrouter", "anthropic", "either"}:
        raise SystemExit("api_provider must be one of: openrouter, anthropic, either")

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not args.dry_run:
        if provider == "openrouter" and not has_openrouter:
            raise SystemExit(
                f"OPENROUTER_API_KEY is not set. Put it in {env_file} or export it."
            )
        if provider == "anthropic" and not has_anthropic:
            raise SystemExit(
                f"ANTHROPIC_API_KEY is not set. Put it in {env_file} or export it."
            )
        if provider == "either" and not (has_openrouter or has_anthropic):
            raise SystemExit(
                f"Neither OPENROUTER_API_KEY nor ANTHROPIC_API_KEY is set. Put keys in {env_file} or export them."
            )

    group = str(cfg.get("group", "tdc")).strip().lower()
    if group not in {"tdc", "proteingym", "all"}:
        raise SystemExit("group must be one of: tdc, proteingym, all")

    repeats = int(cfg.get("repeats", 3))
    max_parallel = int(cfg.get("max_parallel", 3))
    config_glob = str(cfg.get("config_glob", "benchmark_config_c4_*_openrouter_tasktext_others.json"))
    batch_tag = str(cfg.get("batch_tag", "")).strip()
    if not batch_tag:
        batch_tag = f"openrouter_{group}_r{repeats}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    task_csv = resolve_path(root, str(cfg["task_csv"]))
    configs_dir = resolve_path(root, str(cfg["configs_dir"]))
    runner = resolve_path(root, str(cfg["runner"]))
    out_root = resolve_path(root, str(cfg["out_root"]))

    models = cfg.get("models", [])
    if not isinstance(models, list) or not models:
        raise SystemExit("config.models must be a non-empty list, e.g. [{\"name\":\"GPT5\",\"model_id\":\"openai/gpt-5\"}]")

    allowed_task_ids = read_task_ids(task_csv, group)
    src_cfgs = find_source_configs(configs_dir, config_glob, allowed_task_ids)
    if not src_cfgs:
        raise SystemExit("No matching task configs found.")

    out_batch_dir = out_root / batch_tag
    run_cfg_dir = out_batch_dir / "_tmp_configs"
    out_batch_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        make_job(
            src_cfg=s,
            run_cfg_dir=run_cfg_dir,
            out_batch_dir=out_batch_dir,
            batch_tag=batch_tag,
            models=models,
            repeats=repeats,
        )
        for s in src_cfgs
    ]

    print(f"[INFO] config={cfg_path}")
    print(f"[INFO] env_file={env_file} loaded_vars={loaded_count}")
    print(f"[INFO] provider={provider}")
    print(f"[INFO] group={group} tasks={len(jobs)} repeats={repeats} max_parallel={max_parallel}")
    print(f"[INFO] batch_dir={out_batch_dir}")

    manifest = {
        "config": str(cfg_path),
        "group": group,
        "repeats": repeats,
        "max_parallel": max_parallel,
        "batch_dir": str(out_batch_dir),
        "jobs": [
            {
                "run_id": j.run_id,
                "task_cfg": str(j.source_cfg),
                "run_cfg": str(j.run_cfg),
                "log_file": str(j.log_file),
            }
            for j in jobs
        ],
        "results": [],
    }
    manifest_path = out_batch_dir / "batch_manifest.json"
    save_json(manifest_path, manifest)

    if args.dry_run:
        print(f"[DRY_RUN] manifest={manifest_path}")
        return 0

    failed = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        future_map = {ex.submit(run_job, runner, job): job for job in jobs}
        for fut in as_completed(future_map):
            job = future_map[fut]
            rc = fut.result()
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{status}] {job.run_id}")
            manifest["results"].append(
                {
                    "run_id": job.run_id,
                    "returncode": rc,
                    "log_file": str(job.log_file),
                }
            )
            if rc != 0:
                failed += 1
            save_json(manifest_path, manifest)

    print(f"[DONE] batch_dir={out_batch_dir}")
    if failed:
        print(f"[WARN] failed_jobs={failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
