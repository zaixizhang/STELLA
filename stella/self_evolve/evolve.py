"""One round of STELLA's parameter-level self-evolution, end to end.

A round is: collect trajectories -> verify them -> build datasets -> SFT ->
GRPO -> evaluate on held-out tasks -> promote or discard. The last step is the
important one. A round is only kept if the evolved policy beats the policy it
started from on tasks that were filtered out of its own training data; if it
does not, the checkpoint is recorded and discarded rather than promoted. That
is what separates evolution from drift.

    python -m stella.self_evolve.evolve \
        --round 1 \
        --base_model Qwen/Qwen3-8B \
        --heldout self_evolve_data/heldout_tasks.jsonl \
        --references self_evolve_data/references.json \
        --work_dir self_evolve_data/rounds

Each round writes ``round_<n>/round.json`` recording the data it used, the
checkpoints it produced, both evaluation reports and the promotion decision.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .dataset_builder import BuildConfig, build_datasets
from .evaluate import EvalReport, compare, evaluate_policy, load_tasks
from .schema import DATA_ROOT, Trajectory
from .serve import serve_command, wait_for_server

__all__ = ["make_stella_rollout", "served_policy", "collect_trajectories", "run_round"]


# ---------------------------------------------------------------------------
# Rollouts through the real agent (used by GRPO --mode agentic)
# ---------------------------------------------------------------------------

def make_stella_rollout(
    *,
    use_template: bool = True,
    enable_tool_creation: bool = False,
    log: bool = True,
    domain: str = "",
) -> Callable[[str, str], str]:
    """Return ``rollout(task, plan) -> final_answer`` backed by a real STELLA run.

    The candidate completion is passed to the agent as a proposed plan, the
    agent executes it with real tools, and the answer is what gets verified.
    Every rollout is logged as a trajectory, so the rollouts of one round
    become training data for the next.
    """
    import stella.core as stella_core  # noqa: PLC0415

    if getattr(stella_core, "manager_agent", None) is None:
        stella_core.initialize_stella(use_template=use_template, enable_tool_creation=enable_tool_creation)

    from .trajectory_logger import log_run  # noqa: PLC0415

    def rollout(task: str, plan: str = "") -> str:
        agent = stella_core.manager_agent
        prompt = task if not plan.strip() else (
            f"{task}\n\nA candidate plan has been proposed. Follow it where it is sound, "
            f"correct it where it is not, and report the result:\n{plan}"
        )
        result = agent.run(prompt)
        if log:
            log_run(agent, task, result, domain=domain, tags=["grpo_rollout"])
        return str(result)

    return rollout


def collect_trajectories(
    tasks: Sequence[Dict[str, Any]],
    *,
    use_template: bool = True,
    enable_tool_creation: bool = False,
    attempts: int = 1,
) -> int:
    """Run STELLA over a list of training tasks, logging each attempt.

    ``tasks`` rows are {"task": ..., "task_id": ..., "domain": ...}. Several
    attempts per task are useful before GRPO: the group needs both successes
    and failures to produce a gradient.
    """
    import stella.core as stella_core  # noqa: PLC0415

    if getattr(stella_core, "manager_agent", None) is None:
        stella_core.initialize_stella(use_template=use_template, enable_tool_creation=enable_tool_creation)
    from .trajectory_logger import log_run  # noqa: PLC0415

    n = 0
    for row in tasks:
        task = str(row.get("task", "")).strip()
        if not task:
            continue
        for attempt in range(attempts):
            started = time.time()
            try:
                result = stella_core.manager_agent.run(task)
            except Exception as e:  # noqa: BLE001 - a failed run is data too
                print(f"[evolve] run failed ({row.get('task_id')}, attempt {attempt + 1}): {e}")
                result = ""
            log_run(
                stella_core.manager_agent,
                task,
                result,
                task_id=str(row.get("task_id", "")),
                domain=str(row.get("domain", "")),
                split=str(row.get("split", "train")),
                benchmark=str(row.get("benchmark", "")),
                tags=["collect"],
            )
            n += 1
            print(f"[evolve] collected {row.get('task_id') or task[:40]!r} attempt {attempt + 1} "
                  f"in {time.time() - started:.1f}s")
    return n


# ---------------------------------------------------------------------------
# Serving a checkpoint for the duration of an evaluation
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def served_policy(
    model: str,
    *,
    lora: str = "",
    served_name: str = "stella-evolved",
    port: int = 8000,
    timeout: float = 1200.0,
    **serve_kwargs: Any,
) -> Iterator[str]:
    """Serve a checkpoint with vLLM for the duration of the block.

    Yields the API base URL. The server is always shut down on exit, including
    on error, so a failed evaluation cannot leave a GPU pinned.
    """
    cmd = serve_command(model, served_name=served_name, port=port, lora=lora, **serve_kwargs)
    print("[evolve] starting server: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    api_base = f"http://127.0.0.1:{port}/v1"
    try:
        if not wait_for_server(api_base, timeout=timeout):
            raise RuntimeError(f"server for {model} did not come up within {timeout:.0f}s")
        yield api_base
    finally:
        print("[evolve] stopping server")
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# A round
# ---------------------------------------------------------------------------

@dataclass
class RoundConfig:
    round_id: int = 1
    base_model: str = "Qwen/Qwen3-8B"
    work_dir: str = os.path.join(DATA_ROOT, "rounds")
    trajectories: str = os.path.join(DATA_ROOT, "trajectories")
    heldout_tasks_path: str = ""
    references_path: str = ""
    verifiers_path: str = ""
    default_verifier: str = ""
    min_reward: float = 0.6
    max_per_task: int = 2
    run_grpo: bool = True
    grpo_mode: str = "response"
    eval_mode: str = "response"
    min_delta: float = 0.0
    port: int = 8000
    sft_args: Sequence[str] = field(default_factory=tuple)
    grpo_args: Sequence[str] = field(default_factory=tuple)
    skip_eval: bool = False


def _load_json(path: str) -> Any:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_round(cfg: RoundConfig) -> Dict[str, Any]:
    """Run one evolution round and return its manifest."""
    from . import train_grpo, train_sft  # noqa: PLC0415

    round_dir = os.path.join(cfg.work_dir, f"round_{cfg.round_id}")
    os.makedirs(round_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "round": cfg.round_id,
        "base_model": cfg.base_model,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trajectories": os.path.abspath(cfg.trajectories),
    }

    heldout = load_tasks(cfg.heldout_tasks_path) if cfg.heldout_tasks_path else []
    if not heldout and not cfg.skip_eval:
        raise ValueError(
            "a round needs held-out tasks to decide whether to promote its checkpoint; "
            "pass --heldout, or --skip_eval to build and train without a promotion gate"
        )

    # 1. verify + build datasets, filtering out everything resembling held-out work
    print(f"\n[evolve] round {cfg.round_id}: building datasets")
    built = build_datasets(
        cfg.trajectories,
        os.path.join(round_dir, "datasets"),
        BuildConfig(
            min_reward=cfg.min_reward,
            max_per_task=cfg.max_per_task,
            heldout_tasks=tuple(t.task for t in heldout),
            allow_no_heldout=not heldout,
        ),
        references=_load_json(cfg.references_path) or {},
        verifiers=_load_json(cfg.verifiers_path) or {},
        default_verifier=cfg.default_verifier,
    )
    manifest["data"] = built["stats"]
    manifest["dataset_paths"] = built["paths"]

    # 2. SFT
    sft_dir = os.path.join(round_dir, "sft")
    print(f"\n[evolve] round {cfg.round_id}: SFT -> {sft_dir}")
    rc = train_sft.main(
        ["--model", cfg.base_model, "--dataset", built["paths"]["sft"], "--output_dir", sft_dir, *cfg.sft_args]
    )
    if rc != 0:
        raise RuntimeError(f"SFT failed (rc={rc})")
    manifest["sft_checkpoint"] = sft_dir
    evolved = sft_dir

    # 3. GRPO on top of the SFT policy
    if cfg.run_grpo:
        grpo_dir = os.path.join(round_dir, "grpo")
        print(f"\n[evolve] round {cfg.round_id}: GRPO -> {grpo_dir}")
        rc = train_grpo.main(
            [
                "--model",
                sft_dir,
                "--dataset",
                built["paths"]["grpo"],
                "--output_dir",
                grpo_dir,
                "--mode",
                cfg.grpo_mode,
                *cfg.grpo_args,
            ]
        )
        if rc != 0:
            print(f"[evolve] GRPO failed (rc={rc}); keeping the SFT checkpoint for evaluation")
        else:
            manifest["grpo_checkpoint"] = grpo_dir
            evolved = grpo_dir
    manifest["evolved_checkpoint"] = evolved

    # 4. promotion gate: same held-out tasks, same verifiers, two policies
    if cfg.skip_eval:
        manifest["promoted"] = None
        manifest["note"] = "evaluation skipped; checkpoint not promoted"
    else:
        print(f"\n[evolve] round {cfg.round_id}: evaluating baseline")
        with served_policy(cfg.base_model, served_name="stella-baseline", port=cfg.port) as api_base:
            before = evaluate_policy(
                heldout,
                policy=f"baseline:{cfg.base_model}",
                api_base=api_base,
                model_id="stella-baseline",
                mode=cfg.eval_mode,
                log_path=os.path.join(round_dir, "eval_baseline.json"),
            )
        print(f"\n[evolve] round {cfg.round_id}: evaluating evolved policy")
        with served_policy(
            cfg.base_model, lora=evolved, served_name="stella-evolved", port=cfg.port
        ) as api_base:
            after = evaluate_policy(
                heldout,
                policy=f"evolved:{evolved}",
                api_base=api_base,
                model_id="stella-evolved",
                mode=cfg.eval_mode,
                log_path=os.path.join(round_dir, "eval_evolved.json"),
            )
        decision = compare(before, after, min_delta=cfg.min_delta)
        manifest["comparison"] = decision
        manifest["promoted"] = decision["promote"]
        print("\n[evolve] " + json.dumps(decision, indent=2))
        if decision["promote"]:
            with open(os.path.join(cfg.work_dir, "PROMOTED"), "w", encoding="utf-8") as f:
                f.write(f"round {cfg.round_id}\n{evolved}\n")
            print(f"[evolve] round {cfg.round_id} PROMOTED: {evolved}")
        else:
            print(
                f"[evolve] round {cfg.round_id} NOT promoted "
                f"(delta {decision['delta']:+.4f} <= {cfg.min_delta}); checkpoint kept for the record only"
            )

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = os.path.join(round_dir, "round.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[evolve] manifest -> {path}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run one round of STELLA parameter-level self-evolution.")
    ap.add_argument("--round", type=int, default=1, dest="round_id")
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--work_dir", default=os.path.join(DATA_ROOT, "rounds"))
    ap.add_argument("--trajectories", default=os.path.join(DATA_ROOT, "trajectories"))
    ap.add_argument("--heldout", default="", dest="heldout_tasks_path")
    ap.add_argument("--references", default="", dest="references_path")
    ap.add_argument("--verifiers", default="", dest="verifiers_path")
    ap.add_argument("--default_verifier", default="")
    ap.add_argument("--min_reward", type=float, default=0.6)
    ap.add_argument("--max_per_task", type=int, default=2)
    ap.add_argument("--no_grpo", action="store_true")
    ap.add_argument("--grpo_mode", default="response", choices=["response", "agentic"])
    ap.add_argument("--eval_mode", default="response", choices=["response", "agentic"])
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--collect", default="", help="JSONL of training tasks to run before training")
    ap.add_argument("--collect_attempts", type=int, default=1)
    args = ap.parse_args(argv)

    if args.collect:
        with open(args.collect, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print(f"[evolve] collecting trajectories for {len(rows)} tasks")
        collect_trajectories(rows, attempts=args.collect_attempts)

    cfg = RoundConfig(
        round_id=args.round_id,
        base_model=args.base_model,
        work_dir=args.work_dir,
        trajectories=args.trajectories,
        heldout_tasks_path=args.heldout_tasks_path,
        references_path=args.references_path,
        verifiers_path=args.verifiers_path,
        default_verifier=args.default_verifier,
        min_reward=args.min_reward,
        max_per_task=args.max_per_task,
        run_grpo=not args.no_grpo,
        grpo_mode=args.grpo_mode,
        eval_mode=args.eval_mode,
        min_delta=args.min_delta,
        port=args.port,
        skip_eval=args.skip_eval,
    )
    run_round(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
