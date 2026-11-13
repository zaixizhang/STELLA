"""Evaluate a policy on held-out scientific tasks.

This is the measurement that decides whether a round of self-evolution is kept
or discarded, so it is deliberately strict:

* tasks come from a held-out file that the dataset builder is told to filter
  against, so the evaluation set cannot appear in the training data;
* every task is scored by its own verifier, never by the policy's own report
  of success;
* the comparison is always *same tasks, same verifier, two policies* -- an
  evolved checkpoint against the policy it started from.

Two evaluation modes, matching the two GRPO modes: ``response`` asks the policy
directly, ``agentic`` runs the full STELLA loop with that policy attached and
scores the run's final answer.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .reward import _VERIFIERS  # noqa: PLC2701 - verifier registry is the public contract

__all__ = ["EvalTask", "EvalReport", "load_tasks", "evaluate_policy", "compare"]


@dataclass
class EvalTask:
    task: str
    reference: Any
    verifier: str
    task_id: str = ""
    domain: str = ""
    system: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalTask":
        return cls(
            task=str(d.get("task") or d.get("prompt") or ""),
            reference=d.get("reference"),
            verifier=str(d.get("verifier") or ""),
            task_id=str(d.get("task_id") or ""),
            domain=str(d.get("domain") or ""),
            system=str(d.get("system") or ""),
        )


@dataclass
class EvalReport:
    policy: str
    n: int
    mean_score: float
    pass_rate: float
    per_domain: Dict[str, float] = field(default_factory=dict)
    per_task: List[Dict[str, Any]] = field(default_factory=list)
    wall_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "n": self.n,
            "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "per_domain": {k: round(v, 4) for k, v in self.per_domain.items()},
            "wall_sec": round(self.wall_sec, 1),
            "per_task": self.per_task,
        }


def load_tasks(path: str) -> List[EvalTask]:
    """Load held-out tasks from JSONL or a JSON list."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    rows: List[Dict[str, Any]]
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    tasks = [EvalTask.from_dict(r) for r in rows]
    bad = [t.task_id or t.task[:40] for t in tasks if not t.verifier or t.reference is None]
    if bad:
        raise ValueError(f"held-out tasks without verifier/reference cannot be scored: {bad[:5]}")
    return tasks


# ---------------------------------------------------------------------------
# Policy callers
# ---------------------------------------------------------------------------

def _chat_caller(api_base: str, model_id: str, *, temperature: float, max_tokens: int) -> Callable[[EvalTask], str]:
    """Direct single-turn call to an OpenAI-compatible endpoint."""
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(base_url=api_base, api_key=os.environ.get("STELLA_LOCAL_POLICY_API_KEY", "EMPTY"))

    def call(task: EvalTask) -> str:
        messages = ([{"role": "system", "content": task.system}] if task.system else []) + [
            {"role": "user", "content": task.task}
        ]
        resp = client.chat.completions.create(
            model=model_id, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""

    return call


def _agentic_caller(api_base: str, model_id: str) -> Callable[[EvalTask], str]:
    """Full STELLA run with the policy attached, scored on the final answer."""
    from .serve import attach_policy_to_agents  # noqa: PLC0415

    attach_policy_to_agents(api_base, model_id)
    import stella_core  # noqa: PLC0415

    agent = getattr(stella_core, "manager_agent", None)
    if agent is None:
        raise RuntimeError("stella_core.manager_agent is not initialised; call initialize_stella() first")

    def call(task: EvalTask) -> str:
        return str(agent.run(task.task))

    return call


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    tasks: Sequence[EvalTask],
    *,
    policy: str = "",
    api_base: str = "",
    model_id: str = "",
    mode: str = "response",
    caller: Optional[Callable[[EvalTask], str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    pass_threshold: float = 0.999,
    log_path: str = "",
) -> EvalReport:
    """Score every task with its own verifier.

    ``caller`` can be supplied directly (useful for testing or for a policy
    reached some other way); otherwise one is built from ``api_base``/``mode``.
    """
    if caller is None:
        if not api_base or not model_id:
            raise ValueError("provide either caller=... or api_base=... and model_id=...")
        caller = (
            _agentic_caller(api_base, model_id)
            if mode == "agentic"
            else _chat_caller(api_base, model_id, temperature=temperature, max_tokens=max_tokens)
        )

    started = time.time()
    scores: List[float] = []
    per_domain: Dict[str, List[float]] = {}
    per_task: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        answer, err = "", ""
        try:
            answer = caller(task)
        except Exception as e:  # noqa: BLE001 - a failed call is a zero, and is recorded as such
            err = str(e)[:500]

        fn = _VERIFIERS.get(task.verifier)
        if fn is None:
            raise KeyError(f"unknown verifier {task.verifier!r} for task {task.task_id or task.task[:40]!r}")
        try:
            score, detail = fn(answer, task.reference, task.task) if not err else (0.0, f"call failed: {err}")
        except Exception as e:  # noqa: BLE001
            score, detail = 0.0, f"verifier error: {e}"

        scores.append(float(score))
        per_domain.setdefault(task.domain or "unspecified", []).append(float(score))
        per_task.append(
            {
                "task_id": task.task_id,
                "domain": task.domain,
                "verifier": task.verifier,
                "score": round(float(score), 4),
                "detail": detail,
                "answer_chars": len(answer),
                "error": err,
            }
        )
        print(f"[eval] {i}/{len(tasks)} {task.task_id or task.task[:40]!r}: {score:.3f} ({detail})")

    report = EvalReport(
        policy=policy or model_id or "unknown",
        n=len(tasks),
        mean_score=statistics.fmean(scores) if scores else 0.0,
        pass_rate=(sum(1 for s in scores if s >= pass_threshold) / len(scores)) if scores else 0.0,
        per_domain={k: statistics.fmean(v) for k, v in per_domain.items()},
        per_task=per_task,
        wall_sec=time.time() - started,
    )
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return report


def compare(before: EvalReport, after: EvalReport, *, min_delta: float = 0.0) -> Dict[str, Any]:
    """Round-promotion gate: did the evolved policy actually improve?"""
    delta = after.mean_score - before.mean_score
    regressions = []
    before_by_task = {t["task_id"]: t["score"] for t in before.per_task if t.get("task_id")}
    for t in after.per_task:
        tid = t.get("task_id")
        if tid and tid in before_by_task and t["score"] < before_by_task[tid] - 1e-9:
            regressions.append({"task_id": tid, "before": before_by_task[tid], "after": t["score"]})
    return {
        "before": before.policy,
        "after": after.policy,
        "mean_before": round(before.mean_score, 4),
        "mean_after": round(after.mean_score, 4),
        "delta": round(delta, 4),
        "pass_before": round(before.pass_rate, 4),
        "pass_after": round(after.pass_rate, 4),
        "promote": bool(delta > min_delta),
        "n_regressions": len(regressions),
        "regressions": regressions[:20],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a policy on held-out verified tasks.")
    ap.add_argument("--tasks", required=True, help="held-out tasks (JSONL or JSON list)")
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--mode", default="response", choices=["response", "agentic"])
    ap.add_argument("--policy_name", default="")
    ap.add_argument("--out", default="", help="write the report JSON here")
    ap.add_argument("--baseline_report", default="", help="compare against this earlier report JSON")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args(argv)

    report = evaluate_policy(
        load_tasks(args.tasks),
        policy=args.policy_name or args.model_id,
        api_base=args.api_base,
        model_id=args.model_id,
        mode=args.mode,
        temperature=args.temperature,
        log_path=args.out,
    )
    print(json.dumps({k: v for k, v in report.to_dict().items() if k != "per_task"}, indent=2))

    if args.baseline_report:
        with open(args.baseline_report, "r", encoding="utf-8") as f:
            base = json.load(f)
        before = EvalReport(
            policy=base.get("policy", "baseline"),
            n=base.get("n", 0),
            mean_score=base.get("mean_score", 0.0),
            pass_rate=base.get("pass_rate", 0.0),
            per_domain=base.get("per_domain", {}),
            per_task=base.get("per_task", []),
        )
        print(json.dumps(compare(before, report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
