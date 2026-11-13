"""Turn verified trajectories into SFT and GRPO training sets.

Two datasets are produced from the same pool of verified trajectories:

* **SFT** -- step-level ``(prompt, completion)`` pairs in conversational form.
  One sample per policy decision, so a 9-step run teaches nine decisions, not
  one. Completion-only loss is used (TRL's prompt-completion format), so the
  long tool-output context is conditioned on but never trained on.
* **GRPO** -- task-level prompts plus the metadata a reward function needs
  (reference answer, verifier name, tool catalogue). Rollouts are generated
  during training and scored by :mod:`self_evolve.reward`.

Three filters stand between the trajectory pool and the training set, and all
three are on by default:

``verified``        only trajectories a verifier passed (see reward.py)
``best-of-n``       per task, keep only the top-scoring attempts
``decontamination`` drop anything whose task matches a held-out/benchmark task

Decontamination matters for the claim being made: if evolution data leaks the
evaluation tasks, a gain on those tasks says nothing about transfer. The
default is to refuse to build a dataset when no held-out set is supplied.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .reward import RewardWeights, verify_trajectories
from .schema import DATA_ROOT, StepRecord, Trajectory, iter_trajectories

__all__ = ["BuildConfig", "build_sft_samples", "build_grpo_samples", "build_datasets", "dataset_stats"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BuildConfig:
    # admission
    require_verified: bool = True
    min_reward: float = 0.6
    # best-of-n distillation: keep the n best attempts per task
    max_per_task: int = 2
    # step filtering
    drop_error_steps: bool = True
    min_completion_chars: int = 16
    max_prompt_chars: int = 60_000
    keep_agents: Tuple[str, ...] = ()  # () = all agents
    # hygiene
    dedup: bool = True
    heldout_tasks: Tuple[str, ...] = ()
    allow_no_heldout: bool = False
    ngram: int = 8  # token n-gram used for near-duplicate contamination checks
    # splits
    train_split_only: bool = True  # ignore trajectories marked split="heldout"


# ---------------------------------------------------------------------------
# Contamination / dedup helpers
# ---------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _ngrams(s: str, n: int) -> Set[str]:
    toks = _norm_text(s).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class ContaminationFilter:
    """Exact and near-duplicate matching of a task against a held-out set."""

    def __init__(self, heldout_tasks: Iterable[str], ngram: int = 8):
        self.exact: Set[str] = set()
        self.grams: Set[str] = set()
        self.ngram = ngram
        for t in heldout_tasks or []:
            t = str(t)
            if not t.strip():
                continue
            self.exact.add(_norm_text(t))
            self.grams |= _ngrams(t, ngram)

    def __bool__(self) -> bool:
        return bool(self.exact)

    def is_contaminated(self, task: str) -> bool:
        if not self.exact:
            return False
        if _norm_text(task) in self.exact:
            return True
        return bool(_ngrams(task, self.ngram) & self.grams)


def _fingerprint(prompt: Sequence[Dict[str, Any]], completion: str) -> str:
    h = hashlib.sha256()
    for m in prompt[-2:]:  # last turns carry the discriminating context
        h.update(_norm_text(m.get("content", "")).encode("utf-8", "ignore"))
    h.update(b"||")
    h.update(_norm_text(completion).encode("utf-8", "ignore"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def _clean_prompt(messages: Sequence[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """Keep roles a chat template accepts, and trim from the middle if long.

    The system prompt and the most recent turns are what the policy needs; the
    middle of a long tool-output history is dropped first.
    """
    msgs = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in messages
        if str(m.get("content", "")).strip() and str(m.get("role", "")) in ("system", "user", "assistant", "tool")
    ]
    if not msgs:
        return []
    total = sum(len(m["content"]) for m in msgs)
    if total <= max_chars:
        return msgs
    head = [msgs[0]] if msgs[0]["role"] == "system" else []
    tail: List[Dict[str, Any]] = []
    budget = max_chars - sum(len(m["content"]) for m in head)
    for m in reversed(msgs[len(head) :]):
        if budget - len(m["content"]) < 0:
            break
        tail.insert(0, m)
        budget -= len(m["content"])
    if not tail:  # a single enormous turn: truncate it rather than drop it
        last = dict(msgs[-1])
        last["content"] = last["content"][-max(budget, 2000) :]
        tail = [last]
    return head + tail


def _step_completion(step: StepRecord) -> str:
    """The text to train on for a step: the policy's own output."""
    if step.output_message and str(step.output_message.get("content", "")).strip():
        return str(step.output_message["content"])
    return step.output_text


def build_sft_samples(trajectories: Sequence[Trajectory], cfg: Optional[BuildConfig] = None) -> List[Dict[str, Any]]:
    """Step-level prompt/completion pairs from admitted trajectories."""
    cfg = cfg or BuildConfig()
    admitted = _admit(trajectories, cfg)
    contam = ContaminationFilter(cfg.heldout_tasks, cfg.ngram)
    seen: Set[str] = set()
    samples: List[Dict[str, Any]] = []

    for traj in admitted:
        if contam.is_contaminated(traj.task):
            continue
        for step in traj.steps:
            if cfg.keep_agents and step.agent not in cfg.keep_agents:
                continue
            if cfg.drop_error_steps and step.error:
                continue
            completion = _step_completion(step)
            if len(completion.strip()) < cfg.min_completion_chars:
                continue
            prompt = _clean_prompt(step.input_messages, cfg.max_prompt_chars)
            if not prompt:
                continue
            fp = _fingerprint(prompt, completion)
            if cfg.dedup and fp in seen:
                continue
            seen.add(fp)
            samples.append(
                {
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": completion}],
                    "trajectory_id": traj.trajectory_id,
                    "task_id": traj.task_id,
                    "agent": step.agent,
                    "domain": traj.domain,
                    "reward": traj.outcome_reward,
                    "tools": [tc.name for tc in step.tool_calls],
                    "fingerprint": fp,
                }
            )
    return samples


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

def build_grpo_samples(
    trajectories: Sequence[Trajectory],
    cfg: Optional[BuildConfig] = None,
    *,
    references: Optional[Dict[str, Any]] = None,
    tool_catalog: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Task-level prompts for on-policy optimisation.

    One row per distinct task, carrying what the reward function needs. Tasks
    that no verifier can score are skipped: GRPO without a verifiable reward
    optimises the proxy, not the science.
    """
    cfg = cfg or BuildConfig()
    references = references or {}
    contam = ContaminationFilter(cfg.heldout_tasks, cfg.ngram)
    rows: Dict[str, Dict[str, Any]] = {}

    for traj in trajectories:
        if cfg.train_split_only and traj.split == "heldout":
            continue
        if contam.is_contaminated(traj.task):
            continue
        key = traj.task_id or _norm_text(traj.task)
        if key in rows:
            continue
        verifier = traj.verifier
        reference = references.get(traj.task_id, references.get(traj.task))
        if not verifier or reference is None:
            continue
        first = traj.steps[0] if traj.steps else None
        system = ""
        if first:
            for m in first.input_messages:
                if str(m.get("role")) == "system":
                    system = str(m.get("content", ""))
                    break
        prompt = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": traj.task}
        ]
        rows[key] = {
            "prompt": prompt,
            "task": traj.task,
            "task_id": traj.task_id,
            "domain": traj.domain,
            "verifier": verifier,
            "reference": reference,
            "tool_catalog": list(tool_catalog or []),
        }
    return list(rows.values())


# ---------------------------------------------------------------------------
# Admission (verified gate + best-of-n)
# ---------------------------------------------------------------------------

def _admit(trajectories: Sequence[Trajectory], cfg: BuildConfig) -> List[Trajectory]:
    pool: List[Trajectory] = []
    for t in trajectories:
        if cfg.train_split_only and t.split == "heldout":
            continue
        if cfg.require_verified and not t.verified:
            continue
        if (t.outcome_reward or 0.0) < cfg.min_reward:
            continue
        pool.append(t)

    if cfg.max_per_task <= 0:
        return pool
    by_task: Dict[str, List[Trajectory]] = collections.defaultdict(list)
    for t in pool:
        by_task[t.task_id or _norm_text(t.task)].append(t)
    kept: List[Trajectory] = []
    for _key, group in by_task.items():
        group.sort(key=lambda t: (t.outcome_reward or 0.0, -t.n_steps), reverse=True)
        kept.extend(group[: cfg.max_per_task])
    return kept


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def dataset_stats(trajectories: Sequence[Trajectory], samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_domain = collections.Counter(t.domain or "unspecified" for t in trajectories)
    verified = [t for t in trajectories if t.verified]
    return {
        "trajectories": len(trajectories),
        "verified": len(verified),
        "verified_fraction": round(len(verified) / len(trajectories), 4) if trajectories else 0.0,
        "distinct_tasks": len({t.task_id or _norm_text(t.task) for t in trajectories}),
        "sft_samples": len(samples),
        "by_domain": dict(by_domain),
        "by_agent": dict(collections.Counter(s.get("agent", "") for s in samples)),
        "mean_steps": round(sum(t.n_steps for t in trajectories) / len(trajectories), 2) if trajectories else 0.0,
        "mean_reward_verified": round(sum(t.outcome_reward or 0 for t in verified) / len(verified), 4)
        if verified
        else 0.0,
    }


def _write_jsonl(rows: Sequence[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def build_datasets(
    trajectory_path: str,
    out_dir: str,
    cfg: Optional[BuildConfig] = None,
    *,
    references: Optional[Dict[str, Any]] = None,
    verifiers: Optional[Dict[str, str]] = None,
    default_verifier: str = "",
    tool_catalog: Optional[Sequence[str]] = None,
    weights: Optional[RewardWeights] = None,
) -> Dict[str, Any]:
    """Verify, filter and write ``sft.jsonl``, ``grpo.jsonl`` and ``stats.json``."""
    cfg = cfg or BuildConfig()
    if not cfg.heldout_tasks and not cfg.allow_no_heldout:
        raise ValueError(
            "No held-out task list supplied. Training on unfiltered trajectories risks "
            "contaminating the evaluation set; pass heldout_tasks=... or set "
            "allow_no_heldout=True to state explicitly that no evaluation set exists."
        )

    trajectories = list(iter_trajectories(trajectory_path))
    if references or verifiers or default_verifier:
        verify_trajectories(
            trajectories,
            references=references,
            verifiers=verifiers,
            default_verifier=default_verifier,
            tool_catalog=tool_catalog,
            weights=weights,
        )

    sft = build_sft_samples(trajectories, cfg)
    grpo = build_grpo_samples(trajectories, cfg, references=references, tool_catalog=tool_catalog)
    stats = dataset_stats(trajectories, sft)
    stats["grpo_prompts"] = len(grpo)
    stats["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}
    stats["config"]["heldout_tasks"] = len(cfg.heldout_tasks)  # store the count, not the tasks

    paths = {
        "sft": _write_jsonl(sft, os.path.join(out_dir, "sft.jsonl")),
        "grpo": _write_jsonl(grpo, os.path.join(out_dir, "grpo.jsonl")),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    paths["stats"] = os.path.join(out_dir, "stats.json")
    return {"paths": paths, "stats": stats}


def _load_json(path: str) -> Any:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build SFT/GRPO datasets from STELLA trajectories.")
    ap.add_argument("--trajectories", default=os.path.join(DATA_ROOT, "trajectories"))
    ap.add_argument("--out_dir", default=os.path.join(DATA_ROOT, "datasets"))
    ap.add_argument("--references", default="", help="JSON mapping task_id -> reference answer")
    ap.add_argument("--verifiers", default="", help="JSON mapping task_id -> verifier name")
    ap.add_argument("--default_verifier", default="")
    ap.add_argument("--heldout", default="", help="JSON list of held-out task strings (decontamination)")
    ap.add_argument("--allow_no_heldout", action="store_true")
    ap.add_argument("--min_reward", type=float, default=0.6)
    ap.add_argument("--max_per_task", type=int, default=2)
    ap.add_argument("--keep_agents", default="", help="comma-separated agent names to keep")
    args = ap.parse_args(argv)

    heldout = _load_json(args.heldout) or []
    cfg = BuildConfig(
        min_reward=args.min_reward,
        max_per_task=args.max_per_task,
        heldout_tasks=tuple(heldout),
        allow_no_heldout=args.allow_no_heldout,
        keep_agents=tuple(a.strip() for a in args.keep_agents.split(",") if a.strip()),
    )
    out = build_datasets(
        args.trajectories,
        args.out_dir,
        cfg,
        references=_load_json(args.references) or {},
        verifiers=_load_json(args.verifiers) or {},
        default_verifier=args.default_verifier,
    )
    print(json.dumps(out["stats"], indent=2))
    for k, v in out["paths"].items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
