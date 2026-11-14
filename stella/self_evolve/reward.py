"""Verifier-first reward model for STELLA's parameter-level self-evolution.

Nothing enters the training set because the agent believed it succeeded. Every
trajectory is scored by an explicit verifier, and only trajectories that a
verifier passes are allowed to shape parameters. This is the gate that keeps
self-evolution from amplifying its own mistakes: the reward is computed from an
outcome that can be checked outside the policy, not from the policy's own
confidence.

Reward components
-----------------
``outcome``        verifier score in [0, 1] -- the only component that can gate
``tool_validity``  fraction of tool calls that name a real tool and parse
``format``         the step produced a parseable action / final answer
``efficiency``     shorter solutions preferred, relative to a step budget
``error_free``     penalises steps that raised

Only ``outcome`` decides admission (``verified``); the others shape the scalar
reward used by GRPO and the ranking used for best-of-n distillation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import Trajectory

__all__ = [
    "RewardWeights",
    "RewardResult",
    "register_verifier",
    "available_verifiers",
    "score_trajectory",
    "verify_trajectories",
]


# ---------------------------------------------------------------------------
# Verifiers: callable(final_answer, reference, task) -> (score in [0,1], detail)
# ---------------------------------------------------------------------------

VerifierFn = Callable[[str, Any, str], Tuple[float, str]]
_VERIFIERS: Dict[str, VerifierFn] = {}


def register_verifier(name: str) -> Callable[[VerifierFn], VerifierFn]:
    """Register a custom verifier, e.g. a wet-lab or benchmark checker."""

    def deco(fn: VerifierFn) -> VerifierFn:
        _VERIFIERS[name] = fn
        return fn

    return deco


def available_verifiers() -> List[str]:
    return sorted(_VERIFIERS)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


@register_verifier("exact_match")
def _exact_match(answer: str, reference: Any, task: str = "") -> Tuple[float, str]:
    refs = reference if isinstance(reference, (list, tuple, set)) else [reference]
    a = _norm(answer)
    for r in refs:
        if a == _norm(r):
            return 1.0, "exact match"
    return 0.0, "no exact match"


@register_verifier("contains")
def _contains(answer: str, reference: Any, task: str = "") -> Tuple[float, str]:
    """Reference string(s) must appear in the answer. All must appear."""
    refs = reference if isinstance(reference, (list, tuple, set)) else [reference]
    refs = [r for r in refs if str(r).strip()]
    if not refs:
        return 0.0, "no reference given"
    a = _norm(answer)
    hits = [r for r in refs if _norm(r) in a]
    return len(hits) / len(refs), f"{len(hits)}/{len(refs)} references present"


@register_verifier("mcq")
def _mcq(answer: str, reference: Any, task: str = "") -> Tuple[float, str]:
    """Single multiple-choice letter, taken from the last A-E token."""
    letters = re.findall(r"\b([A-E])\b", str(answer).upper())
    if not letters:
        return 0.0, "no option letter found"
    return (1.0, f"chose {letters[-1]}") if letters[-1] == _norm(reference).upper() else (0.0, f"chose {letters[-1]}")


@register_verifier("numeric")
def _numeric(answer: str, reference: Any, task: str = "") -> Tuple[float, str]:
    """Last number in the answer, within 1% relative tolerance."""
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(answer).replace(",", ""))
    if not nums:
        return 0.0, "no number found"
    try:
        got, want = float(nums[-1]), float(reference)
    except (TypeError, ValueError):
        return 0.0, "unparseable reference"
    tol = abs(want) * 0.01 if want else 1e-6
    return (1.0, f"{got} ~= {want}") if abs(got - want) <= tol else (0.0, f"{got} != {want}")


@register_verifier("ranked_contains")
def _ranked_contains(answer: str, reference: Any, task: str = "") -> Tuple[float, str]:
    """Discovery tasks: a known target must appear, the earlier the better.

    ``reference`` is {"target": "BTN3A1", "top_k": 10} (or just a string). The
    answer is split into lines/rows and scored by the rank at which the target
    first appears, so a run that puts the validated target at rank 1 scores
    higher than one that buries it at rank 40. This is the verifier used for
    the NK-cell/AML and enzyme cases, where the ground truth is a wet-lab
    result obtained after the run.
    """
    if isinstance(reference, dict):
        target, top_k = reference.get("target", ""), int(reference.get("top_k", 10) or 10)
    else:
        target, top_k = reference, 10
    if not str(target).strip():
        return 0.0, "no target given"

    target_n = _norm(target)
    rows = [r for r in re.split(r"[\n\r]+", str(answer)) if r.strip()]
    for i, row in enumerate(rows):
        if re.search(rf"\b{re.escape(target_n)}\b", _norm(row)):
            if i >= top_k:
                return 0.25, f"{target} found at row {i + 1}, outside top {top_k}"
            return 1.0 - 0.5 * (i / max(top_k, 1)), f"{target} found at row {i + 1}"
    return 0.0, f"{target} absent"


@register_verifier("json_valid")
def _json_valid(answer: str, reference: Any, task: str = "") -> Tuple[float, str]:
    """Answer must be JSON; if reference is a list of keys, they must be present."""
    text = str(answer).strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not m:
        return 0.0, "no JSON found"
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return 0.0, f"invalid JSON: {e}"
    keys = reference if isinstance(reference, (list, tuple)) else []
    if keys:
        first = obj[0] if isinstance(obj, list) and obj else obj
        if not isinstance(first, dict):
            return 0.5, "JSON parsed but not an object/record list"
        missing = [k for k in keys if k not in first]
        if missing:
            return 0.5, f"JSON parsed, missing keys: {missing}"
    return 1.0, "valid JSON"


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    outcome: float = 1.0
    tool_validity: float = 0.2
    format: float = 0.1
    efficiency: float = 0.1
    error_free: float = 0.1
    step_budget: int = 12
    verify_threshold: float = 0.999  # outcome score required to admit for training


@dataclass
class RewardResult:
    total: float
    verified: bool
    outcome: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    detail: str = ""
    verifier: str = ""


def _tool_validity(traj: Trajectory, tool_catalog: Optional[Iterable[str]]) -> float:
    calls = [tc for s in traj.steps for tc in s.tool_calls]
    if not calls:
        return 1.0  # nothing to get wrong
    catalog = set(tool_catalog) if tool_catalog else None
    ok = 0
    for tc in calls:
        name_ok = (tc.name in catalog) if catalog is not None else bool(tc.name)
        args_ok = True
        if isinstance(tc.arguments, str) and tc.arguments.strip().startswith(("{", "[")):
            try:
                json.loads(tc.arguments)
            except json.JSONDecodeError:
                args_ok = False
        if name_ok and args_ok and not tc.error:
            ok += 1
    return ok / len(calls)


def _format_score(traj: Trajectory) -> float:
    if not traj.steps:
        return 0.0
    produced_output = sum(1 for s in traj.steps if s.output_text.strip())
    reached_final = any(s.is_final_answer for s in traj.steps) or bool(traj.final_answer)
    return 0.5 * (produced_output / len(traj.steps)) + 0.5 * float(reached_final)


def _efficiency(traj: Trajectory, budget: int) -> float:
    if traj.n_steps == 0:
        return 0.0
    return max(0.0, min(1.0, budget / max(traj.n_steps, 1)))


def _error_free(traj: Trajectory) -> float:
    if not traj.steps:
        return 0.0
    return 1.0 - sum(1 for s in traj.steps if s.error) / len(traj.steps)


def score_trajectory(
    traj: Trajectory,
    *,
    reference: Any = None,
    verifier: str = "",
    tool_catalog: Optional[Iterable[str]] = None,
    weights: Optional[RewardWeights] = None,
) -> RewardResult:
    """Score one trajectory. ``verified`` is driven only by the verifier.

    With no verifier and no reference, ``verified`` is None-equivalent (False)
    and the trajectory is *not* admitted to training -- an unverifiable run is
    treated as unusable, not as a success.
    """
    w = weights or RewardWeights()
    name = verifier or traj.verifier
    outcome, detail = 0.0, "no verifier: not admissible"
    if name:
        fn = _VERIFIERS.get(name)
        if fn is None:
            raise KeyError(f"unknown verifier {name!r}; available: {available_verifiers()}")
        try:
            outcome, detail = fn(traj.final_answer, reference, traj.task)
        except Exception as e:  # noqa: BLE001 - a broken verifier must not pass runs
            outcome, detail = 0.0, f"verifier error: {e}"

    breakdown = {
        "outcome": outcome,
        "tool_validity": _tool_validity(traj, tool_catalog),
        "format": _format_score(traj),
        "efficiency": _efficiency(traj, w.step_budget),
        "error_free": _error_free(traj),
    }
    total = (
        w.outcome * breakdown["outcome"]
        + w.tool_validity * breakdown["tool_validity"]
        + w.format * breakdown["format"]
        + w.efficiency * breakdown["efficiency"]
        + w.error_free * breakdown["error_free"]
    )
    denom = w.outcome + w.tool_validity + w.format + w.efficiency + w.error_free
    return RewardResult(
        total=total / denom if denom else 0.0,
        verified=bool(name) and outcome >= w.verify_threshold,
        outcome=outcome,
        breakdown=breakdown,
        detail=detail,
        verifier=name,
    )


def verify_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    references: Optional[Dict[str, Any]] = None,
    verifiers: Optional[Dict[str, str]] = None,
    default_verifier: str = "",
    tool_catalog: Optional[Iterable[str]] = None,
    weights: Optional[RewardWeights] = None,
) -> List[Trajectory]:
    """Annotate trajectories in place with verified / outcome_reward.

    ``references`` and ``verifiers`` are keyed by ``task_id`` (falling back to
    the task text), so an answer key can be kept in one file next to the
    trajectories and never inside them.
    """
    references = references or {}
    verifiers = verifiers or {}
    for traj in trajectories:
        key = traj.task_id or traj.task
        res = score_trajectory(
            traj,
            reference=references.get(key, references.get(traj.task)),
            verifier=verifiers.get(key, traj.verifier or default_verifier),
            tool_catalog=tool_catalog,
            weights=weights,
        )
        traj.verified = res.verified
        traj.outcome_reward = res.total
        traj.reward_breakdown = dict(res.breakdown)
        traj.verifier = res.verifier
    return list(trajectories)
