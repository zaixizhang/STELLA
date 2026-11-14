"""Capture full execution trajectories from STELLA's smolagents agents.

STELLA already records *that* a run happened (AutoMemory) and *what workflow*
it followed (SkillManager/SkillStore). Neither keeps the token-level material
needed to train a policy: the exact messages the model conditioned on and the
exact text it produced. This module harvests that from the agent's own memory
after a run, so it adds no overhead to the run itself and cannot change the
agent's behaviour.

Usage (non-invasive, opt-in)::

    from stella.self_evolve.trajectory_logger import log_run

    result = manager_agent.run(task)
    log_run(manager_agent, task, result, domain="immunology")

or wrap the agent once and forget about it::

    from stella.self_evolve.trajectory_logger import instrument

    instrument(manager_agent, domain="immunology")
    result = manager_agent.run(task)   # logged automatically

Set ``STELLA_TRAJECTORY_LOGGING=0`` to disable logging globally.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Dict, List, Optional

from .schema import (
    StepRecord,
    ToolCallRecord,
    Trajectory,
    append_trajectory,
    new_trajectory_id,
)

__all__ = ["logging_enabled", "extract_trajectory", "log_run", "instrument"]


def logging_enabled() -> bool:
    return os.environ.get("STELLA_TRAJECTORY_LOGGING", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# smolagents -> Trajectory
# ---------------------------------------------------------------------------

def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Normalise a smolagents ChatMessage (or dict) to a JSON-safe dict.

    smolagents message content may be a plain string or a list of content
    blocks ({"type": "text", "text": ...}); both are flattened to text, and
    image blocks are dropped (they cannot be used for text-only SFT).
    """
    if msg is None:
        return {}
    if isinstance(msg, dict):
        role, content, tool_calls = msg.get("role"), msg.get("content"), msg.get("tool_calls")
    else:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)

    role = getattr(role, "value", role)  # MessageRole enum -> str

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
            elif block is not None:
                parts.append(str(block))
        content = "\n".join(parts)

    out: Dict[str, Any] = {"role": str(role) if role is not None else "user", "content": content or ""}

    if tool_calls:
        serialised = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                serialised.append(tc)
            else:
                fn = getattr(tc, "function", None)
                serialised.append(
                    {
                        "id": getattr(tc, "id", "") or "",
                        "name": getattr(fn, "name", "") or getattr(tc, "name", "") or "",
                        "arguments": getattr(fn, "arguments", None)
                        if fn is not None
                        else getattr(tc, "arguments", None),
                    }
                )
        out["tool_calls"] = serialised
    return out


def _tool_calls_from_step(step: Any) -> List[ToolCallRecord]:
    records: List[ToolCallRecord] = []
    for tc in getattr(step, "tool_calls", None) or []:
        name = getattr(tc, "name", "") or ""
        args = getattr(tc, "arguments", None)
        if not isinstance(args, (str, int, float, bool, list, dict, type(None))):
            args = str(args)
        records.append(
            ToolCallRecord(
                name=str(name),
                arguments=args,
                call_id=str(getattr(tc, "id", "") or ""),
                ok=not bool(getattr(step, "error", None)),
                error=str(getattr(step, "error", "") or "")[:2000],
            )
        )
    return records


def _step_duration(step: Any) -> float:
    timing = getattr(step, "timing", None)
    for attr in ("duration", "elapsed", "total_duration"):
        v = getattr(timing, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    start, end = getattr(timing, "start_time", None), getattr(timing, "end_time", None)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return float(end - start)
    return 0.0


def extract_trajectory(
    agent: Any,
    task: str,
    result: Any = None,
    *,
    agent_name: str = "",
    **meta: Any,
) -> Trajectory:
    """Build a Trajectory from an agent's post-run memory.

    Only ``ActionStep``s carry model input/output, so planning and task steps
    are skipped. Sub-agent (managed agent) memories are harvested too when the
    agent exposes them, each tagged with its own agent name.
    """
    traj = Trajectory(
        trajectory_id=new_trajectory_id(),
        task=task,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        policy_model=_policy_name(agent),
    )
    for k, v in meta.items():
        if hasattr(traj, k) and v is not None:
            setattr(traj, k, v)

    name = agent_name or getattr(agent, "name", None) or type(agent).__name__
    traj.steps = _steps_from_agent(agent, str(name))

    # Managed sub-agents keep their own memories; include them so the Dev and
    # Critic policies can be trained from the same run.
    managed = getattr(agent, "managed_agents", None) or {}
    if isinstance(managed, dict):
        managed = managed.values()
    for sub in managed or []:
        try:
            traj.steps.extend(_steps_from_agent(sub, str(getattr(sub, "name", "") or type(sub).__name__)))
        except Exception as e:  # noqa: BLE001
            print(f"[self_evolve] could not harvest sub-agent memory: {e}")

    traj.final_answer = "" if result is None else str(result)
    traj.duration_sec = sum(s.duration_sec for s in traj.steps)
    # The agent's own view of success: it produced a final answer and the last
    # step did not error. A verifier may overrule this (see reward.py).
    traj.success = bool(traj.final_answer) and not (traj.steps and traj.steps[-1].error)
    return traj


def _policy_name(agent: Any) -> str:
    model = getattr(agent, "model", None)
    for attr in ("model_id", "model_name", "id"):
        v = getattr(model, attr, None)
        if v:
            return str(v)
    return type(model).__name__ if model is not None else ""


def _steps_from_agent(agent: Any, agent_name: str) -> List[StepRecord]:
    memory = getattr(agent, "memory", None)
    steps = list(getattr(memory, "steps", None) or [])
    out: List[StepRecord] = []
    for step in steps:
        if not hasattr(step, "model_output") and not hasattr(step, "model_output_message"):
            continue  # TaskStep / PlanningStep: no policy decision to learn from
        usage = getattr(step, "token_usage", None)
        out.append(
            StepRecord(
                step_number=int(getattr(step, "step_number", len(out) + 1) or len(out) + 1),
                agent=agent_name,
                input_messages=[_message_to_dict(m) for m in (getattr(step, "model_input_messages", None) or [])],
                output_text=str(getattr(step, "model_output", "") or ""),
                output_message=_message_to_dict(getattr(step, "model_output_message", None)) or None,
                tool_calls=_tool_calls_from_step(step),
                observations=str(getattr(step, "observations", "") or "")[:20000],
                error=str(getattr(step, "error", "") or "")[:4000],
                code_action=str(getattr(step, "code_action", "") or ""),
                is_final_answer=bool(getattr(step, "is_final_answer", False)),
                duration_sec=_step_duration(step),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def log_run(agent: Any, task: str, result: Any = None, *, path: str = "", **meta: Any) -> Optional[str]:
    """Harvest and append one trajectory. Returns the file written, or None.

    Never raises: a logging failure must not fail a scientific run.
    """
    if not logging_enabled():
        return None
    try:
        traj = extract_trajectory(agent, task, result, **meta)
        if not traj.steps:
            return None
        return append_trajectory(traj, path=path)
    except Exception as e:  # noqa: BLE001
        print(f"[self_evolve] trajectory logging failed: {e}")
        return None


def instrument(agent: Any, **meta: Any) -> Any:
    """Wrap ``agent.run`` so every run is logged. Idempotent."""
    if getattr(agent, "_self_evolve_instrumented", False):
        return agent
    original_run = agent.run

    @functools.wraps(original_run)
    def run_and_log(task: str, *args: Any, **kwargs: Any) -> Any:
        result = original_run(task, *args, **kwargs)
        log_run(agent, task, result, **meta)
        return result

    agent.run = run_and_log
    agent._self_evolve_instrumented = True
    return agent
