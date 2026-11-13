"""Data model for STELLA's parameter-level self-evolution.

A *trajectory* is one complete attempt at a scientific task: the task text, the
per-step model inputs/outputs, the tools that were called, the observations
returned, and the verified outcome. Trajectories are the raw material that is
distilled into model parameters (SFT + GRPO), so they are stored verbatim and
append-only: nothing is overwritten once written.

The on-disk format is JSON Lines, one trajectory per line, under
``self_evolve_data/trajectories/``. This is deliberately a flat file format
rather than the skill SQLite database: trajectories are large, written once,
read in bulk by the training jobs, and must remain inspectable by a reviewer
without running any STELLA code.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

SCHEMA_VERSION = "1.0"

DATA_ROOT = os.environ.get(
    "STELLA_SELF_EVOLVE_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "self_evolve_data"),
)
TRAJECTORY_DIR = os.path.join(DATA_ROOT, "trajectories")


@dataclass
class ToolCallRecord:
    """One tool invocation inside a step."""

    name: str
    arguments: Any = None
    call_id: str = ""
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepRecord:
    """One reasoning/acting step of one agent.

    ``input_messages`` is the exact message list the policy conditioned on and
    ``output_text`` is what it produced, which is what makes a step usable as a
    supervised example. ``tool_calls`` and ``observations`` are what the step
    actually did, which is what makes it scorable.
    """

    step_number: int
    agent: str = "manager"
    input_messages: List[Dict[str, Any]] = field(default_factory=list)
    output_text: str = ""
    output_message: Optional[Dict[str, Any]] = None
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    observations: str = ""
    error: str = ""
    code_action: str = ""
    is_final_answer: bool = False
    duration_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_calls"] = [tc.to_dict() if isinstance(tc, ToolCallRecord) else tc for tc in self.tool_calls]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord":
        d = dict(d)
        d["tool_calls"] = [ToolCallRecord(**tc) for tc in d.get("tool_calls", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Trajectory:
    """One complete task attempt, with its verified outcome."""

    trajectory_id: str
    task: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: str = ""

    # Outcome. ``success`` is the agent's own view; ``verified`` and
    # ``outcome_reward`` are filled in by a verifier (see reward.py) and are the
    # only signals allowed to gate training data.
    success: bool = False
    verified: Optional[bool] = None
    outcome_reward: Optional[float] = None
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    verifier: str = ""

    # Provenance.
    domain: str = ""
    skill_id: str = ""
    track_id: str = ""
    policy_model: str = ""
    benchmark: str = ""
    task_id: str = ""
    split: str = ""  # train / heldout — set by the caller, never inferred
    tags: List[str] = field(default_factory=list)

    duration_sec: float = 0.0
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_dict() if isinstance(s, StepRecord) else s for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["steps"] = [StepRecord.from_dict(s) for s in d.get("steps", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # -- convenience ------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_names(self) -> List[str]:
        return [tc.name for s in self.steps for tc in s.tool_calls]

    @property
    def had_error(self) -> bool:
        return any(s.error for s in self.steps)


def new_trajectory_id() -> str:
    return f"traj_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def shard_path(shard: str = "", data_dir: str = TRAJECTORY_DIR) -> str:
    """Path of the JSONL shard to append to (one shard per UTC day by default)."""
    shard = shard or time.strftime("%Y%m%d", time.gmtime())
    return os.path.join(data_dir, f"trajectories_{shard}.jsonl")


def append_trajectory(traj: Trajectory, path: str = "") -> str:
    """Append one trajectory as a single JSON line. Never overwrites."""
    path = path or shard_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(traj.to_dict(), ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def iter_trajectories(path_or_dir: str = TRAJECTORY_DIR) -> Iterator[Trajectory]:
    """Yield every trajectory under a file or directory, skipping bad lines.

    A corrupt line (a crash mid-write, a partially synced NFS file) is skipped
    with a warning rather than aborting a training run.
    """
    paths: List[str] = []
    if os.path.isdir(path_or_dir):
        for root, _dirs, files in os.walk(path_or_dir):
            paths.extend(os.path.join(root, fn) for fn in sorted(files) if fn.endswith(".jsonl"))
    elif os.path.isfile(path_or_dir):
        paths = [path_or_dir]

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Trajectory.from_dict(json.loads(line))
                except Exception as e:  # noqa: BLE001 - one bad line must not stop training
                    print(f"[self_evolve] skipping corrupt line {p}:{lineno}: {e}")


def load_trajectories(path_or_dir: str = TRAJECTORY_DIR) -> List[Trajectory]:
    return list(iter_trajectories(path_or_dir))


def write_trajectories(trajs: List[Trajectory], path: str) -> str:
    """Write a fresh JSONL file (used for curated/verified subsets)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")
    return path
