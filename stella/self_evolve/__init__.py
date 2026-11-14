"""Parameter-level self-evolution for STELLA.

STELLA's published self-evolution operates on its *context*: it retrieves
thinking templates and writes new tools. Everything it has learned must be
re-retrieved on every run, and nothing it has learned changes the policy that
drives it. This package adds the complementary mechanism: STELLA's own verified
execution history is used to post-train an open-weight policy (Qwen, DeepSeek)
with SFT and GRPO, so experience is written into parameters and improves future
tasks without retrieval.

Pipeline::

    trajectory_logger   capture full trajectories from the running agents
    reward              score them with explicit verifiers (the admission gate)
    dataset_builder     verified trajectories -> SFT + GRPO datasets
    train_sft           behaviour-clone verified decisions (LoRA)
    train_grpo          optimise against verifiable scientific outcomes
    transfer            ScienceRSI: specialise -> verify -> transfer -> reuse
    evaluate            held-out, verifier-scored comparison of two policies
    evolve              one round end to end, with a promotion gate
    serve               serve the evolved policy back into STELLA

Heavy dependencies (torch, trl, peft, vllm) are imported only inside the
training/serving entry points, so ``import stella.self_evolve`` stays cheap and the
data pipeline runs on a machine with no GPU.
"""

from __future__ import annotations

from .dataset_builder import BuildConfig, build_datasets, build_grpo_samples, build_sft_samples, dataset_stats
from .reward import (
    RewardResult,
    RewardWeights,
    available_verifiers,
    register_verifier,
    score_trajectory,
    verify_trajectories,
)
from .schema import (
    DATA_ROOT,
    SCHEMA_VERSION,
    StepRecord,
    ToolCallRecord,
    Trajectory,
    append_trajectory,
    iter_trajectories,
    load_trajectories,
    new_trajectory_id,
    write_trajectories,
)
from .trajectory_logger import extract_trajectory, instrument, log_run, logging_enabled

__all__ = [
    "SCHEMA_VERSION",
    "DATA_ROOT",
    # schema
    "Trajectory",
    "StepRecord",
    "ToolCallRecord",
    "new_trajectory_id",
    "append_trajectory",
    "iter_trajectories",
    "load_trajectories",
    "write_trajectories",
    # logging
    "log_run",
    "instrument",
    "extract_trajectory",
    "logging_enabled",
    # reward
    "RewardWeights",
    "RewardResult",
    "score_trajectory",
    "verify_trajectories",
    "register_verifier",
    "available_verifiers",
    # datasets
    "BuildConfig",
    "build_sft_samples",
    "build_grpo_samples",
    "build_datasets",
    "dataset_stats",
]

__version__ = "0.1.0"
