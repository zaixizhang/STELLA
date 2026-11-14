"""Stage 2 of parameter-level self-evolution: GRPO against verifiable rewards.

SFT clones decisions that worked. GRPO goes further: the policy proposes
several solutions to the same task, each is scored by a verifier, and the
relative advantage within the group updates the weights. This is the
strategy-level optimisation step -- the policy is optimised against the
*outcome* of a scientific task, not against the surface form of a past
solution.

Rewards come from :mod:`stella.self_evolve.reward`, so the same verifiers that gate
the SFT data also define the objective here. Two reward regimes are provided:

``--mode response`` (default)
    The policy answers the task in one shot and the verifier scores the
    answer, plus shaping terms for tool-call validity and format. Cheap, fully
    on-policy, no external tool calls during training.

``--mode agentic``
    Each rollout is executed as a real STELLA run through a user-supplied
    rollout function, and the verifier scores the run's final answer. This
    optimises the policy in the environment it will actually be used in, at
    the cost of live tool calls per rollout.

    python -m stella.self_evolve.train_grpo \
        --model checkpoints/stella-sft-qwen3-8b \
        --dataset self_evolve_data/datasets/grpo.jsonl \
        --output_dir checkpoints/stella-grpo-qwen3-8b

Requires ``pip install trl`` (>=0.12); ``--use_vllm`` additionally needs vllm.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from .reward import RewardWeights, _VERIFIERS  # noqa: PLC2701 - verifier registry is the public contract

__all__ = ["make_reward_functions", "load_grpo_rows", "main"]


# ---------------------------------------------------------------------------
# Completion helpers
# ---------------------------------------------------------------------------

def _completion_text(completion: Any) -> str:
    """TRL hands back either a string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            str(m.get("content", "")) for m in completion if isinstance(m, dict) and m.get("content")
        )
    return str(completion or "")


_TOOL_CALL_PATTERNS = (
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),  # fenced JSON action
    re.compile(r'"name"\s*:\s*"([a-zA-Z_][\w]*)"'),  # tool-calling agent action
    re.compile(r"\b([a-z_][\w]*)\s*\(", re.MULTILINE),  # code agent call
)


def extract_tool_names(text: str) -> List[str]:
    """Best-effort recovery of the tools a completion tried to call."""
    names: List[str] = []
    for m in _TOOL_CALL_PATTERNS[0].finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            names.append(obj["name"])
    names.extend(_TOOL_CALL_PATTERNS[1].findall(text))
    if not names:
        names.extend(n for n in _TOOL_CALL_PATTERNS[2].findall(text) if not n.startswith("print"))
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Reward functions (TRL signature: fn(completions, **dataset_columns) -> list[float])
# ---------------------------------------------------------------------------

def outcome_reward(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Verifier score of each completion. The only reward that can be decisive."""
    references = kwargs.get("reference") or []
    verifiers = kwargs.get("verifier") or []
    tasks = kwargs.get("task") or []
    out: List[float] = []
    for i, c in enumerate(completions):
        name = verifiers[i] if i < len(verifiers) else ""
        fn = _VERIFIERS.get(str(name))
        if fn is None:
            out.append(0.0)
            continue
        ref = references[i] if i < len(references) else None
        task = tasks[i] if i < len(tasks) else ""
        try:
            score, _detail = fn(_completion_text(c), ref, task)
        except Exception:  # noqa: BLE001 - a broken verifier scores zero, never crashes training
            score = 0.0
        out.append(float(score))
    return out


def tool_validity_reward(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Fraction of named tools that exist in the Tool Ocean catalogue.

    Hallucinated tool names are the most common failure in agent rollouts and
    are free to check, so they are penalised directly rather than waiting for
    the outcome signal to carry the blame.
    """
    catalogs = kwargs.get("tool_catalog") or []
    out: List[float] = []
    for i, c in enumerate(completions):
        catalog = set(catalogs[i]) if i < len(catalogs) and catalogs[i] else set()
        names = extract_tool_names(_completion_text(c))
        if not names:
            out.append(0.0 if catalog else 0.5)  # no tool call at all is not a solution
            continue
        if not catalog:
            out.append(0.5)
            continue
        out.append(sum(1 for n in names if n in catalog) / len(names))
    return out


def format_reward(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Rewards a parseable action and a stated final answer."""
    out: List[float] = []
    for c in completions:
        text = _completion_text(c)
        score = 0.0
        if text.strip():
            score += 0.3
        if re.search(r"final_answer|Final answer|```", text):
            score += 0.3
        if extract_tool_names(text):
            score += 0.4
        out.append(min(score, 1.0))
    return out


def make_agentic_reward(rollout_fn: Callable[[str, str], str]) -> Callable[..., List[float]]:
    """Reward that executes each completion as a real STELLA run.

    ``rollout_fn(task, completion_text) -> final_answer`` is supplied by the
    caller (see ``stella.self_evolve.evolve.make_stella_rollout``); it runs the agent
    with the candidate policy and returns the answer, which the task's verifier
    then scores. This is the honest version of the objective -- the policy is
    rewarded for completing the science, not for producing text that looks like
    a solution -- and it costs one agent run per rollout.
    """

    def agentic_reward(completions: Sequence[Any], **kwargs: Any) -> List[float]:
        references = kwargs.get("reference") or []
        verifiers = kwargs.get("verifier") or []
        tasks = kwargs.get("task") or []
        out: List[float] = []
        for i, c in enumerate(completions):
            task = tasks[i] if i < len(tasks) else ""
            try:
                answer = rollout_fn(task, _completion_text(c))
            except Exception as e:  # noqa: BLE001 - a failed rollout is a zero, not a crash
                print(f"[grpo] rollout failed: {e}")
                out.append(0.0)
                continue
            fn = _VERIFIERS.get(str(verifiers[i] if i < len(verifiers) else ""))
            if fn is None:
                out.append(0.0)
                continue
            ref = references[i] if i < len(references) else None
            try:
                score, _ = fn(answer, ref, task)
            except Exception:  # noqa: BLE001
                score = 0.0
            out.append(float(score))
        return out

    return agentic_reward


def make_reward_functions(
    mode: str = "response",
    rollout_fn: Optional[Callable[[str, str], str]] = None,
) -> tuple:
    """Reward functions and their weights for the requested mode."""
    if mode == "agentic":
        if rollout_fn is None:
            raise ValueError("--mode agentic needs a rollout function (see stella.self_evolve.evolve)")
        return ([make_agentic_reward(rollout_fn), format_reward], [1.0, 0.1])
    w = RewardWeights()
    return ([outcome_reward, tool_validity_reward, format_reward], [w.outcome, w.tool_validity, w.format])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_grpo_rows(path: str) -> List[Dict[str, Any]]:
    """Read grpo.jsonl, keeping prompt plus the columns the rewards consume."""
    keep = ("prompt", "task", "task_id", "verifier", "reference", "tool_catalog")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj.get("prompt"), list):
                raise ValueError(f"{path}:{lineno}: expected a conversational 'prompt' list")
            if not obj.get("verifier"):
                continue  # unverifiable prompts cannot define a reward
            rows.append({k: obj.get(k) for k in keep if k in obj})
    return rows


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GRPO a STELLA policy against verifiable scientific rewards.")
    ap.add_argument("--model", required=True, help="base or SFT checkpoint to start from")
    ap.add_argument("--dataset", required=True, help="grpo.jsonl from stella.self_evolve.dataset_builder")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mode", default="response", choices=["response", "agentic"])

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_generations", type=int, default=8, help="group size G")
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient to the reference policy")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--full_finetune", action="store_true")
    ap.add_argument("--use_vllm", action="store_true", help="generate rollouts with vLLM (much faster)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--min_prompts", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rows = load_grpo_rows(args.dataset)
    print(f"[grpo] {args.dataset}: {len(rows)} verifiable prompts")
    by_verifier: Dict[str, int] = {}
    for r in rows:
        by_verifier[str(r.get("verifier"))] = by_verifier.get(str(r.get("verifier")), 0) + 1
    print(f"[grpo] verifiers: {json.dumps(by_verifier)}")
    if len(rows) < args.min_prompts:
        raise SystemExit(
            f"[grpo] only {len(rows)} verifiable prompts (min {args.min_prompts}). GRPO on a tiny "
            f"prompt set collapses onto those tasks; collect more verified trajectories first."
        )
    if args.dry_run:
        reward_funcs, weights = make_reward_functions("response")
        sample = [{"role": "assistant", "content": "query_pubmed(query='BTN3A1 NK cell AML')"}]
        print("[grpo] reward smoke test:", [f([sample], **{k: [v] for k, v in rows[0].items()}) for f in reward_funcs])
        print("[grpo] dry run: data and rewards validated, no training performed.")
        return 0

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    rollout_fn = None
    if args.mode == "agentic":
        from .evolve import make_stella_rollout  # noqa: PLC0415 - optional, pulls in stella.core

        rollout_fn = make_stella_rollout()
    reward_funcs, reward_weights = make_reward_functions(args.mode, rollout_fn)

    peft_config = None
    if not args.full_finetune:
        from peft import LoraConfig  # noqa: PLC0415

        from .train_sft import DEFAULT_LORA_TARGETS  # noqa: PLC0415

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,  # policy-gradient updates are noisy enough without dropout
            target_modules=list(DEFAULT_LORA_TARGETS),
            bias="none",
            task_type="CAUSAL_LM",
        )

    bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        reward_weights=reward_weights,
        logging_steps=1,
        save_strategy="epoch",
        bf16=bf16,
        use_vllm=args.use_vllm,
        seed=args.seed,
        report_to=[],
        model_init_kwargs={"attn_implementation": args.attn, "dtype": "bfloat16" if bf16 else "auto"},
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=Dataset.from_list(rows),
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    with open(os.path.join(args.output_dir, "self_evolve_grpo.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "start_model": args.model,
                "dataset": os.path.abspath(args.dataset),
                "mode": args.mode,
                "prompts": len(rows),
                "verifiers": by_verifier,
                "num_generations": args.num_generations,
                "beta": args.beta,
                "lr": args.lr,
            },
            f,
            indent=2,
        )
    print(f"[grpo] done -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
