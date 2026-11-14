"""Stage 1 of parameter-level self-evolution: SFT on verified trajectories.

Behaviour cloning of STELLA's own verified decisions into an open-weight model
(Qwen or DeepSeek). This internalises *what tool to call next and how* -- the
part of the agent's competence that currently lives in retrieved templates and
has to be re-retrieved on every run.

Only ``stella.self_evolve.dataset_builder`` output is consumed, so the verified gate
and decontamination filter are always upstream of the optimiser.

    python -m stella.self_evolve.train_sft \
        --model Qwen/Qwen3-8B \
        --dataset self_evolve_data/datasets/sft.jsonl \
        --output_dir checkpoints/stella-sft-qwen3-8b

Requires ``pip install trl`` (>=0.12) alongside transformers/peft/accelerate.
LoRA is the default: a rank-32 adapter on an 8B model trains on a single 80GB
card and keeps each evolution round small enough to archive next to the
trajectories that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT an open-weight policy on verified STELLA trajectories.")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF id, e.g. Qwen/Qwen3-8B or deepseek-ai/DeepSeek-V3.1-Terminus")
    ap.add_argument("--dataset", required=True, help="sft.jsonl from stella.self_evolve.dataset_builder")
    ap.add_argument("--eval_dataset", default="", help="optional held-out sft.jsonl")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=16384)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--full_finetune", action="store_true", help="disable LoRA (needs much more memory)")
    ap.add_argument("--merge_adapter", action="store_true", help="also save a merged full-weight copy")

    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    ap.add_argument("--min_samples", type=int, default=64, help="refuse to train on less than this")
    ap.add_argument("--dry_run", action="store_true", help="validate data and config, then exit")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sft_rows(path: str) -> List[Dict[str, Any]]:
    """Read sft.jsonl and keep only the columns TRL needs."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt, completion = obj.get("prompt"), obj.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list):
                raise ValueError(f"{path}:{lineno}: expected conversational prompt/completion lists")
            if not prompt or not completion:
                continue
            rows.append({"prompt": prompt, "completion": completion})
    return rows


def describe(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    p_chars = [sum(len(m.get("content", "")) for m in r["prompt"]) for r in rows]
    c_chars = [sum(len(m.get("content", "")) for m in r["completion"]) for r in rows]
    n = max(len(rows), 1)
    return {
        "samples": len(rows),
        "mean_prompt_chars": round(sum(p_chars) / n),
        "max_prompt_chars": max(p_chars) if p_chars else 0,
        "mean_completion_chars": round(sum(c_chars) / n),
        "max_completion_chars": max(c_chars) if c_chars else 0,
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rows = load_sft_rows(args.dataset)
    stats = describe(rows)
    print(f"[sft] {args.dataset}: {json.dumps(stats)}")
    if len(rows) < args.min_samples:
        raise SystemExit(
            f"[sft] only {len(rows)} samples (min {args.min_samples}). Collect more verified "
            f"trajectories before training; a policy fitted to a handful of runs will not transfer."
        )
    if args.dry_run:
        print("[sft] dry run: data and config validated, no training performed.")
        return 0

    import torch  # noqa: PLC0415 - heavy imports stay out of module import
    from datasets import Dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    train_ds = Dataset.from_list(rows)
    eval_ds = Dataset.from_list(load_sft_rows(args.eval_dataset)) if args.eval_dataset else None

    peft_config = None
    if not args.full_finetune:
        from peft import LoraConfig  # noqa: PLC0415

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(DEFAULT_LORA_TARGETS),
            bias="none",
            task_type="CAUSAL_LM",
        )

    bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,  # prompt/completion pairs: never pack across samples
        completion_only_loss=True,  # condition on tool output, train on decisions
        seed=args.seed,
        report_to=[],
        model_init_kwargs={
            "attn_implementation": args.attn,
            "dtype": "bfloat16" if bf16 else "auto",
        },
    )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    try:
        trainer.processing_class.save_pretrained(args.output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[sft] could not save tokenizer: {e}")

    with open(os.path.join(args.output_dir, "self_evolve_sft.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.model,
                "dataset": os.path.abspath(args.dataset),
                "data_stats": stats,
                "lora": None if args.full_finetune else {"r": args.lora_r, "alpha": args.lora_alpha},
                "epochs": args.epochs,
                "lr": args.lr,
            },
            f,
            indent=2,
        )

    if args.merge_adapter and peft_config is not None:
        merged_dir = os.path.join(args.output_dir, "merged")
        print(f"[sft] merging adapter into base weights -> {merged_dir}")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        try:
            trainer.processing_class.save_pretrained(merged_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[sft] could not save tokenizer with merged model: {e}")

    print(f"[sft] done -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
