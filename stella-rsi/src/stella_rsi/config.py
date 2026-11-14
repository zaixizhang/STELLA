"""Explicit JSON recipes, with paths resolved relative to the recipe file."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from . import DEFAULT_MODEL
from .data import read_jsonl
from .messages import normalize_messages
from .rewards import validate_reference


def load_config(stage, argv=None):
    parser = argparse.ArgumentParser(description=f"STELLA-RSI {stage.upper()}")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--adapter")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = json.loads(path.read_text())
    allowed = {"model", "revision", "dataset", "eval_dataset", "output_dir", "adapter",
               "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout", "target_modules",
               "max_length", "max_prompt_length", "training", "reward", "attn_implementation"}
    if set(config) - allowed:
        raise ValueError(f"Unknown recipe keys: {sorted(set(config) - allowed)}")
    config.setdefault("model", DEFAULT_MODEL)
    config.setdefault("revision", "main")
    config.setdefault("training", {})
    for key in ("dataset", "eval_dataset", "output_dir", "adapter"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    for key in ("model", "dataset", "output_dir", "adapter"):
        value = getattr(args, key, None)
        if value:
            config[key] = str(Path(value).resolve()) if key != "model" else value
    if args.model:
        # The recipe's pinned revision belongs to its default model only.
        config["revision"] = "main"
    for key in ("dataset", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing {key}")
    if stage == "grpo" and not config.get("adapter"):
        raise ValueError("GRPO requires the SFT adapter; set adapter in the recipe or --adapter")
    return config, args


def validated_rows(path, stage):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    for row in rows:
        normalize_messages(row["prompt"])
        if stage == "sft":
            messages = normalize_messages(row["completion"])
            if any(m["role"] != "assistant" for m in messages):
                raise ValueError("Only assistant decisions can be SFT targets")
        else:
            validate_reference(row["verifier"], json.loads(row["reference_json"]))
    return rows


def validate_training(config, stage):
    training = config["training"]
    batch = training.get("per_device_train_batch_size", 1)
    accum = training.get("gradient_accumulation_steps", 1)
    if batch < 1 or accum < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if stage == "grpo":
        n = training.get("num_generations", 4)
        generation_batch = training.get("generation_batch_size", batch * accum * int(os.getenv("WORLD_SIZE", "1")))
        if n < 2 or generation_batch % n:
            raise ValueError("GRPO needs >=2 generations and a generation batch divisible by num_generations")
    if config.get("load_in_4bit") and training.get("use_cpu"):
        raise ValueError("The QLoRA recipe requires CUDA; disable load_in_4bit for CPU smoke tests")


def save_manifest(config, trainer, stage, counts):
    if not trainer.is_world_process_zero():
        return
    versions = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        versions[name] = importlib.metadata.version(name)
    model = trainer.model
    resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    manifest = {"stage": stage, "recipe": config, "versions": versions, "counts": counts,
                "resolved_model_revision": resolved_revision,
                "dataset_sha256": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest()}
    (Path(config["output_dir"]) / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
