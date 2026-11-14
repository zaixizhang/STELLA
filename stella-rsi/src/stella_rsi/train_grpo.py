"""Stage 2: sample new next responses and optimize verifiable group-relative rewards."""

import importlib
import json

from .config import load_config, save_manifest, validate_training, validated_rows
from .messages import render_prompt
from .rewards import outcome_reward


def render_dataset(rows, tokenizer, max_prompt_length):
    rendered = []
    for row in rows:
        prompt = render_prompt(tokenizer, row["prompt"], row.get("tools"))
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= max_prompt_length:
            rendered.append({"prompt": prompt, "verifier": row["verifier"],
                             "reference_json": row["reference_json"]})
    if not rendered:
        raise ValueError("No GRPO prompts fit max_prompt_length")
    return rendered


def main(argv=None):
    config, args = load_config("grpo", argv)
    rows = validated_rows(config["dataset"], "grpo")
    validate_training(config, "grpo")
    reward = outcome_reward
    if config.get("reward"):
        module, name = config["reward"].split(":", 1)
        reward = getattr(importlib.import_module(module), name)
    print(json.dumps({"stage": "grpo", "model": config["model"], "prompts": len(rows),
                      "adapter": config["adapter"], "dry_run": args.dry_run}))
    if args.dry_run:
        print("Dry-run does not require a completed SFT adapter, download weights, or execute training.")
        return 0
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from .modeling import load_policy

    model, tokenizer = load_policy(config, "grpo")
    train = render_dataset(rows, tokenizer, config.get("max_prompt_length", 3072))
    print(json.dumps({"kept": len(train), "overlength_dropped": len(rows) - len(train)}))
    settings = dict(config["training"])
    settings.update(output_dir=config["output_dir"], report_to="none", push_to_hub=False)
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=GRPOConfig(**settings),
                          train_dataset=Dataset.from_list(train), reward_funcs=reward)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(config["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config["output_dir"])
    save_manifest(config, trainer, "grpo", {"input": len(rows), "kept": len(train),
                                           "overlength_dropped": len(rows) - len(train)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
