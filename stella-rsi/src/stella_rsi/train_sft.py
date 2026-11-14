"""Stage 1: assistant-decision SFT; user messages and tool observations are masked."""

import json

from .config import load_config, save_manifest, validate_training, validated_rows
from .messages import render_pair


def render_dataset(rows, tokenizer, max_length):
    rendered = []
    for row in rows:
        pair = render_pair(tokenizer, row)
        if len(tokenizer(pair["prompt"] + pair["completion"], add_special_tokens=False)["input_ids"]) <= max_length:
            rendered.append(pair)
    if not rendered:
        raise ValueError("No SFT examples fit max_length; increase it or shorten conversations explicitly")
    return rendered


def main(argv=None):
    config, args = load_config("sft", argv)
    rows = validated_rows(config["dataset"], "sft")
    eval_rows = validated_rows(config["eval_dataset"], "sft") if config.get("eval_dataset") else None
    validate_training(config, "sft")
    print(json.dumps({"stage": "sft", "model": config["model"], "samples": len(rows), "dry_run": args.dry_run}))
    if args.dry_run:
        return 0
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from .modeling import load_policy

    model, tokenizer = load_policy(config, "sft")
    maximum = config.get("max_length", 4096)
    train = render_dataset(rows, tokenizer, maximum)
    evaluation = render_dataset(eval_rows, tokenizer, maximum) if eval_rows else None
    print(json.dumps({"kept": len(train), "overlength_dropped": len(rows) - len(train),
                      "eval_kept": len(evaluation) if evaluation else 0}))
    settings = dict(config["training"])
    settings.update(output_dir=config["output_dir"], max_length=maximum,
                    completion_only_loss=True, packing=False, report_to="none", push_to_hub=False)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=SFTConfig(**settings),
                         train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(evaluation) if evaluation else None)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(config["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config["output_dir"])
    save_manifest(config, trainer, "sft", {"input": len(rows), "kept": len(train),
                                          "overlength_dropped": len(rows) - len(train)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
