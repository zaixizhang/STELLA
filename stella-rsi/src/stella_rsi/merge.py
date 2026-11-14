"""Export the final adapter as a standalone, text-only model for inference."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", help="Defaults to the base revision recorded in run_manifest.json")
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        raise ValueError("Export directory must be empty")
    manifest_path = Path(args.adapter) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    revision = args.revision or manifest.get("resolved_model_revision") or manifest.get("recipe", {}).get("revision")
    if not revision:
        raise ValueError("Adapter has no recorded base revision; pass --revision explicitly")
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = PeftConfig.from_pretrained(args.adapter)
    # Load unquantized weights to merge a QLoRA adapter. This needs enough host RAM.
    model = AutoModelForCausalLM.from_pretrained(adapter.base_model_name_or_path,
                                               revision=revision, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
