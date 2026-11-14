"""Load Qwen's text policy and continue the same adapter across both stages."""

import os
import json
from pathlib import Path


def load_policy(config, stage):
    import torch
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() and not config["training"].get("use_cpu"):
        raise RuntimeError("Training needs CUDA; use --dry-run for data/config validation")
    source = config["model"]
    adapter = config.get("adapter")
    if stage == "grpo":
        if not adapter or not (Path(adapter) / "adapter_config.json").is_file():
            raise ValueError("SFT adapter missing; finish SFT before starting GRPO")
        adapter_config = PeftConfig.from_pretrained(adapter)
        if adapter_config.base_model_name_or_path != source:
            raise ValueError("SFT adapter base model differs from the configured GRPO base model")
        manifest_path = Path(adapter) / "run_manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())["recipe"].get("revision", "main")
            if previous != config.get("revision", "main"):
                raise ValueError("SFT and GRPO must use the same base-model revision")
    bf16 = config["training"].get("bf16", True)
    kwargs = {"revision": config.get("revision", "main"),
              "dtype": torch.bfloat16 if bf16 else torch.float32,
              "attn_implementation": config.get("attn_implementation", "sdpa")}
    if config.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float32,
        )
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    # Transformers 5.14.1 maps Qwen3.5's composite qwen3_5 config to the
    # text-only Qwen3_5ForCausalLM and converts language_model weight prefixes.
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    tokenizer_source = adapter if adapter else source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, **({"revision": config.get("revision", "main")} if not adapter else {}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if stage == "grpo" else "right"
    model.config.use_cache = False
    if config.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if adapter:
        # Do not initialize a fresh GRPO adapter and discard what SFT learned.
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=config.get("lora_r", 32), lora_alpha=config.get("lora_alpha", 64),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules", "all-linear"),
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    return model, tokenizer
