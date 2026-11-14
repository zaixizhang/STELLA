"""Optional offline integration test: local tiny model, real trainers, no API calls."""

import json
import os
from pathlib import Path

import pytest

from stella_rsi.config import load_config, validate_training
from stella_rsi.data import write_jsonl


def test_config_paths_and_group_size(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"model": "default/model", "revision": "pinned",
                                  "dataset": "train.jsonl", "output_dir": "output",
                                  "adapter": "sft", "training": {}}))
    config, _ = load_config("grpo", ["--config", str(recipe), "--model", "new/model", "--dry-run"])
    assert config["revision"] == "main"
    assert config["dataset"] == str(tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match="divisible"):
        validate_training(config, "grpo")
    config["training"] = {"num_generations": 2, "gradient_accumulation_steps": 2}
    validate_training(config, "grpo")


def alternating_test_reward(completions, **kwargs):
    # Deliberately nonconstant gradients to exercise the optimizer in this test.
    # This is NOT a task-quality reward and is never used by shipped recipes.
    return [float(i % 2) for i in range(len(completions))]


@pytest.mark.skipif(os.getenv("STELLA_RSI_TRAINING_TEST") != "1",
                    reason="Set STELLA_RSI_TRAINING_TEST=1 with the train extra installed")
@pytest.mark.parametrize("architecture", ["gpt2", "qwen3_5"])
def test_real_sft_then_grpo_continues_adapter(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from stella_rsi.modeling import load_policy
    from stella_rsi.train_sft import main as sft
    from stella_rsi.train_grpo import main as grpo

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model_dir = tmp_path / "model"
    vocab = {token: i for i, token in enumerate(
        ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>", "What", "is", "two", "plus", "?", "four", "4", "3"])}
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
                                        pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<' + m['role'] + '> ' + m['content'] + ' ' }}"
        "{% if m['role'] == 'assistant' %}{{ eos_token + ' ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")
    if architecture == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=2, n_embd=32,
                                          n_positions=128, bos_token_id=2, eos_token_id=2, pad_token_id=0))
    else:
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config={"vocab_size": len(vocab), "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                         "head_dim": 16, "layer_types": ["linear_attention", "full_attention"],
                         "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                         "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                         "bos_token_id": 2, "eos_token_id": 2, "pad_token_id": 0,
                         "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                             "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16} ))
    original_embeddings = model.get_input_embeddings().weight.detach().clone()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    prompt = [{"role": "user", "content": "What is two plus two ?"}]
    write_jsonl(tmp_path / "sft.jsonl", [{"prompt": prompt, "completion": [{"role": "assistant", "content": "four"}]}] * 2)
    write_jsonl(tmp_path / "grpo.jsonl", [{"prompt": prompt, "verifier": "numeric", "reference_json": "4"}] * 2)
    training = {"use_cpu": True, "bf16": False, "max_steps": 1, "learning_rate": 0.01,
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
                "save_strategy": "no", "logging_steps": 1, "disable_tqdm": True,
                "gradient_checkpointing": architecture == "qwen3_5",
                "gradient_checkpointing_kwargs": {"use_reentrant": False}, "seed": 42}
    config = {"model": str(model_dir), "revision": "main", "dataset": "sft.jsonl",
              "output_dir": "sft", "load_in_4bit": False, "lora_r": 2, "lora_alpha": 4,
              "lora_dropout": 0.0, "max_length": 64, "training": training}
    path = tmp_path / "sft-config.json"
    path.write_text(json.dumps(config))
    initial_config, _ = load_config("sft", ["--config", str(path)])
    initial, _ = load_policy(initial_config, "sft")
    assert torch.equal(original_embeddings, initial.get_input_embeddings().weight.cpu())
    if architecture == "qwen3_5":
        assert initial.config.model_type == "qwen3_5_text"
        assert not any("visual" in name for name, _ in initial.named_parameters())
    del initial
    assert sft(["--config", str(path)]) == 0
    saved = load_file(tmp_path / "sft/adapter_model.safetensors")
    assert any(t.abs().sum() > 0 for key, t in saved.items() if "lora_B" in key)
    config.update(dataset="grpo.jsonl", output_dir="grpo", adapter="sft", max_prompt_length=48,
                  reward=f"{__name__}:alternating_test_reward")
    config["training"] = {**training, "gradient_accumulation_steps": 2, "num_generations": 2,
                          "max_completion_length": 8, "beta": 0.0,
                          "mask_truncated_completions": False}
    path = tmp_path / "grpo-config.json"
    path.write_text(json.dumps(config))
    resolved, _ = load_config("grpo", ["--config", str(path)])
    continued, _ = load_policy(resolved, "grpo")
    restored = get_peft_model_state_dict(continued)
    assert saved.keys() == restored.keys()
    assert all(torch.equal(t, restored[key].cpu()) for key, t in saved.items())
    assert any(p.requires_grad for name, p in continued.named_parameters() if "lora_" in name)
    del continued
    with pytest.raises(ValueError, match="same base-model revision"):
        load_policy({**resolved, "revision": "different-revision"}, "grpo")
    assert grpo(["--config", str(path)]) == 0
    updated = load_file(tmp_path / "grpo/adapter_model.safetensors")
    assert any(not torch.equal(t, updated[key]) for key, t in saved.items())
    for stage in ("sft", "grpo"):
        manifest = json.loads((tmp_path / stage / "run_manifest.json").read_text())
        assert manifest["stage"] == stage and manifest["counts"]["kept"] == 2
    from stella_rsi.merge import main as merge
    from transformers import AutoModelForCausalLM
    merge(["--adapter", str(tmp_path / "grpo"), "--output-dir", str(tmp_path / "merged")])
    exported = AutoModelForCausalLM.from_pretrained(tmp_path / "merged")
    assert not any("lora_" in name for name, _ in exported.named_parameters())
    assert exported.config.model_type == ("qwen3_5_text" if architecture == "qwen3_5" else "gpt2")
