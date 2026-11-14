#!/usr/bin/env bash
set -euo pipefail
recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Configure Accelerate for your hardware first. QLoRA uses one local device per process.
accelerate launch -m stella_rsi.train_sft --config "$recipe_root/configs/sft_qwen3.8_27b.json"
accelerate launch -m stella_rsi.train_grpo --config "$recipe_root/configs/grpo_qwen3.8_27b.json"
