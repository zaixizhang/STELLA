#!/usr/bin/env bash
set -euo pipefail
recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python -m stella_rsi.data \
  --input "$recipe_root/examples/interactions.jsonl" \
  --labels "$recipe_root/examples/labels.json" \
  --splits "$recipe_root/examples/splits.json" \
  --output-dir "$recipe_root/data/processed"
python -m stella_rsi.train_sft --config "$recipe_root/configs/sft_qwen3.8_27b.json" --dry-run
python -m stella_rsi.train_grpo --config "$recipe_root/configs/grpo_qwen3.8_27b.json" --dry-run
