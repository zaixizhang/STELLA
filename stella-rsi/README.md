# STELLA-RSI

Fine-tune an open Qwen policy from **STELLA user interaction trajectories**:
reviewed assistant decisions → **SFT** → **GRPO** on newly generated responses.
The default is [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
with a pinned model revision in both recipes. This folder is an independently
installable package inside the STELLA repository.

This implementation trains a **text policy**, including tool calls and recorded
tool observations. SFT learns approved decisions; GRPO samples alternative next
responses to recorded prefixes and recomputes verifiable rewards. GRPO does not
execute tools or roll out complete live STELLA conversations. Images, audio, and
video are rejected rather than silently removed. There are no real user records
or trained checkpoints in this repository.

## Layout

```text
stella-rsi/
├── configs/                 # Pinned Qwen3.8-27B SFT and GRPO recipes
├── examples/                # Synthetic trajectories, labels, splits, collector
├── scripts/                 # Demo validation and sequential training
├── src/stella_rsi/
│   ├── capture.py           # Record new manager/sub-agent steps per user turn
│   ├── data.py              # Review filtering and session-based splits
│   ├── messages.py          # Chat-template and tool-call normalization
│   ├── rewards.py           # Verifiers for freshly sampled responses
│   ├── modeling.py         # Text-policy loading and adapter continuation
│   ├── train_sft.py         # Assistant-completion-only SFT
│   ├── train_grpo.py        # On-policy next-response GRPO
│   └── merge.py             # Export the final adapter into a text model
└── tests/                   # Data/reward tests and optional CPU training test
```

`data/` and `outputs/` are generated locally and ignored by Git.

## Install and validate without a GPU

Run the following from the **STELLA repository root**, using Python 3.11+:

```bash
python -m venv .venv-rsi
source .venv-rsi/bin/activate
python -m pip install -e './stella-rsi[test]'
bash stella-rsi/scripts/prepare_demo.sh
python -m pytest stella-rsi/tests -q
```

The demo creates 4 training SFT decisions and 5 training GRPO prompts, plus
validation/test files. It uses synthetic arithmetic examples to demonstrate
multi-turn context, tool calls, and a failed answer that remains usable as a
GRPO prompt. **These examples are fixtures, not a fine-tuning dataset.** Dry-run
checks records, paths, and generation-group sizes without downloading weights
or requiring an SFT adapter. It does not validate GPU capacity or run a trainer.

Install the training stack in this environment when ready:

```bash
# First install a CUDA-compatible PyTorch build for your machine.
python -m pip install -e './stella-rsi[train,qlora,test]'
```

Training dependencies pin Transformers 5.14.1, TRL 1.18.0, PEFT 0.20.0,
Accelerate 1.14.0, and Datasets 4.8.4. The pinned Transformers loader extracts
`Qwen3_5ForCausalLM` from Qwen3.8's composite checkpoint; vision weights are not
trained or exported. The default recipe uses 4-bit NF4 QLoRA and BF16 compute.
Use a supported CUDA GPU with BF16 support. Memory use depends on prompt lengths,
completion lengths, and generation batch size; a full 27B hardware run has not
been validated by this code's CPU integration test. Install into a dedicated
environment to avoid conflicts with STELLA's optional training dependencies.

## Collect or import interactions

For live collection, install STELLA in the collection environment, configure its
API credentials as described in the [main README](../README.md#api-keys), and run:

```bash
python -m pip install -e . -e ./stella-rsi
python stella-rsi/examples/collect_chat.py --output-dir stella-rsi/data/raw
```

Each invocation starts one conversation. `/quit` ends it. The recorder stores
only newly produced manager/sub-agent steps for each turn, preserving the full
model input context, output messages, tool-call arguments, and function schemas
for native tool-calling agents. All turns share a `session_id`. Capture is opt-in;
the existing STELLA web UI is not automatically instrumented. Integrations can
wrap a manager with `stella_rsi.capture.InteractionRecorder` and call
`recorder.run(task, reset=False)` for successive turns. A recorder belongs to one
conversation and should not be shared between concurrent sessions.

Alternatively, export JSONL records with this chat schema:

```json
{"record_id":"turn-1","session_id":"conversation-1","messages":[{"role":"user","content":"What is 3 * 7?"},{"role":"assistant","content":"21"}]}
```

`messages` can contain system messages, previous user/assistant turns, assistant
`tool_calls`, and tool messages with `tool_call_id`. An optional `tools` field
contains OpenAI-style function schemas. Each target's prompt is its entire
preceding conversation; assistant targets are never fabricated from task text.

Native STELLA logs are also accepted: `trajectory_id` or `record_id`, `session_id`,
and a `steps` list containing `input_messages` and `output_message` (or legacy
`output_text`). Each step can carry its own `tools`. Existing logs without
`session_id` must first be mapped to their actual conversations; assigning a
different session to every turn would allow leakage across splits. Error-marked
steps/records are excluded. Review and de-identify real interaction data before
training; the recorder preserves the text supplied to the model.

## Review, label rewards, and split

Create a labels JSON object keyed by record ID, for example:

```json
{
  "turn-1": {
    "approved_decisions": [1],
    "grpo_targets": [
      {"decision_index": 1, "verifier": "numeric", "reference": 21}
    ]
  }
}
```

Decision indices are **zero-based message positions** in chat records, or
**zero-based step positions** in native records. `approved_decisions` selects
reviewed assistant outputs for SFT. `sft_approved: true` approves every valid
decision in a record; use it only after reviewing all those decisions. Unlabelled
records never become training targets. A bad historical answer can supply a
GRPO prompt if its independently verified reference is available, without being
approved for SFT.

| Verifier | Reference | Reward on a new completion |
| --- | --- | --- |
| `exact_match` | Nonempty string | Case/whitespace-normalized exact match |
| `contains` | Nonempty string | Case/whitespace-normalized substring match |
| `numeric` | Number, or `{"value":21,"atol":0,"rtol":0.01}` | Last numeric value within tolerance |
| `tool_call` | `{"name":"mean","arguments":{"values":[2,4,6]}}` | Exactly one matching function call and arguments |

The tool verifier supports Qwen3.8's `<function=...>` / `<parameter=...>` format
and JSON tool calls. No generated code or tool call is executed by a verifier.
Text rewards ignore `<think>...</think>` blocks. The default generation prompt
disables thinking, while SFT preserves reasoning if the reviewed target contains
it. These simple verifiers suit tasks with explicit references: substring and
last-number checks are weak proxies for open-ended biomedical correctness.
Use a task-specific evaluator for those tasks. A recipe can set
`"reward": "your_package.rewards:score_batch"` for a TRL-compatible callable;
prepared rows still require valid built-in verifier/reference metadata. Never
reuse a historical user vote as the reward for a newly sampled answer.

Create session holdouts in `stella-rsi/data/splits.json`:

```json
{"validation":["validation-session-id"],"test":["test-session-id"]}
```

All remaining sessions are training sessions. At least one held-out session is
required. The default SFT recipe evaluates on validation data, so provide approved
validation examples as well. Preparation splits **before expanding decisions**,
excludes records sharing exact normalized user tasks with a held-out split, and
deduplicates identical SFT decisions within each split. Near duplicates and shared
documents still need dataset-specific review. Never tune on the test sessions.

```bash
python -m stella_rsi.data \
  --input stella-rsi/data/raw \
  --labels stella-rsi/data/labels.json \
  --splits stella-rsi/data/splits.json \
  --output-dir stella-rsi/data/processed
```

This writes `sft_{train,validation,test}.jsonl`,
`grpo_{train,validation,test}.jsonl`, and `manifest.json` with counts, splits,
and input/label hashes. GRPO references are separate metadata and are not appended
to the model prompt. Inspect the manifest and samples before training. Running
preparation again replaces these generated files.

## Train SFT, then GRPO

After preparing your reviewed dataset, inspect the recipes in
[`configs/`](configs). All recipe paths resolve **relative to the JSON file**;
CLI path overrides resolve relative to the current working directory.

```bash
accelerate config
python -m stella_rsi.train_sft --config stella-rsi/configs/sft_qwen3.8_27b.json --dry-run
python -m stella_rsi.train_grpo --config stella-rsi/configs/grpo_qwen3.8_27b.json --dry-run
bash stella-rsi/scripts/train.sh
```

For a single GPU, choose one process and no distributed training in Accelerate.
Standard multi-GPU DDP places a complete 4-bit policy on each GPU; it does not
shard a policy that is too large for one device. FSDP/DeepSpeed are not configured
by these recipes. The GRPO generation batch must be divisible by
`num_generations`; by default it is
`per_device_train_batch_size × gradient_accumulation_steps × world_size`.

The stages can also run separately:

```bash
accelerate launch -m stella_rsi.train_sft \
  --config stella-rsi/configs/sft_qwen3.8_27b.json
accelerate launch -m stella_rsi.train_grpo \
  --config stella-rsi/configs/grpo_qwen3.8_27b.json \
  --adapter stella-rsi/outputs/sft
```

SFT masks prompt tokens, including user messages and tool observations, and
learns only the selected assistant completion. GRPO **loads and continues the
SFT adapter**, generates groups of fresh completions, and computes group-relative
advantages from their rewards. The default `beta=0` disables a KL penalty.
Groups whose rewards are all equal supply no relative learning signal; monitor
reward variance, not just mean reward. Both stages drop overlength examples as
whole examples rather than truncating tool histories, and report counts in their
run manifests. GRPO does not use validation/test files during optimization;
reserve those files for a separate fixed-protocol comparison of the base, SFT,
and GRPO policies. This package does not claim measured quality improvements.

Each stage saves a PEFT adapter, tokenizer, and `run_manifest.json` with the
recipe, dependency versions, model revision, dataset hash, and retained-example
counts. Epoch checkpoints also contain optimizer/trainer state. Resume an
interrupted stage with `--resume-from-checkpoint path/to/checkpoint-N`.

Use `--model <Hugging-Face-ID-or-local-model>` to switch the base model in **both**
stages. This override resets the default model's pinned revision to `main`; for
reproducibility, edit both JSON recipes with the alternative model's exact
revision instead. Use a compatible causal model/chat template and an SFT adapter
trained from that same model. Optional CLI overrides include `--dataset` and
`--output-dir`; edit `eval_dataset` in the SFT JSON when changing validation data.

## Export

```bash
python -m stella_rsi.merge \
  --adapter stella-rsi/outputs/grpo \
  --output-dir stella-rsi/outputs/merged
```

Export loads the unquantized base on CPU, merges the LoRA adapter, and saves a
standalone **text-only** model. It requires sufficient host RAM for the 27B base.
The base revision comes from the training manifest; for an external adapter
without a manifest, provide `--revision` explicitly. The destination must be
empty. Deployment back into STELLA is a separate step; its default hosted model
is not changed by training or export.

## Tests and scope of validation

```bash
python -m pytest stella-rsi/tests -q
# Optional: real SFT and GRPO optimizer steps on a tiny locally created model.
# Requires the train extra, runs on CPU, downloads no model weights.
STELLA_RSI_TRAINING_TEST=1 python -m pytest stella-rsi/tests -q
```

Tests cover session splits, preserved context, explicit review labels, tool-call
reward parsing, recorder behavior, config paths, and adapter continuation. The
optional integration test uses tiny local GPT-2 models and the `qwen3_5`
architecture shared by Qwen3.5/3.8 to verify
composite-to-text weight loading, SFT updates, exact adapter restoration, a GRPO
optimizer update, and merged-model export. Its synthetic reward tests optimizer
mechanics only. It is not evidence of 27B GPU performance or scientific task
accuracy.

API references: [TRL SFT](https://huggingface.co/docs/trl/v1.18.0/en/sft_trainer),
[TRL GRPO](https://huggingface.co/docs/trl/v1.18.0/en/grpo_trainer), and the
[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).
