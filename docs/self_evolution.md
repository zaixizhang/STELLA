# Parameter-level self-evolution

STELLA's published self-evolution operates on its **context**: it retrieves thinking
templates and writes new tools. Two things follow from that. Everything STELLA has
learned has to be re-retrieved on every run, and nothing it has learned changes the
policy that drives it — the backbone model is exactly as capable after a thousand
successful runs as it was before the first.

This module adds the complementary mechanism. STELLA's own **verified** execution
history is used to post-train an open-weight policy (Qwen, DeepSeek) with SFT and
GRPO, so experience is written into **parameters**. The result is a policy that calls
the right tool, in the right order, with the right arguments, without retrieving a
template first — and that can be evaluated on tasks it has never seen.

```
   runs                verified history            parameters
 ┌────────┐  log   ┌──────────────────┐  SFT   ┌──────────────┐  serve
 │ STELLA │ ─────► │   trajectories   │ ─────► │   policy     │ ─────► STELLA
 │ agents │        │   + verifiers    │  GRPO  │ (Qwen/DS)    │        (next round)
 └────────┘        └──────────────────┘        └──────────────┘
      ▲                      │                        │
      └──────────────────────┴── held-out evaluation ──┘
                                 promote or discard
```

## What makes this different from template reuse

| | template reuse (published) | parameter-level (this module) |
|---|---|---|
| where experience lives | retrieved text in the context | model weights |
| cost at inference | retrieval + long prompt every run | none |
| optimisation signal | a template was reused | verified task outcome (GRPO advantage) |
| transfer to unseen tasks | only if a template matches | tested directly on held-out tasks |
| can it get worse | silently, by retrieving a bad template | blocked: a round is promoted only if held-out score improves |

## Pipeline

| module | role |
|---|---|
| `trajectory_logger.py` | harvest full trajectories (model inputs, outputs, tool calls, observations) from the running agents, after the fact, with no overhead |
| `reward.py` | score trajectories with **explicit verifiers**; this is the admission gate |
| `dataset_builder.py` | verified trajectories → SFT (step-level) + GRPO (task-level) datasets, with best-of-*n* selection and decontamination |
| `train_sft.py` | behaviour-clone verified decisions (LoRA by default) |
| `train_grpo.py` | optimise against verifiable scientific outcomes |
| `transfer.py` | ScienceRSI: specialise → verify → transfer → reuse |
| `evaluate.py` | held-out, verifier-scored comparison of two policies |
| `evolve.py` | one round end to end, with a promotion gate |
| `serve.py` | serve the evolved policy back into STELLA |

## Three rules the code enforces

**1. Nothing is learned from an unverified run.** A trajectory enters the training
set only if a verifier — not the agent's own report of success — passes it. A run
with no applicable verifier is treated as unusable, not as a success. See
`score_trajectory`: with no verifier, `verified` is `False`.

**2. The evaluation set is never in the training data.** `build_datasets` refuses to
run without a held-out task list (or an explicit `allow_no_heldout=True`), and drops
any trajectory whose task matches a held-out task exactly or by 8-gram overlap. A
gain measured on tasks that leaked into training says nothing about transfer.

**3. A round is kept only if it helps.** `evolve.run_round` serves the base policy
and the evolved policy, scores both on the same held-out tasks with the same
verifiers, and promotes only if the mean verified score improves. Otherwise the
checkpoint is recorded and discarded.

## Install

Run the commands below from the repository root. Source modules live in
[`stella/self_evolve/`](../stella/self_evolve/).

```bash
pip install -r requirements/self-evolve.txt      # trl, vllm on top of the base install
```

The data pipeline (logging, verification, dataset building) needs no GPU and no
extra dependencies.

## Use

### 1. Collect verified trajectories

Logging through `stella/core.py` is off unless asked for. Turn it on and run STELLA as usual:

```bash
export STELLA_TRAJECTORY_LOGGING=1
python -m stella
```

Or from Python:

`log_run` and `instrument` log unless `STELLA_TRAJECTORY_LOGGING=0`, so a script that
calls them directly needs no environment variable:

```python
from stella.self_evolve import instrument, log_run

instrument(manager_agent, domain="immunology")   # every run is logged
result = manager_agent.run(task)

# or log one run explicitly, with metadata
log_run(manager_agent, task, result, domain="immunology", task_id="nk_aml_01")
```

Trajectories are appended as JSON Lines under `self_evolve_data/trajectories/`.
Nothing is overwritten, and a corrupt line (a crash mid-write) is skipped at read
time rather than aborting a training job.

To replay a set of training tasks and collect several attempts each — GRPO needs
both successes and failures in a group to produce a gradient:

```python
from stella.self_evolve.evolve import collect_trajectories
collect_trajectories(tasks, attempts=4)   # tasks: [{"task":..., "task_id":..., "domain":...}]
```

### 2. Write the answer key and the held-out set

References live **next to** the trajectories, never inside them:

```json
// references.json       task_id -> reference answer
{
  "nk_aml_01": {"target": "BTN3A1", "top_k": 10},
  "rsstr_r2":  ["M276L", "V176F"],
  "dbqa_0042": "B"
}
// verifiers.json        task_id -> verifier name
{"nk_aml_01": "ranked_contains", "rsstr_r2": "contains", "dbqa_0042": "mcq"}
```

Built-in verifiers: `exact_match`, `contains`, `mcq`, `numeric`, `json_valid`, and
`ranked_contains` — which scores a discovery task by *where* a wet-lab-validated
target appears in the ranked output, so putting BTN3A1 at rank 1 scores above
burying it at rank 40. Register your own:

```python
from stella.self_evolve import register_verifier

@register_verifier("hplc_improved")
def hplc_improved(answer, reference, task):
    """reference = {"variant": "M276L", "fold": 2.1} from the wet lab."""
    ...
    return score, detail        # score in [0, 1]
```

### 3. Build datasets

```bash
python -m stella.self_evolve.dataset_builder \
    --trajectories self_evolve_data/trajectories \
    --out_dir      self_evolve_data/datasets \
    --references   self_evolve_data/references.json \
    --verifiers    self_evolve_data/verifiers.json \
    --heldout      self_evolve_data/heldout_task_strings.json \
    --min_reward 0.6 --max_per_task 2
```

Writes `sft.jsonl`, `grpo.jsonl` and `stats.json`. `stats.json` reports how many
trajectories were verified, how many distinct tasks they cover and the per-domain
breakdown — the numbers to quote when describing what the policy was trained on.

### 4. Train

```bash
# SFT: clone verified decisions
python -m stella.self_evolve.train_sft \
    --model Qwen/Qwen3-8B \
    --dataset self_evolve_data/datasets/sft.jsonl \
    --output_dir checkpoints/stella-sft-qwen3-8b

# GRPO: optimise against verifiable outcomes
python -m stella.self_evolve.train_grpo \
    --model checkpoints/stella-sft-qwen3-8b \
    --dataset self_evolve_data/datasets/grpo.jsonl \
    --output_dir checkpoints/stella-grpo-qwen3-8b \
    --num_generations 8 --use_vllm
```

Both accept `--dry_run`, which validates the data, the config and (for GRPO) the
reward functions without touching a GPU. Use it before queueing a job.

`--mode agentic` runs each GRPO rollout as a **real STELLA run** with live tools and
scores the run's final answer, which is the honest version of the objective: the
policy is rewarded for completing the science, not for producing text that looks
like a solution. It costs one agent run per rollout.

### 5. Evaluate and serve

```bash
# serve
python -m stella.self_evolve.serve --model Qwen/Qwen3-8B --lora checkpoints/stella-grpo-qwen3-8b

# score on held-out tasks, against a baseline report
python -m stella.self_evolve.evaluate \
    --tasks self_evolve_data/heldout_tasks.jsonl \
    --api_base http://localhost:8000/v1 --model_id stella-evolved \
    --out reports/evolved.json --baseline_report reports/baseline.json
```

To drive STELLA with the evolved policy, set two environment variables — no code
change, no other part of STELLA is affected:

```bash
export STELLA_LOCAL_POLICY_API_BASE=http://localhost:8000/v1
export STELLA_LOCAL_POLICY_MODEL=stella-evolved
python -m stella
```

### 6. A full round

```bash
python -m stella.self_evolve.evolve \
    --round 1 \
    --base_model Qwen/Qwen3-8B \
    --heldout    self_evolve_data/heldout_tasks.jsonl \
    --references self_evolve_data/references.json \
    --verifiers  self_evolve_data/verifiers.json \
    --work_dir   self_evolve_data/rounds
```

Writes `round_1/round.json` recording the data used, the checkpoints produced, both
evaluation reports and the promotion decision. Each round is reproducible from that
manifest plus the trajectory shards it names.

## ScienceRSI: specialise → verify → transfer → reuse

Training one policy on everything mixes immunology, protein engineering and
literature work into a single gradient. `transfer.py` instead trains compact domain
experts, keeps only the contributions a shared verifier confirms, and combines those
into one target policy.

```python
from stella.self_evolve.transfer import specialise, verify_experts, transfer_by_merge

experts = specialise("self_evolve_data/datasets/sft.jsonl", "Qwen/Qwen3-4B", "work/rsi")
kept    = verify_experts(experts, heldout_tasks, baseline_report, evaluate_fn=my_eval)
merged  = transfer_by_merge(kept, "Qwen/Qwen3-4B", "work/rsi/merged")
```

An expert that does not beat the policy it started from **on its own domain's
held-out tasks** is dropped, and the surviving adapters are merged with weights
proportional to their verified gain. For a different or larger target
(4B → 8B → 27B), `transfer_by_distillation` has each expert answer the training
prompts, keeps only answers that pass the task's verifier, and fine-tunes the target
on that verified pool.

## Tests

```bash
python -m tests.self_evolve.test_pipeline
```

69 checks over verification, the admission gate, best-of-*n* selection,
decontamination, prompt trimming, the SFT/GRPO dataset formats, JSONL round-tripping
with a corrupt line, the GRPO reward functions and trajectory extraction. No GPU and
no `trl` needed.

## What this does not claim

The module provides the mechanism and the measurement, not the result. Whether a
given round of SFT+GRPO improves STELLA on held-out biomedical tasks is an empirical
question that `evolve.py` answers per round, with the numbers in
`round_<n>/round.json`. A round that does not improve the held-out score is not
promoted, and that outcome is recorded as faithfully as a successful one.
