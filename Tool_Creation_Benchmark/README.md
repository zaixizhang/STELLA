# Tool Creation Benchmark

This benchmark evaluates biomedical AI agents across three capability groups organized into two difficulty tiers:

| Set | Categories | Tasks | Evaluation |
|-----|-----------|-------|------------|
| **Simple Set** | Category 2 (Protocol/Design/Computation) + Category 3 (Database/Web Retrieval) | 99 tasks, 308 MCQs | Objective multiple-choice questions |
| **Hard Set** | Category 4 (Biology-Oriented ML) | 18 tasks (8 ProteinGym + 10 TDC/Polaris) | Metric-based predictions graded against task-specific leaderboards |

---

## Benchmark Structure

```
Tool_Creation_Benchmark/
├── simple_set/          # Category 2 + Category 3 (MCQ-based)
│   ├── tasks.csv        # Unified task definitions (99 tasks)
│   ├── mcq.csv          # MCQ answer-key table (308 questions)
│   ├── run_benchmark.py # Main benchmark runner
│   ├── score_results.py # Scoring script
│   └── config_example.json
└── hard_set/            # Category 4 (biology ML)
    ├── task_ids.csv          # 18 evaluated tasks
    ├── run_benchmark.py      # Full-panel runner (STELLA + baselines)
    ├── run_openrouter_batch.py  # OpenRouter/baseline runner
    ├── summarize_eval.py     # Aggregation and leaderboard-percentile script
    └── results/              # Pre-computed summary tables
```

---

## Simple Set (Categories 2 & 3)

### Overview

- **Category 2** — Protocol, design, and computational analysis tasks (52 tasks, 167 MCQs).
  Representative tasks: PCR primer design, restriction digest planning, molecular cloning, bioinformatics workflow reasoning.

- **Category 3** — Database and information retrieval tasks (47 tasks, 141 MCQs).
  Representative tasks: KEGG pathway extraction, DGIdb interaction summaries, GEPIA2 gene pages, structured web retrieval.

Each task is paired with 2–4 objective MCQs derived from verified reference outputs and source materials. Scores are binary (correct = 1, incorrect = 0), aggregated as mean accuracy per task and then per model.

### Prerequisites

```bash
# Python 3.9+
pip install requests

# Set API keys
export OPENROUTER_API_KEY=<your_openrouter_api_key>

# To run STELLA, point to your STELLA checkout
export STELLA_DEV_DIR=/path/to/STELLA

# To run Biomni, point to the Biomni repo
export BIOMNI_REPO=/path/to/Biomni
```

### Running the Benchmark

```bash
cd Tool_Creation_Benchmark/simple_set

# Run with the example config (edit models/run_id as needed)
python run_benchmark.py --config config_example.json
```

The runner executes each task sequentially. For each task it runs STELLA (if selected), Biomni (if selected), and OpenRouter-backed models concurrently. Results are appended to `outputs/result.csv`.

**Config fields:**

| Field | Description |
|-------|-------------|
| `task_csv` | Path to task definitions CSV |
| `mcq_csv` | Path to MCQ answer-key CSV |
| `out_csv` | Output CSV path |
| `run_id` | Unique identifier for this run |
| `models` | List of models to evaluate (`STELLA`, `Biomni`, `GPT4o`, `DeepSeek`, `Gemini`, `Grok`, `o3`, `ClaudeOpus`) |
| `repeats` | Number of independent runs per task (default 3) |
| `max_workers` | Parallel workers for OpenRouter models |
| `resume_no_dupes` | Skip already-completed task/model/repeat combos |

### Scoring a Run

```bash
python score_results.py \
  --run_id <run_id> \
  --result_csv outputs/result.csv \
  --ground_truth_csv mcq.csv \
  --out_dir outputs/scored
```

This produces:
- `outputs/scored/score_answers_scored.csv` — question-level strict/lenient correctness
- `outputs/scored/score_summary.csv` — per-model strict and lenient accuracy with standard deviation

### Scoring Protocol

- **Strict**: invalid/missing outputs count as 0.
- **Lenient**: invalid/missing outputs are excluded from the denominator.
- Task score = mean MCQ accuracy across all questions for that task.
- Model score = mean task score across all tasks.
- Each task-model pair is run 3 times; mean ± SD are reported.

---

## Hard Set (Category 4)

### Overview

18 biology-oriented machine-learning tasks evaluated with objective task-specific graders and leaderboard-normalized percentiles:

- **8 ProteinGym tasks** — deep mutational scanning / protein engineering (e.g., `SPIKE_SARS2_Starr_2020_binding`, `BRCA1_HUMAN_Findlay_2018`)
- **10 TDC / Polaris tasks** — ADMET and drug-discovery properties (e.g., `tdcommons-bbb-martins`, `tdcommons-caco2-wang`)

Tasks are run end-to-end inside a standardized containerized CPU-only runtime (`biomlbench`) with an 8-hour per-task time limit. Each task-model pair is run 3 independent times. The reported metric is `leaderboard_percentile` (higher = better), aggregated as:

- **Non-penalized**: mean over valid/gradable runs only.
- **Penalized**: missing/invalid runs assigned percentile 0.

### Prerequisites

- Docker with the `biomlbench` image
- `biomlbench` CLI installed and tasks prepared (`biomlbench prepare <task_id>`)
- API keys in `env/reviewer_api_keys.env` (see the `env/` directory in the original repro package)

### Running the Benchmark

**STELLA full panel:**
```bash
cd Tool_Creation_Benchmark/hard_set
python run_benchmark.py --config <config.json> --out_dir outputs/stella_run
```

**OpenRouter / baseline models:**
```bash
python run_openrouter_batch.py \
  --config config_openrouter_tdc_r3.json \
  --out_dir outputs/openrouter_run
```

**Summarize results:**
```bash
python summarize_eval.py \
  --run_dirs outputs/stella_run outputs/openrouter_run \
  --task_csv task_ids.csv \
  --out_dir results/
```

### Pre-Computed Results

The `results/` directory contains reviewer-facing summary tables from our benchmark runs:

| File | Description |
|------|-------------|
| `current_model_summary.csv` | Per-model task-normalized percentile summary |
| `current_model_official_lb_summary.csv` | Non-penalized leaderboard percentile (top-k repeat mean) |
| `current_model_official_lb_penalized_summary.csv` | Penalized leaderboard percentile |
| `current_model_official_lb_summary_all_mean.csv` | All-valid-runs mean (non-penalized) |
| `current_model_official_lb_penalized_summary_all_mean.csv` | All-valid-runs mean (penalized) |
| `current_model_robustness_summary.csv` | Run-to-run variance and completion rates |
| `current_task_model_selected.csv` | Per-task per-model selected score |
| `current_task_model_selected_official_lb.csv` | Per-task per-model leaderboard percentile |

---

## Key Results

### Simple Set (Categories 2 & 3 combined, strict accuracy)

| Model | Category 2 | Category 3 | Overall |
|-------|-----------|-----------|---------|
| STELLA | 0.9132 | **0.8735** | **0.9082** |
| Biomni | **0.9341** | 0.8333 | 0.9033 |
| o3 | 0.9311 | 0.7870 | 0.8919 |
| GPT-5 | 0.8953 | 0.8156 | 0.8693 |
| ClaudeSonnet4 | 0.8408 | 0.7600 | 0.8029 |

Results reported as mean over 3 independent runs.

### Hard Set (Category 4, penalized avg. leaderboard percentile)

| Model | Valid Tasks | Avg Leaderboard %ile (penalized) |
|-------|------------|----------------------------------|
| STELLA | 18/18 | **42.87** |
| GPT-5 | 12/18 | 12.96 |
| ClaudeSonnet4 | 6/18 | 5.56 |

---

## Citation

If you use this benchmark, please cite:

```bibtex
@article{jin2025stella,
  title={STELLA: Towards a Biomedical World Model with Self-Evolving Multimodal Agents},
  author={Jin, Ruofan and Xu, Mingyang and Meng, Fei and Wan, Guancheng and Cai, Qingran and Jiang, Yize and Han, Jin and Chen, Yuanyuan and Lu, Wanqing and Wang, Mengyang and Lan, Zhiqian and Jiang, Yuxuan and Liu, Junhong and Wang, Dongyao and Cong, Le and Zhang, Zaixi},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.07.01.662467}
}
```
