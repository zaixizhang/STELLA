#!/usr/bin/env python3
"""
Run the Category 2/3 MCQ benchmark in a reviewer-facing layout.
Tasks run sequentially. For each task: STELLA, Biomni, then OpenRouter models
(with max_workers concurrency for OpenRouter). CSV append is thread-safe.
"""
import os
import re
import csv
import json
import sys
import time
import subprocess
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Python <3.9

try:
    import requests
except ImportError:
    print("requests not found. Install for this Python:", file=sys.stderr)
    print(f"  {sys.executable} -m pip install requests", file=sys.stderr)
    sys.exit(1)

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_CSV = BENCHMARK_DIR / "benchmark_tasks_c2_c3_unified.csv"
DEFAULT_MCQ_CSV = BENCHMARK_DIR / "benchmark_mcq_c2_c3_unified.csv"
OUTPUT_DIR = BENCHMARK_DIR / "outputs"
DEFAULT_OUT_CSV = OUTPUT_DIR / "result.csv"

# Paths set from config in main() (so existing logic can stay unchanged)
CATEGORY2_CSV = DEFAULT_TASK_CSV
QUESTION_CSV = DEFAULT_MCQ_CSV
OUTPUT_BASE = OUTPUT_DIR / "runs" / "default_run"
RESULT_CSV = DEFAULT_OUT_CSV
run_id = "test_run_021926_daytime"
MAX_CONCURRENT_OPENROUTER = 3
MODELS = []  # filled from model registry by selected keys (OpenRouter only)
RUN_STELLA = True  # set in apply_config when "STELLA" is in selected models
RUN_BIOMNI = False  # set in apply_config when "Biomni" is in selected models

# Model registry: key -> (openrouter_id or None for STELLA/Biomni, file prefix)
# STELLA and Biomni run separately (sequential); others are OpenRouter models.
MODEL_REGISTRY = {
    "STELLA": (None, "STELLA"),
    "Biomni": (None, "Biomni"),
    "GPT4o": ("openai/gpt-4o", "GPT4o"),
    "DeepSeek": ("deepseek/deepseek-r1", "DeepSeek"),
    "Gemini": ("google/gemini-2.5-pro", "Gemini"),
    "Grok": ("x-ai/grok-4", "Grok"),
    "o3": ("openai/o3", "o3"),
    "ClaudeOpus": ("anthropic/claude-opus-4", "ClaudeOpus"),
    "ClaudeSonnet4": ("anthropic/claude-sonnet-4", "ClaudeSonnet4"),
}
DEFAULT_MODEL_KEYS = [k for k in MODEL_REGISTRY if k not in ("STELLA", "Biomni")]  # OpenRouter only for MODELS list
# Biomni A1 data: prefer existing biominibenchmark/biomni_data (no re-download); else use Test_run/biomni_data
REPO_ROOT = Path(__file__).resolve().parents[3]
_BIOMNI_LEGACY = REPO_ROOT / "Appeal" / "biominibenchmark"
BIOMNI_DATA_PATH = _BIOMNI_LEGACY if (_BIOMNI_LEGACY / "biomni_data").exists() else OUTPUT_DIR
BIOMNI_REPO = Path(os.environ.get("BIOMNI_REPO", REPO_ROOT / "Biomni"))
RUN_BIOMNI_ONE_TASK = BENCHMARK_DIR / "run_biomni_one_task.py"
# Biomni subprocess cwd: agent writes under this dir so Appeal_benchmark root stays clean
BIOMNI_FILE_DIR = BENCHMARK_DIR / "biomini_file"


def _get_biomni_python():
    """Return Path to biomni_e1 Python, or None if not found."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        p = Path(conda_prefix).parent / "envs" / "biomni_e1" / "bin" / "python"
        if p.exists():
            return p
    for base in (Path.home() / "miniconda3", Path.home() / "anaconda3"):
        p = base / "envs" / "biomni_e1" / "bin" / "python"
        if p.exists():
            return p
    return None


# Retry config
RETRY_BACKOFF_SEC = 5

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# STELLA codebase path. Set STELLA_DEV_DIR explicitly when running this bundle.
STELLA_DEV = Path(os.environ.get("STELLA_DEV_DIR", REPO_ROOT / "STELLA_dev"))
RUN_ONE_TASK = STELLA_DEV / "run_one_task.py"
STELLA_ANSWER_DELIMITER = "---MULTIPLE_CHOICE_ANSWERS---"

RESULT_CSV_HEADER = ["task_id", "q_id", "model", "answer", "correct_answer", "is_correct", "run_id", "timestamp", "notes"]

# Regex to parse repeat index from notes (e.g. "repeat=2" or "RESULT_FAILED" with no repeat -> 1)
_REPEAT_IN_NOTES = re.compile(r"repeat=(\d+)", re.I)


def _parse_repeat_from_notes(notes):
    """Return repeat index from notes; default 1 if not present."""
    if not notes:
        return 1
    m = _REPEAT_IN_NOTES.search(notes)
    return int(m.group(1)) if m else 1


def parse_args():
    """Parse CLI and optional --config JSON. Returns namespace with all config."""
    defaults = {
        "task_csv": str(DEFAULT_TASK_CSV),
        "mcq_csv": str(DEFAULT_MCQ_CSV),
        "out_csv": str(DEFAULT_OUT_CSV),
        "run_id": None,  # auto-generate if None
        "models": "",  # empty = all
        "task_ids": None,  # comma-separated
        "task_range": None,  # "start:end" inclusive
        "repeats": 1,
        "max_workers": 3,
        "resume_no_dupes": True,
        "config": None,
    }
    p = argparse.ArgumentParser(
        description="Run the Category 2/3 MCQ benchmark: STELLA, Biomni, and OpenRouter models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_category23_mcq_benchmark.py --config category23_mcq_benchmark_config_c3_new_part2.json
  python run_category23_mcq_benchmark.py --task_range 1:5 --max_workers 3 --run_id demo_run
  python run_category23_mcq_benchmark.py --task_ids 1,2 --models STELLA,GPT4o --run_id demo_subset
""",
    )
    p.add_argument("--task_csv", default=defaults["task_csv"], help="Path to task CSV (default: benchmark_tasks_c2_c3_unified.csv)")
    p.add_argument("--mcq_csv", default=defaults["mcq_csv"], help="Path to MCQ CSV (default: benchmark_mcq_c2_c3_unified.csv)")
    p.add_argument("--out_csv", default=defaults["out_csv"], help="Path to output result CSV")
    p.add_argument("--run_id", default=defaults["run_id"], help="Run ID (default: auto test_run_YYYYMMDD_HHMMSS)")
    p.add_argument("--models", default=defaults["models"], help="Comma-separated model keys (default: all). Keys: STELLA,Biomni,GPT4o,DeepSeek,Gemini,Grok,o3,ClaudeOpus")
    p.add_argument("--task_ids", default=defaults["task_ids"], help="Comma-separated task IDs, e.g. 14,15,16")
    p.add_argument("--task_range", default=defaults["task_range"], help="Inclusive range start:end, e.g. 23:27")
    p.add_argument("--repeats", type=int, default=defaults["repeats"], metavar="N", help="Repeats per (task_id, model) (default: 1)")
    p.add_argument("--max_workers", type=int, default=defaults["max_workers"], help="Concurrency for OpenRouter models within a task (default: 3)")
    p.add_argument("--resume_no_dupes", action="store_true", default=True, help="Skip duplicate (task_id, q_id, model, run_id, repeat) rows (default: True)")
    p.add_argument("--no_resume_no_dupes", action="store_false", dest="resume_no_dupes", help="Disable dedup / allow duplicate rows")
    p.add_argument("--config", default=defaults["config"], help="Optional JSON config file (overrides defaults)")
    args = p.parse_args()

    # Optional JSON overlay
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            overlay = json.load(f)
        for k, v in overlay.items():
            if hasattr(args, k):
                setattr(args, k, v)
            if k == "task_ids" and isinstance(v, list):
                setattr(args, k, ",".join(str(x) for x in v))

    # Auto run_id
    if not args.run_id or not args.run_id.strip():
        args.run_id = "test_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    return args


def _sanitize_run_id_for_path(run_id_str):
    """Make run_id safe as a folder name (no / \\ : * ? etc)."""
    s = str(run_id_str).strip() or "default_run"
    for c in '/\\:*?"<>|':
        s = s.replace(c, "_")
    return s[:120]  # avoid very long paths


def apply_config(args):
    """Set global paths and model list from parsed args."""
    global CATEGORY2_CSV, QUESTION_CSV, RESULT_CSV, OUTPUT_BASE, run_id, MAX_CONCURRENT_OPENROUTER, MODELS, RUN_STELLA, RUN_BIOMNI
    CATEGORY2_CSV = Path(args.task_csv)
    QUESTION_CSV = Path(args.mcq_csv)
    RESULT_CSV = Path(args.out_csv)
    run_id = args.run_id
    # Each run_id gets its own folder under Test_run so runs don't overwrite each other
    OUTPUT_BASE = RESULT_CSV.parent / _sanitize_run_id_for_path(run_id)
    MAX_CONCURRENT_OPENROUTER = args.max_workers
    # models: from CLI is comma-separated string; from JSON can be string or list
    if isinstance(getattr(args, "models", None), list):
        keys = [str(x).strip() for x in args.models if x]
    else:
        keys = [k.strip() for k in (args.models or "").split(",") if k.strip()]
    if not keys:
        keys = list(MODEL_REGISTRY.keys())
    RUN_STELLA = "STELLA" in keys
    RUN_BIOMNI = "Biomni" in keys
    MODELS = []
    for k in keys:
        if k not in MODEL_REGISTRY:
            print(f"Unknown model key '{k}'; skipping. Valid: {list(MODEL_REGISTRY.keys())}", file=sys.stderr)
            continue
        model_id, prefix = MODEL_REGISTRY[k]
        if model_id is not None:
            MODELS.append((model_id, prefix))


def get_all_task_ids(csv_path):
    """Return sorted list of unique task IDs from task CSV (first column). Supports numeric (14, 33) or string (D1, D2) IDs."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ids = []
        for row in reader:
            if row and row[0].strip():
                tid = row[0].strip()
                if tid != (header[0].strip() if header else "Task ID"):
                    ids.append(tid)
    return sorted(set(ids), key=_task_id_sort_key)


def validate_task_csv(path):
    """Check task CSV has required columns for prompts. Returns (True, None) or (False, error_msg)."""
    if not path or not Path(path).exists():
        return False, f"Task CSV not found: {path}"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            return False, "Task CSV is empty or header-only"
        # Need at least task identifier and content for prompt (Task Name / Task Description / Input)
        required = ["Task ID", "Task Name", "Task Type", "Task Description", "Input", "Success Criteria"]
        missing = [c for c in required if c not in row]
        if missing:
            return False, f"Task CSV missing columns: {missing}"
    return True, None


def validate_mcq_csv(path):
    """Check MCQ CSV has Q_ID, Answer, and task-id column for joining. Returns (True, None) or (False, error_msg)."""
    if not path or not Path(path).exists():
        return False, f"MCQ CSV not found: {path}"
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            return False, "MCQ CSV is empty or header-only"
        for col in ("Task_ID", "Q_ID", "Question", "Option_A", "Option_B", "Option_C", "Option_D", "Answer"):
            if col not in row:
                return False, f"MCQ CSV missing column: {col}"
    return True, None


def load_existing_csv_keys(result_csv=None, resume_no_dupes=True):
    """Load result CSV if it exists; return set of (task_id, q_id, model, run_id, repeat_idx) for dedup."""
    keys = set()
    if not resume_no_dupes:
        return keys
    path = result_csv if result_csv is not None else RESULT_CSV
    if not path.exists() or path.stat().st_size == 0:
        return keys
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("task_id", "").strip()
            q = row.get("q_id", "").strip()
            m = row.get("model", "").strip()
            r = row.get("run_id", "").strip()
            rep = _parse_repeat_from_notes(row.get("notes", ""))
            if t and q and m and r:
                keys.add((t, q, m, r, rep))
    return keys


def _is_retryable(err):
    """True for 429, 5xx, timeout, connection errors."""
    if isinstance(err, requests.exceptions.Timeout):
        return True
    if isinstance(err, requests.exceptions.ConnectionError):
        return True
    if isinstance(err, requests.exceptions.HTTPError) and err.response is not None:
        status = err.response.status_code
        if status == 429:
            return True
        if 500 <= status < 600:
            return True
    return False


def openrouter_chat_with_retry(model_id, messages, api_key, timeout=300):
    """Call OpenRouter; retry once after RETRY_BACKOFF_SEC on 429, 5xx, timeout, connection."""
    last_err = None
    for attempt in range(2):
        try:
            return openrouter_chat(model_id, messages, api_key, timeout=timeout)
        except Exception as e:
            last_err = e
            if attempt == 0 and _is_retryable(e):
                time.sleep(RETRY_BACKOFF_SEC)
                continue
            raise
    raise last_err


def read_task_row(csv_path, task_id):
    """Read full row for task_id from category2.csv (handles multiline fields)."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row and row[0].strip() == str(task_id):
                return dict(zip(header, row))
    return None


def read_questions_for_task(csv_path, task_id):
    """Return list of question row dicts for this task."""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Task_ID", "").strip() == str(task_id):
                rows.append(row)
    return rows


def build_stella_combined_prompt(task_row, question_rows):
    """One STELLA prompt: task result + delimiter + MCQ answers. Critical: all in one continuous block."""
    tid = task_row.get("Task ID", "")
    name = task_row.get("Task Name", "")
    ttype = task_row.get("Task Type", "")
    desc = task_row.get("Task Description", "")
    inp = task_row.get("Input", "")
    criteria = task_row.get("Success Criteria", "")
    q_ids = [q.get("Q_ID", f"Q{i}") for i, q in enumerate(question_rows, 1)]
    parts = [
        f"You are reading {CATEGORY2_CSV}. For Task ID {tid}, use the fields below to generate the final scientific result. Do not add extra assumptions beyond Input and Success Criteria.",
        "",
        f"Task Name: {name}",
        f"Task Type: {ttype}",
        f"Task Description: {desc}",
        "",
        f"Input:\n{inp}",
        "",
        f"Success Criteria:\n{criteria}",
        "",
        "CRITICAL — You MUST output everything below in ONE continuous block (no intermediate steps, no code blocks, no extra labels):",
        "1) Your complete scientific output (primers, calculations, analysis, etc.).",
        "2) Then exactly this line on its own: ---MULTIPLE_CHOICE_ANSWERS---",
        "3) Then for each question, exactly these three lines (no extra text between):",
        "    Q_ID: <question id>",
        "    <exactly one letter: A or B or C or D>",
        "    Justification: <one sentence>",
        "",
        "Example for one question:",
        "Q_ID: " + (q_ids[0] if q_ids else "15_Q1"),
        "D",
        "Justification: Option D is the best because...",
        "",
        "If the delimiter or the Q_ID/letter/Justification block is missing or in a different format, your MCQ answers cannot be recorded.",
        "",
    ]
    for i, q in enumerate(question_rows, 1):
        qid = q.get("Q_ID", f"Q{i}")
        parts.append(f"Question {i} ({qid}): {q.get('Question', '')}")
        parts.append(f"  A: {q.get('Option_A', '')}")
        parts.append(f"  B: {q.get('Option_B', '')}")
        parts.append(f"  C: {q.get('Option_C', '')}")
        parts.append(f"  D: {q.get('Option_D', '')}")
        parts.append("")
    parts.append("Output order: scientific result, then ---MULTIPLE_CHOICE_ANSWERS---, then for each question: Q_ID: <id>, then one line A/B/C/D, then Justification: ...")
    return "\n".join(parts)


def split_result_and_answers(full_text):
    """Split STELLA output into result part and MCQ answers part."""
    if not full_text:
        return "", ""
    if STELLA_ANSWER_DELIMITER not in full_text:
        return full_text.strip(), ""
    before, _, after = full_text.partition(STELLA_ANSWER_DELIMITER)
    return before.strip(), after.strip()


def run_stella_subprocess(prompt, env):
    """Run STELLA via run_one_task.py; return (stdout+stderr, returncode)."""
    proc = subprocess.run(
        [sys.executable, str(RUN_ONE_TASK), prompt],
        cwd=str(STELLA_DEV),
        env=env,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return out, proc.returncode


def extract_result_from_log(full_log):
    """Extract content between [RESULT] and [DONE] from STELLA log."""
    m = re.search(r"\[RESULT\]\s*\n(.*?)(?=\n\[DONE\]|\n\[ERROR\]|\Z)", full_log, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_result_and_answers_from_log(full_log):
    """
    Get result_text and answer_raw from STELLA log. Uses [RESULT]...[DONE] first;
    if answer block is missing there, fallback: search full log for last occurrence
    of ---MULTIPLE_CHOICE_ANSWERS--- and use text after it (avoids losing answers
    when STELLA outputs in multiple steps).
    """
    raw_result = extract_result_from_log(full_log)
    result_text, answer_raw = split_result_and_answers(raw_result)
    if not answer_raw and full_log and STELLA_ANSWER_DELIMITER in full_log:
        idx = full_log.rfind(STELLA_ANSWER_DELIMITER)
        if idx >= 0:
            after = full_log[idx + len(STELLA_ANSWER_DELIMITER) :].strip()
            end = len(after)
            for marker in ("\n[DONE]", "\n[ERROR]", "\n[RESULT]"):
                p = after.find(marker)
                if p >= 0:
                    end = min(end, p)
            after = after[: min(3000, end)].strip()
            if after and re.search(r"[A-Da-d]", after):
                answer_raw = after
    return result_text, answer_raw


def build_task_prompt(task_row):
    """Prompt for generating task result only. No extra assumptions."""
    name = task_row.get("Task Name", "")
    ttype = task_row.get("Task Type", "")
    desc = task_row.get("Task Description", "")
    inp = task_row.get("Input", "")
    criteria = task_row.get("Success Criteria", "")
    return (
        f"For Task ID {task_row.get('Task ID', '')}, use the following to generate the final scientific result. "
        "Do not add extra assumptions beyond the given Input and Success Criteria.\n\n"
        f"Task Name: {name}\n"
        f"Task Type: {ttype}\n"
        f"Task Description: {desc}\n\n"
        f"Input:\n{inp}\n\n"
        f"Success Criteria:\n{criteria}\n\n"
        "Generate only the scientific output required by the task (e.g. primers, calculations, analysis). "
        "Produce a clear final result."
    )


def build_mcq_prompt(result_text, question_rows):
    """Prompt for answering MCQs; require output format: 14_Q1\\nA\\n\\n14_Q2\\nC\\n\\n14_Q3\\nB."""
    parts = [
        "Use the following task result to answer each multiple-choice question. Output ONLY the answer letter (A, B, C, or D) for each question.",
        "",
        "---TASK RESULT---",
        (result_text or "(no result)").strip(),
        "---END TASK RESULT---",
        "",
        "Required output format (copy this structure; one line = Q_ID, next line = single letter A/B/C/D, then blank line):",
    ]
    for q in question_rows:
        qid = q.get("Q_ID", "")
        parts.append(qid)
        parts.append("X")
        parts.append("")
    parts.extend([
        "(Replace X with the correct letter for each question.)",
        "",
        "Questions:",
    ])
    for q in question_rows:
        qid = q.get("Q_ID", "")
        parts.append(f"{qid}: {q.get('Question', '')}")
        parts.append(f"  A: {q.get('Option_A', '')}")
        parts.append(f"  B: {q.get('Option_B', '')}")
        parts.append(f"  C: {q.get('Option_C', '')}")
        parts.append(f"  D: {q.get('Option_D', '')}")
        parts.append("")
    parts.append(
        "Reply with each Q_ID on its own line, then the single letter (A, B, C, or D) on the next line, then a blank line, for each question."
    )
    return "\n".join(parts)


def openrouter_chat(model_id, messages, api_key, timeout=300):
    """
    Call OpenRouter chat completions. Returns (content_str, full_response_dict).
    content_str is the assistant message text; full_response_dict is the JSON response for logging.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/stella-benchmark",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.3,
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = ""
    if data.get("choices") and len(data["choices"]) > 0:
        content = (data["choices"][0].get("message") or {}).get("content") or ""
    return content.strip(), data


def ensure_result_csv_header():
    """Create the output result CSV with header if missing or empty."""
    if not RESULT_CSV.exists() or RESULT_CSV.stat().st_size == 0:
        RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(RESULT_CSV_HEADER)


def _correct_answer_from_question(q):
    """Get normalized correct answer (A/B/C/D) from question row (category2_question has Answer column)."""
    raw = (q.get("Answer") or "").strip().upper()
    if raw and raw[0] in "ABCD":
        return raw[0]
    return ""


def _is_correct_value(model_answer, correct_answer):
    """Return '1' if model answer matches correct answer, else '0'. Empty/?/invalid -> '0'."""
    if not correct_answer:
        return "0"
    ma = (model_answer or "").strip().upper()
    if ma and ma[0] in "ABCD":
        return "1" if ma[0] == correct_answer else "0"
    return "0"


def build_result_csv_rows(task_id, question_rows, model_prefix, answers_and_notes, ts_iso, notes_override=None, existing_keys=None, existing_keys_lock=None, repeat_idx=1):
    """
    Build CSV rows for one model (same schema as result.csv). Dedup key includes repeat_idx.
    correct_answer comes from the unified MCQ answer key; is_correct is computed for scoring.
    """
    rows = []
    for i, q in enumerate(question_rows):
        q_id = q.get("Q_ID", "").strip()
        correct_answer = _correct_answer_from_question(q)
        key = (str(task_id), q_id, model_prefix, run_id, repeat_idx)
        if existing_keys is not None:
            if existing_keys_lock:
                with existing_keys_lock:
                    if key in existing_keys:
                        continue
            else:
                if key in existing_keys:
                    continue
        if notes_override:
            notes_str = notes_override
            model_ans = ""
        else:
            ans, notes = answers_and_notes[i] if i < len(answers_and_notes) else ("?", "INVALID_FORMAT")
            notes_str = notes or ""
            model_ans = ans or ""
        if repeat_idx != 1:
            notes_str = (notes_str + " " if notes_str else "") + f"repeat={repeat_idx}"
        is_correct = _is_correct_value(model_ans, correct_answer)
        row = [task_id, q_id, model_prefix, model_ans, correct_answer, is_correct, run_id, ts_iso, notes_str]
        rows.append(row)
        if existing_keys is not None:
            if existing_keys_lock:
                with existing_keys_lock:
                    existing_keys.add(key)
            else:
                existing_keys.add(key)
    return rows


# Lock for CSV appends (multiple workers may finish at different times)
_csv_append_lock = threading.Lock()


def append_rows_safe(rows):
    """Append rows to result.csv immediately. Thread-safe (run-one-write-one)."""
    if not rows:
        return
    with _csv_append_lock:
        ensure_result_csv_header()
        with open(RESULT_CSV, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)


def flush_result_csv():
    """Ensure result.csv is flushed to disk (call after finishing all models for a task)."""
    if not RESULT_CSV.exists():
        return
    with _csv_append_lock:
        with open(RESULT_CSV, "a", encoding="utf-8", newline="") as f:
            f.flush()
            if hasattr(os, "fsync"):
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass


def _run_one_openrouter_model(args):
    """
    Worker: run one OpenRouter model (Step1 + Step2), write per-model files, return CSV rows.
    args: (task_id, model_id, prefix, task_prompt, question_rows, ts_iso, api_key, out_dir, existing_keys, existing_keys_lock, repeat_idx)
    """
    (task_id, model_id, prefix, task_prompt, question_rows, ts_iso, api_key, out_dir, existing_keys, existing_keys_lock, repeat_idx) = args
    r_suffix = f"_r{repeat_idx}" if repeat_idx != 1 else ""
    result_file = out_dir / f"{prefix}{r_suffix}_result.txt"
    log_file = out_dir / f"{prefix}{r_suffix}.log"
    result_content = ""
    try:
        result_content, result_raw = openrouter_chat_with_retry(
            model_id, [{"role": "user", "content": task_prompt}], api_key
        )
        result_file.write_text(result_content, encoding="utf-8")
        log_file.write_text(json.dumps(result_raw, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        result_file.write_text(f"ERROR: {str(e)[:200]}", encoding="utf-8")
        extra = ""
        if hasattr(e, "response") and e.response is not None and getattr(e.response, "text", None):
            extra = "\n" + e.response.text
        log_file.write_text(f"Step 1 failed: {e}{extra}", encoding="utf-8")
        if question_rows:
            return build_result_csv_rows(
                task_id, question_rows, prefix, [], ts_iso, notes_override="RESULT_FAILED",
                existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
            )
        return []

    if not question_rows:
        return []

    try:
        mcq_prompt = build_mcq_prompt(result_content, question_rows)
        mcq_content, mcq_raw = openrouter_chat_with_retry(
            model_id, [{"role": "user", "content": mcq_prompt}], api_key
        )
        existing_log = log_file.read_text(encoding="utf-8")
        log_file.write_text(
            existing_log + "\n\n--- MCQ call ---\n" + json.dumps(mcq_raw, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        answers_and_notes = parse_mcq_to_letters(mcq_content, question_rows)
        return build_result_csv_rows(
            task_id, question_rows, prefix, answers_and_notes, ts_iso,
            existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
        )
    except Exception as e:
        (out_dir / f"{prefix}{r_suffix}_mcq_error.txt").write_text(f"ERROR: {str(e)[:200]}", encoding="utf-8")
        return build_result_csv_rows(
            task_id, question_rows, prefix, [], ts_iso, notes_override="MCQ_FAILED",
            existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
        )


def _run_biomni_for_task(task_id, task_row, question_rows, out_dir, ts_iso, repeat_idx, existing_keys, existing_keys_lock):
    """Run Biomni A1 in a subprocess with biomni_e1 Python; append rows to result CSV."""
    r_suffix = f"_r{repeat_idx}" if repeat_idx != 1 else ""
    result_file = out_dir / f"Biomni{r_suffix}_result.txt"
    log_file = out_dir / f"Biomni{r_suffix}.log"
    biomni_python = _get_biomni_python()
    if not biomni_python or not biomni_python.exists():
        result_file.write_text("ERROR: biomni_e1 Python not found. Install conda env: biomni_e1", encoding="utf-8")
        log_file.write_text("biomni_e1 not found", encoding="utf-8")
        append_rows_safe(build_result_csv_rows(
            task_id, question_rows, "Biomni", [], ts_iso, notes_override="AGENT_FAILED",
            existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
        ))
        return
    prompt = build_stella_combined_prompt(task_row, question_rows)
    prompt_file = out_dir / f"_biomni_prompt{r_suffix}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    # Run Biomni with cwd under biomini_file so agent-generated files don't clutter Appeal_benchmark
    biomni_cwd = BIOMNI_FILE_DIR / run_id / str(task_id)
    biomni_cwd.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(biomni_python),
            str(RUN_BIOMNI_ONE_TASK),
            "--prompt_file", str(prompt_file),
            "--result_file", str(result_file),
            "--log_file", str(log_file),
            "--data_path", str(BIOMNI_DATA_PATH),
            "--biomni_repo", str(BIOMNI_REPO),
        ],
        cwd=str(biomni_cwd),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if not result_file.exists():
            result_file.write_text("[Run failed - see .log for details.]", encoding="utf-8")
        append_rows_safe(build_result_csv_rows(
            task_id, question_rows, "Biomni", [], ts_iso, notes_override="AGENT_FAILED",
            existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
        ))
        return
    full_text = result_file.read_text(encoding="utf-8")
    _, answer_raw = split_result_and_answers(full_text)
    answers_and_notes = parse_mcq_to_letters(answer_raw, question_rows)
    append_rows_safe(build_result_csv_rows(
        task_id, question_rows, "Biomni", answers_and_notes, ts_iso,
        existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
    ))


def timestamp_iso_detroit():
    """ISO8601 local time with America/Detroit timezone if available."""
    try:
        if ZoneInfo is not None:
            tz = ZoneInfo("America/Detroit")
            return datetime.now(tz).isoformat()
    except Exception:
        pass
    return datetime.now().isoformat() + "Z"


def parse_mcq_to_letters(mcq_text, question_rows):
    """
    Extract one letter A/B/C/D per question; first valid letter in text.
    Returns list of (letter, notes): letter is A/B/C/D or "?", notes is "" or "INVALID_FORMAT".
    """
    if not question_rows:
        return []
    q_ids = [q.get("Q_ID", "").strip() for q in question_rows]
    text = (mcq_text or "").strip()
    out = []
    for qid in q_ids:
        letter = "?"
        notes = ""
        idx = text.find(qid)
        if idx >= 0:
            chunk = text[idx : idx + 400]
            m = re.search(r"[A-Da-d]", chunk)
            if m:
                letter = m.group(0).upper()
                notes = ""
            else:
                notes = "INVALID_FORMAT"
        else:
            notes = "INVALID_FORMAT"
        out.append((letter, notes))
    return out


def parse_mcq_to_standard_format(mcq_text, question_rows):
    """
    Parse model MCQ response into standard format:
    Q_ID
    <letter>

    Only the letter (A/B/C/D) per question. Uses question_rows to know Q_IDs.
    """
    if not question_rows:
        return ""
    q_ids = [q.get("Q_ID", "").strip() for q in question_rows]
    text = (mcq_text or "").strip()
    blocks = []
    for qid in q_ids:
        letter = "?"
        idx = text.find(qid)
        if idx >= 0:
            chunk = text[idx : idx + 400]
            m = re.search(r"(?:^|\n)\s*([A-Da-d])\s*(?:\n|$|,|\.)", chunk)
            if not m:
                m = re.search(r":\s*([A-Da-d])\s*", chunk)
            if m:
                letter = m.group(1).upper()
        blocks.append((qid, letter))
    lines = []
    for qid, letter in blocks:
        lines.append(qid)
        lines.append(letter)
        lines.append("")
    return "\n".join(lines).rstrip()


def _task_id_sort_key(x):
    """Sort task IDs: numeric first (by value), then string IDs."""
    s = str(x)
    return (0, int(s)) if s.isdigit() else (1, s)


def _resolve_task_ids(args):
    """Resolve task list: --task_ids > --task_range > all from task_csv. Supports int or string IDs (e.g. D1, D2)."""
    if args.task_ids:
        if isinstance(args.task_ids, list):
            task_ids = [str(x).strip() for x in args.task_ids if x is not None and str(x).strip()]
        else:
            task_ids = [
                int(x.strip()) if x.strip().isdigit() else x.strip()
                for x in str(args.task_ids).split(",") if x.strip()
            ]
        return sorted(set(task_ids), key=_task_id_sort_key)
    if args.task_range:
        part = args.task_range.strip().split(":")
        if len(part) != 2:
            print("--task_range must be start:end (e.g. 23:27)", file=sys.stderr)
            sys.exit(1)
        start, end = int(part[0].strip()), int(part[1].strip())
        return list(range(start, end + 1))
    return get_all_task_ids(CATEGORY2_CSV)


def run_one_task(task_id, api_key, existing_keys, existing_keys_lock, repeats):
    """Run one task: STELLA first (per repeat), then OpenRouter models with concurrency. Uses global paths/config."""
    out_dir = OUTPUT_BASE / str(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_row = read_task_row(CATEGORY2_CSV, task_id)
    if not task_row:
        print(f"Task {task_id} | SKIP (not found in task CSV)", file=sys.stderr)
        return

    question_rows = read_questions_for_task(QUESTION_CSV, task_id)
    task_prompt = build_task_prompt(task_row)

    for repeat_idx in range(1, repeats + 1):
        ts_iso = timestamp_iso_detroit()
        r_suffix = f"_r{repeat_idx}" if repeat_idx != 1 else ""
        stella_result_file = out_dir / f"{task_id}_stella{r_suffix}_result.txt"
        stella_log_file = out_dir / f"{task_id}_stella{r_suffix}.log"

        # --- Step A: STELLA first (sequential) when selected ---
        if RUN_STELLA:
            print(f"Task {task_id} repeat={repeat_idx} | STELLA | Step1 ...", flush=True)
            stella_env = os.environ.copy()
            if not stella_env.get("OPENROUTER_API_KEY"):
                stella_env["OPENROUTER_API_KEY"] = api_key
            if question_rows:
                try:
                    stella_prompt = build_stella_combined_prompt(task_row, question_rows)
                    full_log, _ = run_stella_subprocess(stella_prompt, stella_env)
                    stella_log_file.write_text(full_log, encoding="utf-8")
                    raw_result = extract_result_from_log(full_log)
                    if not raw_result and ("Traceback" in full_log or "Error" in full_log or "401" in full_log):
                        raw_result = "[Run failed - see .log for details.]"
                    result_text, answer_raw = extract_result_and_answers_from_log(full_log)
                    if not result_text and raw_result:
                        result_text, _ = split_result_and_answers(raw_result)
                    if not result_text:
                        result_text = raw_result
                    if answer_raw:
                        stella_result_file.write_text(
                            result_text + "\n\n" + STELLA_ANSWER_DELIMITER + "\n" + answer_raw, encoding="utf-8"
                        )
                    else:
                        stella_result_file.write_text(result_text, encoding="utf-8")
                    answers_and_notes = parse_mcq_to_letters(answer_raw, question_rows)
                    append_rows_safe(build_result_csv_rows(
                        task_id, question_rows, "STELLA", answers_and_notes, ts_iso,
                        existing_keys=existing_keys, repeat_idx=repeat_idx,
                    ))
                except subprocess.TimeoutExpired:
                    stella_result_file.write_text("ERROR: STELLA run timed out.", encoding="utf-8")
                    stella_log_file.write_text("STELLA run timed out.", encoding="utf-8")
                    append_rows_safe(build_result_csv_rows(
                        task_id, question_rows, "STELLA", [], ts_iso, notes_override="RESULT_FAILED",
                        existing_keys=existing_keys, repeat_idx=repeat_idx,
                    ))
                except Exception as e:
                    stella_result_file.write_text(f"ERROR: {str(e)[:200]}", encoding="utf-8")
                    stella_log_file.write_text(f"STELLA failed: {e}", encoding="utf-8")
                    append_rows_safe(build_result_csv_rows(
                        task_id, question_rows, "STELLA", [], ts_iso, notes_override="RESULT_FAILED",
                        existing_keys=existing_keys, repeat_idx=repeat_idx,
                    ))
            else:
                try:
                    stella_prompt = build_task_prompt(task_row)
                    full_log, _ = run_stella_subprocess(stella_prompt, stella_env)
                    stella_log_file.write_text(full_log, encoding="utf-8")
                    result_text = extract_result_from_log(full_log)
                    stella_result_file.write_text(result_text or "[No result extracted.]", encoding="utf-8")
                except Exception as e:
                    stella_result_file.write_text(f"ERROR: {str(e)[:200]}", encoding="utf-8")
                    stella_log_file.write_text(f"STELLA failed: {e}", encoding="utf-8")
            print(f"Task {task_id} repeat={repeat_idx} | STELLA done")

        # --- Step B: Biomni (A1 agent) when selected ---
        if RUN_BIOMNI and question_rows:
            print(f"Task {task_id} repeat={repeat_idx} | Biomni (A1) ...", flush=True)
            _run_biomni_for_task(
                task_id, task_row, question_rows, out_dir, ts_iso, repeat_idx,
                existing_keys, existing_keys_lock,
            )
            print(f"Task {task_id} repeat={repeat_idx} | Biomni done")

        # --- Step C: OpenRouter models concurrently ---
        if MODELS:
            print(f"Task {task_id} repeat={repeat_idx} | OpenRouter models (concurrency={MAX_CONCURRENT_OPENROUTER})", flush=True)
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OPENROUTER) as executor:
                futures = {
                    executor.submit(
                        _run_one_openrouter_model,
                        (task_id, model_id, prefix, task_prompt, question_rows, ts_iso, api_key, out_dir, existing_keys, existing_keys_lock, repeat_idx),
                    ): prefix
                    for model_id, prefix in MODELS
                }
                for future in as_completed(futures):
                    prefix = futures[future]
                    try:
                        rows = future.result()
                        append_rows_safe(rows)
                    except Exception as e:
                        print(f"Task {task_id} | {prefix} | worker failed: {e}", file=sys.stderr)
                        if question_rows:
                            append_rows_safe(build_result_csv_rows(
                                task_id, question_rows, prefix, [], ts_iso, notes_override="RESULT_FAILED",
                                existing_keys=existing_keys, existing_keys_lock=existing_keys_lock, repeat_idx=repeat_idx,
                            ))

    flush_result_csv()
    print(f"Task {task_id} complete; CSV updated", flush=True)


def main():
    args = parse_args()
    apply_config(args)

    # Fail fast: check required scripts and env before running any task
    if RUN_STELLA:
        if not STELLA_DEV.is_dir():
            print(f"STELLA selected but STELLA_dev dir not found: {STELLA_DEV}", file=sys.stderr)
            sys.exit(1)
        if not RUN_ONE_TASK.exists():
            print(f"STELLA selected but run script not found: {RUN_ONE_TASK}", file=sys.stderr)
            sys.exit(1)
    if RUN_BIOMNI:
        if not BIOMNI_REPO.is_dir():
            print(f"Biomni selected but Biomni repo not found: {BIOMNI_REPO}", file=sys.stderr)
            sys.exit(1)
        if not _get_biomni_python():
            print("Biomni selected but biomni_e1 env not found. Create it: conda create -n biomni_e1 ...", file=sys.stderr)
            sys.exit(1)
        if not RUN_BIOMNI_ONE_TASK.exists():
            print(f"Biomni helper script not found: {RUN_BIOMNI_ONE_TASK}", file=sys.stderr)
            sys.exit(1)

    # OpenRouter key needed for STELLA or OpenRouter models
    if RUN_STELLA or MODELS:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key or api_key.lower().startswith("your") or "your-key" in api_key:
            print("Set OPENROUTER_API_KEY to your OpenRouter API key (required for STELLA / OpenRouter models).", file=sys.stderr)
            sys.exit(1)
        if any(ord(c) > 127 for c in api_key):
            print("OPENROUTER_API_KEY must be ASCII-only.", file=sys.stderr)
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or ""

    # Biomni (A1) needs Anthropic key; try loading from Biomni .env
    if RUN_BIOMNI:
        env_file = BIOMNI_REPO / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file, override=False)
            except ImportError:
                pass
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            print("Biomni is selected but ANTHROPIC_API_KEY is not set. Put it in Biomni/.env or export it.", file=sys.stderr)
            sys.exit(1)

    # Validate input CSVs before running
    ok, err = validate_task_csv(CATEGORY2_CSV)
    if not ok:
        print(f">>> Validation failed: {err}", file=sys.stderr)
        sys.exit(1)
    ok, err = validate_mcq_csv(QUESTION_CSV)
    if not ok:
        print(f">>> Validation failed: {err}", file=sys.stderr)
        sys.exit(1)
    print(">>> Input CSVs validated OK")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    ensure_result_csv_header()
    existing_keys = load_existing_csv_keys(RESULT_CSV, args.resume_no_dupes)

    task_ids = _resolve_task_ids(args)
    if not task_ids:
        print(">>> No tasks to run.", file=sys.stderr)
        sys.exit(0)

    repeats = max(1, args.repeats)
    print(f">>> Task list: {len(task_ids)} tasks {task_ids[:5]}{'...' if len(task_ids) > 5 else ''}")
    print(f">>> Repeats per (task, model): {repeats}")
    print(f">>> Run ID: {run_id}")
    print(f">>> Output base: {OUTPUT_BASE}")
    print(f">>> Result CSV: {RESULT_CSV} (dedup keys={len(existing_keys)} loaded)")
    print(f">>> Models: STELLA={RUN_STELLA}, Biomni={RUN_BIOMNI}, OpenRouter={[p for _, p in MODELS]}")
    if RUN_STELLA:
        print(f">>> STELLA: Python={sys.executable}, script={RUN_ONE_TASK}")
    if RUN_BIOMNI:
        bp = _get_biomni_python()
        print(f">>> Biomni: Python={bp}, helper={RUN_BIOMNI_ONE_TASK}, repo={BIOMNI_REPO}")
    print()

    for task_id in task_ids:
        existing_keys_lock = threading.Lock()
        run_one_task(task_id, api_key, existing_keys, existing_keys_lock, repeats)

    print()
    print(f">>> Done. Outputs in {OUTPUT_BASE}, CSV: {RESULT_CSV}")

    # Auto-run scoring for this run_id: strict + lenient
    score_dir = OUTPUT_BASE / "score"
    score_script = BENCHMARK_DIR / "score_category23_mcq_results.py"
    if score_script.exists() and RESULT_CSV.exists():
        print()
        print(f">>> Running scoring for run_id={run_id} -> {score_dir}")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(score_script),
                    "--run_id", run_id,
                    "--result_csv", str(RESULT_CSV),
                    "--ground_truth_csv", str(QUESTION_CSV),
                    "--out_dir", str(score_dir),
                ],
                check=False,
                timeout=120,
            )
            print(f">>> Score outputs: {score_dir / 'score_answers_scored.csv'}, {score_dir / 'score_summary.csv'}")
        except subprocess.TimeoutExpired:
            print(">>> Scoring timed out (120s), skipped.", file=sys.stderr)
        except Exception as e:
            print(f">>> Scoring failed: {e}", file=sys.stderr)
    else:
        if not score_script.exists():
            print(f">>> No score script at {score_script}; skip auto-scoring.")


if __name__ == "__main__":
    main()
