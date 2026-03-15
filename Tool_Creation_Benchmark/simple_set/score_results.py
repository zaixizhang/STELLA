#!/usr/bin/env python3
"""
Score MCQ predictions for a given run_id.
Reads the benchmark result CSV plus the unified MCQ answer key, computes strict/lenient correctness,
and writes score_answers_scored.csv and score_summary.csv. Does not modify the raw result CSV.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
import re

# Default paths
BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_CSV = BUNDLE_DIR / "outputs" / "result.csv"
DEFAULT_GROUND_TRUTH_CSV = BUNDLE_DIR / "benchmark_mcq_c2_c3_unified.csv"
DEFAULT_OUT_DIR = BUNDLE_DIR / "outputs" / "scored"
DEFAULT_RUN_ID = "test_run_021926_daytime"


def normalize_answer(s):
    """Normalize to single letter A/B/C/D or None if invalid/unanswered."""
    if pd.isna(s) or s is None:
        return None
    raw = str(s).strip().upper()
    if raw and raw[0] in "ABCD":
        return raw[0]
    return None


def load_ground_truth(gt_path):
    """Load Q_ID -> correct letter (A/B/C/D) from the unified MCQ answer key."""
    df = pd.read_csv(gt_path, encoding="utf-8-sig")
    out = {}
    for _, row in df.iterrows():
        qid = row.get("Q_ID")
        if pd.isna(qid):
            continue
        a = normalize_answer(row.get("Answer"))
        if a:
            out[str(qid).strip()] = a
    return out


_REPEAT_IN_NOTES = re.compile(r"repeat=(\d+)", re.I)


def parse_repeat_idx(notes):
    """Parse repeat index from notes; default 1 if missing/invalid."""
    if notes is None or (isinstance(notes, float) and pd.isna(notes)):
        return 1
    m = _REPEAT_IN_NOTES.search(str(notes))
    if not m:
        return 1
    try:
        return int(m.group(1))
    except ValueError:
        return 1


def main():
    p = argparse.ArgumentParser(description="Score MCQ run by run_id.")
    p.add_argument("--run_id", default=DEFAULT_RUN_ID, help="Run ID to score")
    p.add_argument("--result_csv", type=Path, default=DEFAULT_RESULT_CSV, help="Predictions CSV")
    p.add_argument("--ground_truth_csv", type=Path, default=DEFAULT_GROUND_TRUTH_CSV, help="Ground truth MCQ CSV")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for scored files")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions (tolerate lines with extra/trailing commas)
    with open(args.result_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row[: len(header)] for row in reader]
    df = pd.DataFrame(rows, columns=header[: len(header)])
    run_id_str = args.run_id.strip()
    run_match = (
        (df["run_id"].astype(str).str.strip() == run_id_str)
        | (df["timestamp"].astype(str).str.strip() == run_id_str)
    )
    df = df[run_match].copy()
    if df.empty:
        print(f"No rows with run_id == '{args.run_id}' in {args.result_csv}")
        return

    # Add repeat index from notes (default 1)
    df["repeat_idx"] = df["notes"].map(parse_repeat_idx)

    # Dedupe: same (q_id, model, run_id, repeat_idx) -> keep latest by timestamp, else last in file order
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values("timestamp", na_position="last")
    df = df.drop_duplicates(subset=["q_id", "model", "run_id", "repeat_idx"], keep="last").reset_index(drop=True)

    # Ground truth
    gt = load_ground_truth(args.ground_truth_csv)

    # Normalize predicted answer
    df["_pred_norm"] = df["answer"].map(normalize_answer)
    df["_valid"] = df["_pred_norm"].notna()
    df["_correct_gt"] = df["q_id"].astype(str).str.strip().map(lambda x: gt.get(x, ""))

    # Strict: invalid/unanswered counts as wrong (0)
    df["is_correct_strict"] = (
        (df["_valid"] & (df["_pred_norm"] == df["_correct_gt"])).astype(int)
    )
    # Lenient: invalid excluded from denominator (blank/NaN for invalid)
    df["is_correct_lenient"] = None
    valid_mask = df["_valid"]
    df.loc[valid_mask, "is_correct_lenient"] = (
        (df.loc[valid_mask, "_pred_norm"] == df.loc[valid_mask, "_correct_gt"]).astype(int)
    )

    # Drop temp columns for output
    out_cols = [c for c in df.columns if not c.startswith("_")]
    scored = df[out_cols].copy()

    # Row-level output
    score_answers_path = out_dir / "score_answers_scored.csv"
    scored.to_csv(score_answers_path, index=False, encoding="utf-8")
    print(f"Wrote {score_answers_path}")

    # Per-model, per-repeat stats
    def _lenient_acc(g):
        if g["_valid"].any():
            return g.loc[g["_valid"], "is_correct_lenient"].mean()
        return float("nan")

    per_repeat = df.groupby(["model", "repeat_idx"], group_keys=False).apply(
        lambda g: pd.Series({
            "n_total": len(g),
            "n_valid": int(g["_valid"].sum()),
            "strict_acc": g["is_correct_strict"].mean(),
            "lenient_acc": _lenient_acc(g),
        }),
        include_groups=False,
    ).reset_index()

    per_repeat["strict_score"] = (per_repeat["strict_acc"] * 100)
    per_repeat["lenient_score"] = (per_repeat["lenient_acc"] * 100)

    # Aggregate across repeats: average + sd
    agg = per_repeat.groupby("model").agg(
        n_repeats=("repeat_idx", "nunique"),
        n_total=("n_total", "mean"),
        n_valid=("n_valid", "mean"),
        strict_acc=("strict_acc", "mean"),
        strict_acc_sd=("strict_acc", "std"),
        lenient_acc=("lenient_acc", "mean"),
        lenient_acc_sd=("lenient_acc", "std"),
        strict_score=("strict_score", "mean"),
        strict_score_sd=("strict_score", "std"),
        lenient_score=("lenient_score", "mean"),
        lenient_score_sd=("lenient_score", "std"),
    ).reset_index()

    # Replace NaN sd (single repeat) with 0
    for col in ["strict_acc_sd", "lenient_acc_sd", "strict_score_sd", "lenient_score_sd"]:
        agg[col] = agg[col].fillna(0)

    summary = agg.copy()
    summary["n_total"] = summary["n_total"].round(0).astype(int)
    summary["n_valid"] = summary["n_valid"].round(0).astype(int)
    summary["strict_acc"] = summary["strict_acc"].round(4)
    summary["lenient_acc"] = summary["lenient_acc"].round(4)
    summary["strict_acc_sd"] = summary["strict_acc_sd"].round(4)
    summary["lenient_acc_sd"] = summary["lenient_acc_sd"].round(4)
    summary["strict_score"] = summary["strict_score"].round(2)
    summary["lenient_score"] = summary["lenient_score"].round(2)
    summary["strict_score_sd"] = summary["strict_score_sd"].round(2)
    summary["lenient_score_sd"] = summary["lenient_score_sd"].round(2)
    summary_path = out_dir / "score_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Wrote {summary_path}")

    # Console table
    print()
    print("model       | repeats | n_total | n_valid | strict_acc±sd | lenient_acc±sd | strict_score±sd | lenient_score±sd")
    print("-" * 112)
    for _, row in summary.iterrows():
        la = f"{row['lenient_acc']:.4f}"
        ls = f"{row['lenient_score']:.2f}"
        print(
            f"{row['model']:<11} |"
            f" {int(row['n_repeats']):>7} |"
            f" {int(row['n_total']):>6} |"
            f" {int(row['n_valid']):>6} |"
            f" {row['strict_acc']:.4f}±{row['strict_acc_sd']:.4f} |"
            f" {la}±{row['lenient_acc_sd']:.4f} |"
            f" {row['strict_score']:.2f}±{row['strict_score_sd']:.2f} |"
            f" {ls}±{row['lenient_score_sd']:.2f}"
        )


if __name__ == "__main__":
    main()
