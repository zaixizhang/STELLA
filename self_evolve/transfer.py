"""ScienceRSI-style specialise / verify / transfer / reuse over STELLA's history.

Training one policy on everything STELLA has ever done mixes immunology,
protein engineering and literature work into a single gradient. The
alternative implemented here is the recursive self-improvement loop of
ScienceRSI: train *compact domain experts*, keep only the contributions a
shared verifier can confirm, and transfer those into one target policy.

    specialise   one LoRA expert per domain, from that domain's verified
                 trajectories only
    verify       each expert is scored on held-out tasks; an expert that does
                 not beat the policy it started from is dropped
    transfer     the surviving experts are combined into the target policy,
                 either by weighted adapter merge (same base model) or by
                 verified distillation (different or larger target)
    reuse        the target is served back into STELLA, and its runs become
                 the next round's trajectories

Two transfer routes are provided because they cover different cases:

* ``transfer_by_merge`` -- experts share one base model, so their adapters can
  be merged directly, weighted by verified gain. Cheap, no generation needed.
* ``transfer_by_distillation`` -- the target is a different or larger model.
  Each expert answers the training prompts, the verifier keeps only correct
  answers, and the target is fine-tuned on that verified pool. This is the
  route that supports 4B -> 8B -> 27B style transfer.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evaluate import EvalReport, EvalTask, evaluate_policy, load_tasks

__all__ = [
    "ExpertSpec",
    "split_sft_by_domain",
    "specialise",
    "verify_experts",
    "transfer_by_merge",
    "transfer_by_distillation",
    "write_manifest",
]


@dataclass
class ExpertSpec:
    domain: str
    dataset: str
    output_dir: str
    n_samples: int = 0
    report: Optional[Dict[str, Any]] = None
    gain: float = 0.0
    kept: bool = False

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


# ---------------------------------------------------------------------------
# specialise
# ---------------------------------------------------------------------------

def split_sft_by_domain(
    sft_path: str,
    out_dir: str,
    *,
    min_samples: int = 64,
    max_domains: int = 0,
) -> List[Tuple[str, str, int]]:
    """Split ``sft.jsonl`` into one file per domain.

    Returns [(domain, path, n_samples)] for domains with enough data. Domains
    below ``min_samples`` are reported and skipped rather than trained on: a
    domain expert fitted to a dozen steps is noise, not a specialist.
    """
    by_domain: Dict[str, List[str]] = collections.defaultdict(list)
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            domain = str(obj.get("domain") or "unspecified")
            by_domain[domain].append(json.dumps({"prompt": obj["prompt"], "completion": obj["completion"]}))

    os.makedirs(out_dir, exist_ok=True)
    ranked = sorted(by_domain.items(), key=lambda kv: len(kv[1]), reverse=True)
    if max_domains:
        ranked = ranked[:max_domains]

    out: List[Tuple[str, str, int]] = []
    for domain, rows in ranked:
        if len(rows) < min_samples:
            print(f"[transfer] skipping domain {domain!r}: {len(rows)} samples < {min_samples}")
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in domain) or "unspecified"
        path = os.path.join(out_dir, f"sft_{safe}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        out.append((domain, path, len(rows)))
        print(f"[transfer] domain {domain!r}: {len(rows)} samples -> {path}")
    return out


def specialise(
    sft_path: str,
    base_model: str,
    work_dir: str,
    *,
    min_samples: int = 64,
    max_domains: int = 0,
    sft_args: Sequence[str] = (),
) -> List[ExpertSpec]:
    """Train one LoRA expert per domain. Returns the specs (untested)."""
    from . import train_sft  # noqa: PLC0415

    splits = split_sft_by_domain(
        sft_path, os.path.join(work_dir, "domain_data"), min_samples=min_samples, max_domains=max_domains
    )
    experts: List[ExpertSpec] = []
    for domain, path, n in splits:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in domain)
        out_dir = os.path.join(work_dir, "experts", safe)
        print(f"\n[transfer] specialising on {domain!r} ({n} samples) -> {out_dir}")
        rc = train_sft.main(
            [
                "--model",
                base_model,
                "--dataset",
                path,
                "--output_dir",
                out_dir,
                "--min_samples",
                str(min_samples),
                *sft_args,
            ]
        )
        if rc != 0:
            print(f"[transfer] expert for {domain!r} failed (rc={rc}); skipping")
            continue
        experts.append(ExpertSpec(domain=domain, dataset=path, output_dir=out_dir, n_samples=n))
    return experts


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify_experts(
    experts: Sequence[ExpertSpec],
    heldout_tasks: Sequence[EvalTask],
    baseline: EvalReport,
    *,
    evaluate_fn: Any = None,
    min_gain: float = 0.0,
    per_domain: bool = True,
) -> List[ExpertSpec]:
    """Score each expert on held-out tasks and keep only the ones that help.

    ``evaluate_fn(expert) -> EvalReport`` is injected by the caller (it needs
    to serve each adapter, which is deployment-specific). An expert is kept
    only if it improves the mean verified score over ``baseline`` by more than
    ``min_gain``; with ``per_domain`` the comparison uses that domain's tasks,
    which is the honest test of a specialist.
    """
    if evaluate_fn is None:
        raise ValueError("verify_experts needs evaluate_fn=callable(ExpertSpec) -> EvalReport")

    kept: List[ExpertSpec] = []
    for spec in experts:
        report = evaluate_fn(spec)
        spec.report = report.to_dict()
        if per_domain and spec.domain in report.per_domain and spec.domain in baseline.per_domain:
            spec.gain = report.per_domain[spec.domain] - baseline.per_domain[spec.domain]
            scope = f"domain {spec.domain!r}"
        else:
            spec.gain = report.mean_score - baseline.mean_score
            scope = "all held-out tasks"
        spec.kept = spec.gain > min_gain
        print(
            f"[transfer] expert {spec.domain!r}: gain {spec.gain:+.4f} on {scope} "
            f"-> {'KEEP' if spec.kept else 'DROP'}"
        )
        if spec.kept:
            kept.append(spec)
    return kept


# ---------------------------------------------------------------------------
# transfer
# ---------------------------------------------------------------------------

def transfer_by_merge(
    experts: Sequence[ExpertSpec],
    base_model: str,
    output_dir: str,
    *,
    combination_type: str = "linear",
    weight_by_gain: bool = True,
) -> str:
    """Merge verified expert adapters into one adapter on the same base model.

    Weights are proportional to each expert's verified gain, so an expert that
    barely helped contributes proportionally little. Returns ``output_dir``.
    """
    if not experts:
        raise ValueError("no verified experts to transfer")

    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    gains = [max(e.gain, 0.0) for e in experts]
    total = sum(gains)
    if weight_by_gain and total > 0:
        weights = [g / total for g in gains]
    else:
        weights = [1.0 / len(experts)] * len(experts)
    names = ["".join(c if c.isalnum() or c in "-_" else "_" for c in e.domain) or f"e{i}" for i, e in enumerate(experts)]

    print(f"[transfer] merging {len(experts)} experts with weights {[round(w, 3) for w in weights]}")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(model, experts[0].output_dir, adapter_name=names[0])
    for name, spec in zip(names[1:], experts[1:]):
        peft_model.load_adapter(spec.output_dir, adapter_name=name)

    merged_name = "stella_transferred"
    peft_model.add_weighted_adapter(
        adapters=names, weights=weights, adapter_name=merged_name, combination_type=combination_type
    )
    peft_model.set_adapter(merged_name)

    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir, selected_adapters=[merged_name])
    try:
        AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[transfer] could not save tokenizer: {e}")
    print(f"[transfer] merged adapter -> {output_dir}")
    return output_dir


def transfer_by_distillation(
    experts: Sequence[ExpertSpec],
    target_model: str,
    grpo_path: str,
    work_dir: str,
    *,
    generate_fn: Any = None,
    verifier_threshold: float = 0.999,
    sft_args: Sequence[str] = (),
) -> str:
    """Distil verified expert answers into a different or larger target model.

    ``generate_fn(expert, prompt_row) -> str`` is injected by the caller (each
    expert has to be served to generate). Only answers that pass the task's own
    verifier enter the distillation set, so the target learns verified
    behaviour rather than imitating an expert's mistakes.
    """
    if generate_fn is None:
        raise ValueError("transfer_by_distillation needs generate_fn=callable(ExpertSpec, row) -> str")

    from .reward import _VERIFIERS  # noqa: PLC0415,PLC2701
    from . import train_sft  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    with open(grpo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    os.makedirs(work_dir, exist_ok=True)
    distil_path = os.path.join(work_dir, "distillation_sft.jsonl")
    kept = dropped = 0
    with open(distil_path, "w", encoding="utf-8") as out:
        for spec in experts:
            domain_rows = [r for r in rows if not spec.domain or r.get("domain") == spec.domain] or rows
            for row in domain_rows:
                fn = _VERIFIERS.get(str(row.get("verifier") or ""))
                if fn is None:
                    continue
                try:
                    answer = generate_fn(spec, row)
                    score, _ = fn(answer, row.get("reference"), row.get("task", ""))
                except Exception as e:  # noqa: BLE001
                    print(f"[transfer] distillation sample failed: {e}")
                    continue
                if score < verifier_threshold:
                    dropped += 1
                    continue
                out.write(
                    json.dumps(
                        {
                            "prompt": row["prompt"],
                            "completion": [{"role": "assistant", "content": answer}],
                            "domain": spec.domain,
                            "task_id": row.get("task_id", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1

    print(f"[transfer] distillation pool: {kept} verified, {dropped} rejected -> {distil_path}")
    if kept == 0:
        raise RuntimeError("no expert answer passed its verifier; nothing to distil")

    out_dir = os.path.join(work_dir, "target")
    rc = train_sft.main(
        ["--model", target_model, "--dataset", distil_path, "--output_dir", out_dir, *sft_args]
    )
    if rc != 0:
        raise RuntimeError(f"target SFT failed (rc={rc})")
    return out_dir


# ---------------------------------------------------------------------------
# reuse
# ---------------------------------------------------------------------------

def write_manifest(
    path: str,
    *,
    base_model: str,
    experts: Sequence[ExpertSpec],
    transferred: str,
    baseline: Optional[EvalReport] = None,
    final: Optional[EvalReport] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Record what was trained, what was verified, and what was kept."""
    manifest = {
        "base_model": base_model,
        "transferred_checkpoint": transferred,
        "experts": [e.to_dict() for e in experts],
        "kept_experts": [e.domain for e in experts if e.kept],
        "baseline": None if baseline is None else {k: v for k, v in baseline.to_dict().items() if k != "per_task"},
        "final": None if final is None else {k: v for k, v in final.to_dict().items() if k != "per_task"},
    }
    manifest.update(extra or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[transfer] manifest -> {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Specialise domain experts from verified STELLA trajectories.")
    ap.add_argument("--sft", required=True, help="sft.jsonl from self_evolve.dataset_builder")
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--min_samples", type=int, default=64)
    ap.add_argument("--max_domains", type=int, default=0)
    ap.add_argument("--split_only", action="store_true", help="only write per-domain datasets, do not train")
    args, sft_args = ap.parse_known_args(argv)

    if args.split_only:
        split_sft_by_domain(
            args.sft,
            os.path.join(args.work_dir, "domain_data"),
            min_samples=args.min_samples,
            max_domains=args.max_domains,
        )
        return 0

    experts = specialise(
        args.sft,
        args.base_model,
        args.work_dir,
        min_samples=args.min_samples,
        max_domains=args.max_domains,
        sft_args=sft_args,
    )
    print(json.dumps([e.to_dict() for e in experts], indent=2))
    print(
        "\n[transfer] next: verify each expert on held-out tasks (self_evolve.evaluate), then\n"
        "            transfer_by_merge (same base) or transfer_by_distillation (larger target)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
