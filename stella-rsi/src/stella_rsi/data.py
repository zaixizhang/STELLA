"""Prepare reviewed interaction decisions, split by session before expansion."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .messages import message_dict, normalize_messages
from .rewards import validate_reference


def read_jsonl(path):
    paths = sorted(Path(path).rglob("*.jsonl")) if Path(path).is_dir() else [Path(path)]
    if not paths:
        raise ValueError(f"No JSONL files found under {path}")
    rows = []
    for file in paths:
        for index, line in enumerate(file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Expected a JSON object")
                rows.append(obj)
            except ValueError as exc:
                raise ValueError(f"{file}:{index}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def decisions(record):
    """Indices are message positions for chat exports, step positions for native logs."""
    if "messages" in record:
        messages = normalize_messages(record["messages"])
        return {i: {"prompt": messages[:i], "completion": [m]}
                for i, m in enumerate(messages) if i and m["role"] == "assistant"}
    result = {}
    for i, step in enumerate(record.get("steps", [])):
        if step.get("error"):
            continue
        inputs = step.get("input_messages", [])
        output = step.get("output_message") or {"role": "assistant", "content": step.get("output_text", "")}
        if inputs and (output.get("content") or output.get("tool_calls")):
            message = message_dict(output)
            if message["role"] != "assistant":
                raise ValueError("A decision output must have the assistant role")
            result[i] = {"prompt": normalize_messages(inputs), "completion": [message],
                         "tools": step.get("tools", record.get("tools", []))}
    return result


def user_texts(record, items):
    texts = {" ".join(m["content"].lower().split())
             for item in items.values() for m in item["prompt"] if m["role"] == "user"}
    if record.get("task"):
        texts.add(" ".join(record["task"].lower().split()))
    return texts - {""}


def prepare(input_path, labels_path, splits_path, output_dir):
    records = read_jsonl(input_path)
    labels = json.loads(Path(labels_path).read_text())
    splits = json.loads(Path(splits_path).read_text())
    if not isinstance(labels, dict) or not isinstance(splits, dict):
        raise ValueError("Labels and splits must be JSON objects")
    if set(splits) - {"validation", "test"}:
        raise ValueError("Splits accepts validation and test session-ID lists; remaining sessions are train")
    if any(not isinstance(v, list) or any(not isinstance(s, str) for s in v) for v in splits.values()):
        raise ValueError("Split values must be lists of session-ID strings")
    validation, test = set(splits.get("validation", [])), set(splits.get("test", []))
    if not validation and not test:
        raise ValueError("At least one held-out session is required")
    if validation & test:
        raise ValueError("A session cannot belong to both validation and test")
    prepared, seen = [], set()
    for record in records:
        rid = record.get("record_id") or record.get("trajectory_id")
        session = record.get("session_id")
        if not rid or not session:
            raise ValueError("Every record needs record_id (or trajectory_id) and session_id; map legacy runs to sessions explicitly")
        if rid in seen:
            raise ValueError(f"Duplicate record ID: {rid}")
        seen.add(rid)
        items = decisions(record)
        split = "test" if session in test else "validation" if session in validation else "train"
        prepared.append((record, rid, session, split, items, user_texts(record, items)))
    unknown_labels = set(labels) - seen
    if unknown_labels:
        raise ValueError(f"Labels reference unknown records: {sorted(unknown_labels)}")
    missing = (validation | test) - {p[2] for p in prepared}
    if missing:
        raise ValueError(f"Held-out sessions missing from input: {sorted(missing)}")
    test_tasks = set().union(*(p[5] for p in prepared if p[3] == "test"))
    heldout_tasks = test_tasks | set().union(*(p[5] for p in prepared if p[3] == "validation"))
    output = {f"{kind}_{split}": [] for kind in ("sft", "grpo") for split in ("train", "validation", "test")}
    stats = Counter(records=len(records))
    fingerprints = set()
    for record, rid, session, split, items, tasks in prepared:
        blocked = heldout_tasks if split == "train" else test_tasks if split == "validation" else set()
        if tasks & blocked:
            stats["excluded_task_overlap"] += 1
            continue
        label = labels.get(rid, {})
        if not isinstance(label, dict):
            raise ValueError(f"Labels for {rid} must be an object")
        approved = label.get("approved_decisions", [])
        if label.get("sft_approved") is True:
            approved = list(items)
        if not isinstance(approved, list) or any(type(i) is not int or i not in items for i in approved):
            raise ValueError(f"Invalid approved decision indices for {rid}")
        if record.get("error"):
            stats["excluded_failed_records"] += 1
            continue
        metadata = {"record_id": rid, "session_id": session, "tools": record.get("tools", [])}
        for i in approved:
            item = items[i]
            fingerprint = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
            if (split, fingerprint) in fingerprints:
                continue
            fingerprints.add((split, fingerprint))
            output[f"sft_{split}"].append({**metadata, **item, "decision_index": i})
        for target in label.get("grpo_targets", []):
            index = target["decision_index"]
            if type(index) is not int or index not in items:
                raise ValueError(f"Invalid GRPO decision index for {rid}: {index}")
            verifier, reference = target["verifier"], target["reference"]
            validate_reference(verifier, reference)
            output[f"grpo_{split}"].append({
                **metadata, "tools": items[index].get("tools", metadata["tools"]),
                "decision_index": index, "prompt": items[index]["prompt"],
                "verifier": verifier, "reference_json": json.dumps(reference, ensure_ascii=False),
            })
    if not output["sft_train"] or not output["grpo_train"]:
        raise ValueError("Need reviewed SFT decisions and verifier-labelled GRPO prompts in training sessions")
    for name, rows in output.items():
        write_jsonl(Path(output_dir) / f"{name}.jsonl", rows)
        stats[name] = len(rows)
    stats["sessions"] = len({p[2] for p in prepared})
    manifest = {"counts": dict(stats), "splits": splits,
                "input_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                "labels_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()}
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.input, args.labels, args.splits, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
