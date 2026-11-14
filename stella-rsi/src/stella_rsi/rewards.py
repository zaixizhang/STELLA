"""Recompute rewards on newly sampled completions; historical votes are not rewards."""

import json
import math
import re

VERIFIERS = {"exact_match", "contains", "numeric", "tool_call"}


def final_text(completion):
    if isinstance(completion, list):
        completion = "\n".join(m.get("content") or "" for m in completion)
    return re.sub(r"<think>.*?</think>", "", str(completion), flags=re.S).strip()


def validate_reference(verifier, reference):
    if verifier not in VERIFIERS:
        raise ValueError(f"Unknown verifier {verifier!r}; choose from {sorted(VERIFIERS)}")
    if verifier == "numeric":
        values = reference if isinstance(reference, dict) else {"value": reference}
        for key in ("value", "atol", "rtol"):
            value = float(values.get(key, 0.0))
            if not math.isfinite(value) or (key != "value" and value < 0):
                raise ValueError("Numeric references and tolerances must be finite, tolerances nonnegative")
        if "value" not in values:
            raise ValueError("Numeric reference needs value")
    elif verifier == "tool_call":
        if not isinstance(reference, dict) or not reference.get("name") or not isinstance(reference.get("arguments"), dict):
            raise ValueError("tool_call reference needs name and arguments")
    elif not isinstance(reference, str) or not reference.strip():
        raise ValueError("Text verifiers require a nonempty reference string")


def parse_tool_call(block, reference):
    """Accept JSON tool calls and Qwen3.5/3.8's function/parameter wire format.

    Parameter types come from the labelled reference: Qwen serializes string
    arguments without JSON quotes, including strings that look like numbers.
    No tool is executed during parsing or reward evaluation.
    """
    if not block.lstrip().startswith("<function="):
        return json.loads(block)
    match = re.fullmatch(r"\s*<function=([^<>\s]+)>\s*(.*?)\s*</function>\s*", block, re.S)
    if not match:
        raise ValueError("Malformed Qwen function call")
    name, body = match.groups()
    pattern = r"<parameter=([^<>\s]+)>\s*\n?(.*?)\n?\s*</parameter>"
    if re.sub(pattern, "", body, flags=re.S).strip():
        raise ValueError("Unexpected function-call content")
    arguments = {}
    for key, raw in re.findall(pattern, body, re.S):
        if key in arguments:
            raise ValueError("Duplicate tool argument")
        raw = raw.strip()
        if isinstance(reference["arguments"].get(key), str):
            arguments[key] = raw
        else:
            arguments[key] = json.loads(raw)
    return {"name": name, "arguments": arguments}


def score(completion, verifier, reference):
    validate_reference(verifier, reference)
    text = final_text(completion)
    norm = lambda s: " ".join(s.lower().split())
    if verifier == "exact_match":
        return float(norm(text) == norm(reference))
    if verifier == "contains":
        return float(norm(reference) in norm(text))
    if verifier == "numeric":
        ref = reference if isinstance(reference, dict) else {"value": reference}
        numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
        return float(bool(numbers) and math.isclose(float(numbers[-1]), float(ref["value"]),
                     abs_tol=float(ref.get("atol", 0)), rel_tol=float(ref.get("rtol", 0.01))))
    calls = []
    if isinstance(completion, list):
        calls = [c.get("function", c) for m in completion for c in m.get("tool_calls", [])]
    if not calls:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
        if not blocks:
            blocks = [text]
        try:
            calls = [parse_tool_call(block, reference) for block in blocks]
        except (ValueError, TypeError):
            return 0.0
    if len(calls) != 1 or not isinstance(calls[0], dict):
        return 0.0
    call = calls[0]
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return 0.0
    return float(call.get("name") == reference["name"] and arguments == reference["arguments"])


def outcome_reward(completions, verifier, reference_json, **kwargs):
    if len(completions) != len(verifier) or len(completions) != len(reference_json):
        raise ValueError("Reward metadata must align with the generated completions")
    return [score(c, v, json.loads(r)) for c, v, r in zip(completions, verifier, reference_json)]
