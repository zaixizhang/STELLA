"""Text conversations with lossless tool-call and reasoning fields."""

import copy
import json


def message_dict(message):
    if isinstance(message, dict):
        out = copy.deepcopy(message)
    elif hasattr(message, "model_dump"):
        out = message.model_dump()
    else:
        out = {key: getattr(message, key, None) for key in
               ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content")}
    out = {key: value for key, value in out.items() if value is not None}
    role = out.get("role", "")
    out["role"] = getattr(role, "value", role)
    out["role"] = {"tool-call": "assistant", "tool-response": "tool"}.get(out["role"], out["role"])
    if out["role"] not in ("system", "user", "assistant", "tool"):
        raise ValueError(f"Unsupported role: {out['role']!r}")
    content = out.get("content", "")
    if isinstance(content, list):
        if any(not isinstance(b, dict) or b.get("type") != "text" for b in content):
            raise ValueError("This recipe accepts text trajectories only; image/video/audio blocks need a multimodal recipe")
        content = "\n".join(b.get("text", "") for b in content)
    if not isinstance(content, str):
        raise ValueError("Message content must be text")
    out["content"] = content
    calls = []
    for call in out.get("tool_calls", []):
        if not isinstance(call, dict):
            call = call.model_dump() if hasattr(call, "model_dump") else vars(call)
        fn = call.get("function", call)
        if not isinstance(fn, dict):
            fn = fn.model_dump() if hasattr(fn, "model_dump") else vars(fn)
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not fn.get("name"):
            raise ValueError("Tool calls need a function name and JSON-object arguments")
        calls.append({"id": call.get("id", ""), "type": "function",
                      "function": {"name": fn["name"], "arguments": arguments}})
    if calls:
        out["tool_calls"] = calls
    else:
        out.pop("tool_calls", None)
    return out


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty messages list")
    return [message_dict(message) for message in messages]


def render_prompt(tokenizer, messages, tools=None):
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=True,
        enable_thinking=False, preserve_thinking=True,
    )


def render_pair(tokenizer, row):
    prompt = render_prompt(tokenizer, row["prompt"], row.get("tools"))
    full = tokenizer.apply_chat_template(
        row["prompt"] + row["completion"], tools=row.get("tools") or None,
        tokenize=False, add_generation_prompt=False,
        enable_thinking=False, preserve_thinking=True,
    )
    if not full.startswith(prompt):
        # Recorded reasoning is a target too; use the model's thinking prefix
        # rather than inserting an empty thinking block ahead of that target.
        prompt = tokenizer.apply_chat_template(
            row["prompt"], tools=row.get("tools") or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=True, preserve_thinking=True,
        )
        if not full.startswith(prompt):
            raise ValueError("Chat-template prompt is not a prefix of the completed conversation; check reasoning/template settings")
    completion = full[len(prompt):]
    if not completion.strip():
        raise ValueError("Empty rendered assistant completion")
    return {"prompt": prompt, "completion": completion}
