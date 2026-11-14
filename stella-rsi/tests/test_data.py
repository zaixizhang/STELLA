import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stella_rsi.capture import InteractionRecorder
from stella_rsi.data import decisions, prepare, read_jsonl
from stella_rsi.messages import message_dict
from stella_rsi.rewards import outcome_reward, score

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def fixture_files(tmp_path):
    for name in ("interactions.jsonl", "labels.json", "splits.json"):
        (tmp_path / name).write_bytes((EXAMPLES / name).read_bytes())
    return tmp_path / "interactions.jsonl", tmp_path / "labels.json", tmp_path / "splits.json"


def test_session_splits_context_and_explicit_reviews(tmp_path):
    paths = fixture_files(tmp_path)
    result = prepare(*paths, tmp_path / "out")
    assert result["counts"]["sft_train"] == 4
    assert result["counts"]["grpo_train"] == 5
    sft = read_jsonl(tmp_path / "out/sft_train.jsonl")
    last = next(r for r in sft if r["record_id"] == "mean-turns" and r["decision_index"] == 6)
    assert len(last["prompt"]) == 6
    assert last["prompt"][2]["tool_calls"][0]["function"]["arguments"] == {"values": [2, 4, 6]}
    assert last["prompt"][3]["tool_call_id"] == "call-1"
    assert not any(r["record_id"] == "wrong-attempt" for r in sft)
    grpo = read_jsonl(tmp_path / "out/grpo_train.jsonl")
    wrong = next(r for r in grpo if r["record_id"] == "wrong-attempt")
    assert wrong["reference_json"] == "21"
    assert "completion" not in wrong
    split_ids = [{r["session_id"] for r in read_jsonl(tmp_path / f"out/sft_{split}.jsonl")}
                 for split in ("train", "validation", "test")]
    assert not split_ids[0] & split_ids[1] and not split_ids[0] & split_ids[2]


def test_same_user_task_cannot_leak_from_heldout(tmp_path):
    paths = fixture_files(tmp_path)
    rows = read_jsonl(paths[0])
    rows[1]["messages"][0] = rows[3]["messages"][0]
    paths[0].write_text("\n".join(json.dumps(row) for row in rows))
    result = prepare(*paths, tmp_path / "out")
    assert result["counts"]["excluded_task_overlap"] == 1
    assert all(row["record_id"] != "units" for row in read_jsonl(tmp_path / "out/grpo_train.jsonl"))


def test_split_and_annotation_mistakes_fail(tmp_path):
    paths = fixture_files(tmp_path)
    paths[2].write_text('{}')
    with pytest.raises(ValueError, match="held-out"):
        prepare(*paths, tmp_path / "out")
    paths[2].write_text('{"validation":["not-collected"]}')
    with pytest.raises(ValueError, match="missing from input"):
        prepare(*paths, tmp_path / "out")


def test_native_steps_preserve_call_only_completions():
    row = {"steps": [{"input_messages": [{"role": "user", "content": "look it up"}],
                      "output_message": {"role": "assistant", "tool_calls": [
                          {"name": "query", "arguments": '{"gene":"TP53"}'}]}}]}
    target = decisions(row)[0]["completion"][0]
    assert target["tool_calls"][0]["function"] == {"name": "query", "arguments": {"gene": "TP53"}}


def test_multimodal_content_is_not_silently_discarded():
    with pytest.raises(ValueError, match="text trajectories only"):
        message_dict({"role": "user", "content": [{"type": "image", "url": "image.png"}]})


def test_rewards_score_new_answers_and_actual_tool_arguments():
    assert outcome_reward(["21", "20"], ["numeric"] * 2, ["21"] * 2) == [1.0, 0.0]
    reference = {"name": "mean", "arguments": {"values": [2, 4, 6]}}
    assert score('<tool_call>{"name":"mean","arguments":{"values":[2,4,6]}}</tool_call>', "tool_call", reference) == 1
    assert score('<tool_call>{"name":"mean","arguments":{"values":[2]}}</tool_call>', "tool_call", reference) == 0
    xml = '<tool_call>\n<function=mean>\n<parameter=values>\n[2, 4, 6]\n</parameter>\n</function>\n</tool_call>'
    assert score(xml, "tool_call", reference) == 1
    assert score(xml.replace('[2, 4, 6]', '[2]'), "tool_call", reference) == 0
    assert score(xml + xml, "tool_call", reference) == 0
    assert score(xml.replace('</parameter>', ''), "tool_call", reference) == 0
    assert score('<think>wrong earlier idea</think>21', "exact_match", "21") == 1
    with pytest.raises(ValueError):
        score("anything", "contains", "")


def test_recorder_excludes_prior_steps_and_preserves_exceptions(tmp_path):
    class Agent:
        name = "manager"
        managed_agents = {}

        def __init__(self):
            self.memory = SimpleNamespace(steps=[])

        def run(self, task, **kwargs):
            if task == "fail":
                raise RuntimeError("original error")
            self.memory.steps.append(SimpleNamespace(
                model_input_messages=[{"role": "user", "content": task}],
                model_output_message={"role": "assistant", "content": task.upper()}))
            return task.upper()

    recorder = InteractionRecorder(Agent(), tmp_path, session_id="one-session")
    assert recorder.run("first", reset=False) == "FIRST"
    assert recorder.run("second", reset=False) == "SECOND"
    with pytest.raises(RuntimeError, match="original error"):
        recorder.run("fail")
    rows = read_jsonl(tmp_path)
    assert len(rows) == 3
    assert {r["session_id"] for r in rows} == {"one-session"}
    assert sorted(len(r["steps"]) for r in rows) == [0, 1, 1]
    assert next(r for r in rows if r["task"] == "fail")["error"]
