"""Tests for the self-evolution data pipeline.

These run without a GPU and without trl/torch: they cover the parts that
decide what the optimiser sees -- verification, the verified gate, best-of-n
selection, decontamination and the dataset formats. The training scripts are
exercised through their ``--dry_run`` paths.

    python -I -m self_evolve.tests.test_pipeline       # from the STELLA root
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from self_evolve.dataset_builder import (  # noqa: E402
    BuildConfig,
    ContaminationFilter,
    build_grpo_samples,
    build_sft_samples,
    build_datasets,
)
from self_evolve.reward import RewardWeights, score_trajectory, verify_trajectories  # noqa: E402
from self_evolve.schema import (  # noqa: E402
    StepRecord,
    ToolCallRecord,
    Trajectory,
    append_trajectory,
    iter_trajectories,
    new_trajectory_id,
)

PASSED = 0
FAILED = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED += 1
        print(f"  FAIL {label} {detail}")


def make_traj(
    task: str,
    answer: str,
    *,
    task_id: str = "",
    domain: str = "immunology",
    n_steps: int = 3,
    error_step: bool = False,
    tools: tuple = ("query_pubmed",),
    split: str = "train",
) -> Trajectory:
    steps = []
    for i in range(n_steps):
        steps.append(
            StepRecord(
                step_number=i + 1,
                agent="manager",
                input_messages=[
                    {"role": "system", "content": "You are STELLA, a biomedical research agent."},
                    {"role": "user", "content": task},
                    {"role": "tool", "content": f"observation {i}" * 3},
                ],
                output_text=f"Thought: step {i}. I will call a tool.\nAction: {tools[0]}(query='x')",
                tool_calls=[ToolCallRecord(name=tools[0], arguments={"query": "x"})],
                observations=f"observation {i}",
                error="boom" if (error_step and i == 1) else "",
                is_final_answer=(i == n_steps - 1),
            )
        )
    return Trajectory(
        trajectory_id=new_trajectory_id(),
        task=task,
        task_id=task_id,
        domain=domain,
        split=split,
        steps=steps,
        final_answer=answer,
        success=True,
    )


def test_verifiers() -> None:
    print("\n[test] verifiers")
    t = make_traj("Find X", "The answer is BTN3A1")
    r = score_trajectory(t, reference="BTN3A1", verifier="contains")
    check("contains passes", r.verified and r.outcome == 1.0, f"got {r.outcome}")

    r = score_trajectory(t, reference="FOXP3", verifier="contains")
    check("contains fails on wrong target", not r.verified and r.outcome == 0.0)

    ranked = make_traj("rank genes", "1. BTN3A1\n2. GPR174\n3. KLRE1")
    r = score_trajectory(ranked, reference={"target": "BTN3A1", "top_k": 10}, verifier="ranked_contains")
    check("ranked_contains rewards rank 1", r.outcome == 1.0, f"got {r.outcome}")

    ranked_low = make_traj("rank genes", "\n".join(f"{i}. GENE{i}" for i in range(1, 20)) + "\n20. BTN3A1")
    r = score_trajectory(ranked_low, reference={"target": "BTN3A1", "top_k": 10}, verifier="ranked_contains")
    check("ranked_contains penalises deep rank", 0 < r.outcome < 0.5, f"got {r.outcome}")

    r = score_trajectory(make_traj("q", "B) option"), reference="B", verifier="mcq")
    check("mcq", r.outcome == 1.0)

    r = score_trajectory(make_traj("q", "the value is 42.1"), reference=42.0, verifier="numeric")
    check("numeric within 1%", r.outcome == 1.0, f"got {r.outcome}")

    r = score_trajectory(make_traj("q", "the value is 99"), reference=42.0, verifier="numeric")
    check("numeric outside tolerance", r.outcome == 0.0)

    r = score_trajectory(make_traj("q", '[{"gene": "BTN3A1", "rank": 1}]'), reference=["gene", "rank"], verifier="json_valid")
    check("json_valid with required keys", r.outcome == 1.0)

    # No verifier at all must never admit a trajectory.
    r = score_trajectory(make_traj("q", "anything"))
    check("unverifiable run is not admitted", not r.verified and r.outcome == 0.0)

    # A trajectory the agent called a success but the verifier rejects.
    r = score_trajectory(make_traj("q", "wrong answer"), reference="BTN3A1", verifier="contains")
    check("agent-reported success cannot override verifier", not r.verified)

    # Shaping terms.
    clean = score_trajectory(make_traj("q", "BTN3A1"), reference="BTN3A1", verifier="contains")
    broken = score_trajectory(
        make_traj("q", "BTN3A1", error_step=True), reference="BTN3A1", verifier="contains"
    )
    check("error steps lower the reward", broken.total < clean.total, f"{broken.total} vs {clean.total}")

    unknown_tool = make_traj("q", "BTN3A1", tools=("not_a_real_tool",))
    r_unknown = score_trajectory(
        unknown_tool, reference="BTN3A1", verifier="contains", tool_catalog=["query_pubmed"]
    )
    check("hallucinated tool lowers the reward", r_unknown.total < clean.total)

    try:
        score_trajectory(make_traj("q", "x"), reference="x", verifier="does_not_exist")
        check("unknown verifier raises", False)
    except KeyError:
        check("unknown verifier raises", True)


def test_admission_and_best_of_n() -> None:
    print("\n[test] admission gate and best-of-n")
    good = make_traj("Task A", "BTN3A1", task_id="A", n_steps=3)
    mediocre = make_traj("Task A", "BTN3A1", task_id="A", n_steps=9, error_step=True)
    wrong = make_traj("Task A", "nothing useful", task_id="A")
    trajs = [good, mediocre, wrong]
    verify_trajectories(trajs, references={"A": "BTN3A1"}, default_verifier="contains")

    check("verified flags set", [t.verified for t in trajs] == [True, True, False])

    samples = build_sft_samples(trajs, BuildConfig(max_per_task=1, allow_no_heldout=True))
    ids = {s["trajectory_id"] for s in samples}
    check("best-of-n keeps only the top attempt", ids == {good.trajectory_id}, f"got {ids}")
    check("unverified trajectory excluded", wrong.trajectory_id not in ids)

    samples2 = build_sft_samples(trajs, BuildConfig(max_per_task=2, allow_no_heldout=True))
    check("max_per_task=2 keeps two attempts", len({s["trajectory_id"] for s in samples2}) == 2)

    # An error step is dropped from an otherwise usable trajectory; the rest of
    # that trajectory still teaches. Checked with dedup off so the count
    # isolates error dropping from cross-attempt deduplication.
    no_dedup = build_sft_samples(
        [mediocre], BuildConfig(max_per_task=2, allow_no_heldout=True, dedup=False)
    )
    check("error step dropped, rest kept", len(no_dedup) == mediocre.n_steps - 1, f"got {len(no_dedup)}")

    # With dedup on, the three steps `mediocre` shares with `good` collapse, so
    # the second attempt only contributes what is new.
    err_samples = [s for s in samples2 if s["trajectory_id"] == mediocre.trajectory_id]
    check(
        "duplicate steps across attempts collapse",
        len(err_samples) == mediocre.n_steps - 1 - (good.n_steps - 1),
        f"got {len(err_samples)}",
    )

    held = make_traj("Task H", "BTN3A1", task_id="H", split="heldout")
    verify_trajectories([held], references={"H": "BTN3A1"}, default_verifier="contains")
    check(
        "heldout-split trajectory never enters training",
        build_sft_samples([held], BuildConfig(allow_no_heldout=True)) == [],
    )


def test_sft_format() -> None:
    print("\n[test] SFT sample format")
    t = make_traj("Task B", "BTN3A1", task_id="B")
    verify_trajectories([t], references={"B": "BTN3A1"}, default_verifier="contains")
    samples = build_sft_samples([t], BuildConfig(allow_no_heldout=True))
    check("one sample per step", len(samples) == t.n_steps, f"got {len(samples)}")

    s = samples[0]
    check("prompt is a message list", isinstance(s["prompt"], list) and s["prompt"][0]["role"] == "system")
    check("completion is one assistant turn", len(s["completion"]) == 1 and s["completion"][0]["role"] == "assistant")
    check("prompt excludes the completion", all(s["completion"][0]["content"] != m["content"] for m in s["prompt"]))
    check("roles are chat-template safe", all(m["role"] in ("system", "user", "assistant", "tool") for m in s["prompt"]))

    # Long prompts are trimmed but keep the system turn and the latest turn.
    big = make_traj("Task C", "BTN3A1", task_id="C", n_steps=1)
    big.steps[0].input_messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "tool", "content": "x" * 50_000},
        {"role": "user", "content": "LATEST"},
    ]
    verify_trajectories([big], references={"C": "BTN3A1"}, default_verifier="contains")
    trimmed = build_sft_samples([big], BuildConfig(allow_no_heldout=True, max_prompt_chars=5_000))[0]["prompt"]
    total = sum(len(m["content"]) for m in trimmed)
    check("long prompt trimmed", total <= 5_000, f"got {total}")
    check("system turn kept", trimmed[0]["content"] == "SYSTEM")
    check("latest turn kept", trimmed[-1]["content"] == "LATEST")

    # Duplicate steps collapse.
    d1 = make_traj("Task D", "BTN3A1", task_id="D", n_steps=2)
    d2 = make_traj("Task D", "BTN3A1", task_id="D2", n_steps=2)
    verify_trajectories([d1, d2], references={"D": "BTN3A1", "D2": "BTN3A1"}, default_verifier="contains")
    deduped = build_sft_samples([d1, d2], BuildConfig(allow_no_heldout=True, dedup=True))
    check("identical steps deduplicated", len(deduped) == 2, f"got {len(deduped)}")


def test_decontamination() -> None:
    print("\n[test] decontamination")
    heldout_task = "Identify novel negative regulators of NK-cell function in the context of AML"
    cf = ContaminationFilter([heldout_task], ngram=8)
    check("exact match caught", cf.is_contaminated(heldout_task))
    check("case/space variation caught", cf.is_contaminated("  identify NOVEL negative regulators of NK-cell   function in the context of AML "))
    check("long overlap caught", cf.is_contaminated("Please identify novel negative regulators of NK-cell function in the context of AML and rank them"))
    check("unrelated task passes", not cf.is_contaminated("Optimize the catalytic activity of strictosidine synthase"))

    leaky = make_traj(heldout_task, "BTN3A1", task_id="L")
    clean = make_traj("Optimize strictosidine synthase activity", "M276L", task_id="K", domain="enzyme")
    verify_trajectories(
        [leaky, clean], references={"L": "BTN3A1", "K": "M276L"}, default_verifier="contains"
    )
    cfg = BuildConfig(heldout_tasks=(heldout_task,))
    samples = build_sft_samples([leaky, clean], cfg)
    check("contaminated trajectory excluded", all(s["task_id"] != "L" for s in samples))
    check("clean trajectory kept", any(s["task_id"] == "K" for s in samples))

    try:
        build_datasets("/nonexistent", tempfile.mkdtemp(), BuildConfig())
        check("builder refuses without a held-out set", False)
    except ValueError:
        check("builder refuses without a held-out set", True)


def test_grpo_rows() -> None:
    print("\n[test] GRPO rows")
    a = make_traj("Task E", "BTN3A1", task_id="E")
    b = make_traj("Task F", "x", task_id="F")
    verify_trajectories([a, b], references={"E": "BTN3A1"}, default_verifier="contains")
    rows = build_grpo_samples(
        [a, b], BuildConfig(allow_no_heldout=True), references={"E": "BTN3A1"}, tool_catalog=["query_pubmed"]
    )
    check("only verifiable tasks become prompts", len(rows) == 1 and rows[0]["task_id"] == "E", f"got {rows}")
    check("prompt carries system + user", [m["role"] for m in rows[0]["prompt"]] == ["system", "user"])
    check("reward metadata present", rows[0]["verifier"] == "contains" and rows[0]["reference"] == "BTN3A1")
    check("tool catalogue attached", rows[0]["tool_catalog"] == ["query_pubmed"])


def test_roundtrip_and_build() -> None:
    print("\n[test] JSONL round-trip and build_datasets")
    with tempfile.TemporaryDirectory() as tmp:
        traj_dir = os.path.join(tmp, "trajectories")
        t = make_traj("Task G", "BTN3A1", task_id="G")
        path = append_trajectory(t, path=os.path.join(traj_dir, "t.jsonl"))
        check("append wrote a file", os.path.exists(path))

        with open(path, "a", encoding="utf-8") as f:
            f.write("{ this is not json\n")  # a crash mid-write must not kill training
        loaded = list(iter_trajectories(traj_dir))
        check("corrupt line skipped", len(loaded) == 1, f"got {len(loaded)}")
        check("round-trip preserves steps", loaded[0].n_steps == t.n_steps)
        check("round-trip preserves tool calls", loaded[0].tool_names == t.tool_names)
        check("round-trip preserves answer", loaded[0].final_answer == "BTN3A1")

        out = build_datasets(
            traj_dir,
            os.path.join(tmp, "datasets"),
            BuildConfig(heldout_tasks=("some other task",)),
            references={"G": "BTN3A1"},
            default_verifier="contains",
        )
        check("sft.jsonl written", os.path.exists(out["paths"]["sft"]))
        check("grpo.jsonl written", os.path.exists(out["paths"]["grpo"]))
        check("stats recorded", out["stats"]["verified"] == 1 and out["stats"]["sft_samples"] > 0)
        check("stats store heldout count, not tasks", out["stats"]["config"]["heldout_tasks"] == 1)

        with open(out["paths"]["sft"], "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        check("sft rows parse", len(rows) == out["stats"]["sft_samples"])

        from self_evolve.train_sft import load_sft_rows  # noqa: PLC0415

        trl_rows = load_sft_rows(out["paths"]["sft"])
        check("trainer loader accepts them", len(trl_rows) == len(rows))
        check("trainer rows have exactly prompt+completion", all(set(r) == {"prompt", "completion"} for r in trl_rows))

        from self_evolve.train_grpo import load_grpo_rows  # noqa: PLC0415

        grpo_rows = load_grpo_rows(out["paths"]["grpo"])
        check("grpo loader accepts them", len(grpo_rows) == 1)


def test_reward_functions() -> None:
    print("\n[test] GRPO reward functions")
    from self_evolve.train_grpo import extract_tool_names, format_reward, outcome_reward, tool_validity_reward

    comp = [{"role": "assistant", "content": "Thought: search.\nAction: query_pubmed(query='BTN3A1 NK AML')"}]
    check(
        "tool name extracted from code action",
        "query_pubmed" in extract_tool_names("query_pubmed(query='x')"),
    )
    check(
        "tool name extracted from JSON action",
        "query_kegg" in extract_tool_names('```json\n{"name": "query_kegg", "arguments": {}}\n```'),
    )

    scores = outcome_reward([comp], reference=["BTN3A1"], verifier=["contains"], task=["t"])
    check("outcome reward uses the verifier", scores == [1.0], f"got {scores}")
    scores = outcome_reward([comp], reference=["FOXP3"], verifier=["contains"], task=["t"])
    check("outcome reward is zero when wrong", scores == [0.0])
    scores = outcome_reward([comp], reference=["BTN3A1"], verifier=[""], task=["t"])
    check("no verifier scores zero", scores == [0.0])

    good = tool_validity_reward([comp], tool_catalog=[["query_pubmed"]])
    bad = tool_validity_reward(
        [{"role": "assistant", "content": "made_up_tool(x=1)"}], tool_catalog=[["query_pubmed"]]
    )
    check("tool validity rewards real tools", good == [1.0], f"got {good}")
    check("tool validity punishes invented tools", bad == [0.0], f"got {bad}")

    f = format_reward([comp, {"role": "assistant", "content": ""}])
    check("format reward separates full from empty", f[0] > f[1], f"got {f}")


def test_trajectory_logger_on_fake_agent() -> None:
    print("\n[test] trajectory extraction from an agent-like object")
    from self_evolve.trajectory_logger import extract_trajectory

    class FakeTiming:
        start_time, end_time = 0.0, 1.5

    class FakeUsage:
        input_tokens, output_tokens = 120, 30

    class FakeToolCall:
        name, arguments, id = "query_pubmed", {"query": "x"}, "call_1"

    class FakeMessage:
        role, content, tool_calls = "assistant", [{"type": "text", "text": "Thought: go"}], None

    class FakeActionStep:
        step_number = 1
        timing = FakeTiming()
        model_input_messages = [{"role": "user", "content": "do it"}]
        tool_calls = [FakeToolCall()]
        error = None
        model_output_message = FakeMessage()
        model_output = "Thought: go"
        code_action = ""
        observations = "found 3 papers"
        is_final_answer = True
        token_usage = FakeUsage()

    class FakeTaskStep:  # no model output: must be skipped
        task = "do it"

    class FakeMemory:
        steps = [FakeTaskStep(), FakeActionStep()]

    class FakeModel:
        model_id = "anthropic/claude-sonnet-4"

    class FakeAgent:
        name = "manager"
        memory = FakeMemory()
        model = FakeModel()
        managed_agents: dict = {}

    traj = extract_trajectory(FakeAgent(), "do it", "BTN3A1", domain="immunology", task_id="T1")
    check("task steps skipped", traj.n_steps == 1, f"got {traj.n_steps}")
    check("policy model recorded", traj.policy_model == "anthropic/claude-sonnet-4")
    check("content blocks flattened", traj.steps[0].output_message["content"] == "Thought: go")
    check("tool call captured", traj.tool_names == ["query_pubmed"])
    check("timing derived", traj.steps[0].duration_sec == 1.5)
    check("tokens captured", traj.steps[0].input_tokens == 120)
    check("metadata applied", traj.domain == "immunology" and traj.task_id == "T1")
    check("agent success heuristic", traj.success is True)


def main() -> int:
    test_verifiers()
    test_admission_and_best_of_n()
    test_sft_format()
    test_decontamination()
    test_grpo_rows()
    test_roundtrip_and_build()
    test_reward_functions()
    test_trajectory_logger_on_fake_agent()
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
