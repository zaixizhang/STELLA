"""Opt-in collection of new agent steps for each user turn, without old-step replay."""

import json
from pathlib import Path
import time
import uuid
import warnings

from .messages import message_dict


class InteractionRecorder:
    """Wrap a STELLA manager; use one recorder per conversation.

    Does not judge success or approve training. Human/external labels are supplied
    separately when preparing data. Logging errors never replace an agent error.
    """

    def __init__(self, agent, output_dir, session_id=None, tools=None):
        self.agent = agent
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        self.tools = tools

    def _agents(self):
        pending, agents, seen = [self.agent], [], set()
        while pending:
            agent = pending.pop(0)
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            agents.append(agent)
            managed = getattr(agent, "managed_agents", {}) or {}
            pending.extend(managed.values() if isinstance(managed, dict) else managed)
        return agents

    @staticmethod
    def _steps(agent):
        return list(getattr(getattr(agent, "memory", None), "steps", []) or [])

    def _tool_schemas(self, agent):
        if self.tools is not None:
            return self.tools
        if not getattr(agent, "tools", None):
            return []
        # CodeAgent describes Python tools in its system prompt. Only native
        # tool-calling agents need function schemas in the Qwen chat template.
        from smolagents import ToolCallingAgent
        from smolagents.models import get_tool_json_schema
        if isinstance(agent, ToolCallingAgent):
            return [get_tool_json_schema(tool) for tool in
                    [*agent.tools.values(), *agent.managed_agents.values()]]
        return []

    def run(self, task, **kwargs):
        # Keep references alive across a reset so Python cannot reuse their IDs.
        prior_steps = {id(agent): self._steps(agent) for agent in self._agents()}
        before = {key: {id(step) for step in steps} for key, steps in prior_steps.items()}
        started, result, error = time.time(), None, ""
        try:
            result = self.agent.run(task, **kwargs)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                steps = []
                for agent in self._agents():
                    for step in self._steps(agent):
                        if id(step) in before.get(id(agent), set()) or not getattr(step, "model_input_messages", None):
                            continue
                        output = getattr(step, "model_output_message", None)
                        steps.append({
                            "agent": getattr(agent, "name", "manager"),
                            "tools": self._tool_schemas(agent),
                            "input_messages": [message_dict(m) for m in step.model_input_messages],
                            "output_message": message_dict(output) if output is not None else
                                {"role": "assistant", "content": str(getattr(step, "model_output", "") or "")},
                            "error": str(getattr(step, "error", "") or ""),
                        })
                row = {"record_id": uuid.uuid4().hex, "session_id": self.session_id,
                       "task": task, "steps": steps, "tools": self.tools or [],
                       "final_answer": str(result or ""), "error": error,
                       "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "duration_sec": time.time() - started}
                # A unique file per turn avoids interleaved writes from workers.
                (self.path / f"{row['record_id']}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                warnings.warn(f"STELLA-RSI could not record this turn: {exc}", RuntimeWarning)
