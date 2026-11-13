"""Serve an evolved policy and hand it to STELLA's agents.

Once experience has been written into weights, STELLA has to actually use
those weights. The checkpoint is served behind an OpenAI-compatible endpoint
(vLLM) and the agents' ``model`` objects are repointed at it, so the rest of
STELLA -- tools, skills, critic loop -- is unchanged and the only difference
is which policy is driving.

    python -m self_evolve.serve --model checkpoints/stella-grpo-qwen3-8b --port 8000

then, in a STELLA process::

    from self_evolve.serve import attach_policy_to_agents
    attach_policy_to_agents(api_base="http://localhost:8000/v1", model_id="stella-evolved")

Requires ``pip install vllm`` for serving. Any OpenAI-compatible server
(SGLang, TGI, llama.cpp) works just as well with ``attach_policy_to_agents``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "serve_command",
    "wait_for_server",
    "local_policy_model",
    "attach_policy_to_agents",
    "serve",
]

DEFAULT_API_BASE = os.environ.get("STELLA_LOCAL_POLICY_API_BASE", "http://localhost:8000/v1")
DEFAULT_MODEL_ID = os.environ.get("STELLA_LOCAL_POLICY_MODEL", "stella-evolved")


def serve_command(
    model: str,
    *,
    served_name: str = DEFAULT_MODEL_ID,
    port: int = 8000,
    host: str = "127.0.0.1",
    lora: str = "",
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    extra: Sequence[str] = (),
) -> List[str]:
    """Build the vLLM serve command.

    ``lora`` points at a LoRA adapter directory to serve on top of ``model``,
    which is how a round's adapter is tested without materialising a full copy
    of the weights.
    """
    cmd = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]
    if lora:
        cmd += ["--enable-lora", "--lora-modules", f"{served_name}={lora}"]
    cmd += list(extra)
    return cmd


def wait_for_server(api_base: str = DEFAULT_API_BASE, timeout: float = 900.0, interval: float = 5.0) -> bool:
    """Poll ``/models`` until the server answers. Returns False on timeout."""
    url = api_base.rstrip("/") + "/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - fixed localhost URL
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(interval)
    return False


def local_policy_model(
    api_base: str = DEFAULT_API_BASE,
    model_id: str = DEFAULT_MODEL_ID,
    *,
    temperature: float = 0.1,
    api_key: str = "",
    **kwargs: Any,
) -> Any:
    """A smolagents model object pointing at the locally served policy."""
    from smolagents import OpenAIServerModel  # noqa: PLC0415

    return OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key or os.environ.get("STELLA_LOCAL_POLICY_API_KEY", "EMPTY"),
        temperature=temperature,
        **kwargs,
    )


def attach_policy_to_agents(
    api_base: str = DEFAULT_API_BASE,
    model_id: str = DEFAULT_MODEL_ID,
    *,
    agents: Optional[Sequence[Any]] = None,
    include: Sequence[str] = ("manager_agent", "dev_agent", "critic_agent", "tool_creation_agent"),
) -> Dict[str, str]:
    """Repoint STELLA's agents at the evolved policy.

    Agents hold their model object from construction time, so rebinding the
    module-level variables is not enough: each agent's ``.model`` is replaced,
    and managed sub-agents are followed. Returns {agent name: previous model}
    so a caller can roll back after an evaluation.
    """
    model = local_policy_model(api_base, model_id)
    previous: Dict[str, str] = {}

    targets: List[Any] = list(agents) if agents is not None else []
    if not targets:
        import stella_core  # noqa: PLC0415 - imported lazily so this module stays standalone

        for name in include:
            agent = getattr(stella_core, name, None)
            if agent is not None:
                targets.append(agent)

    seen: set = set()

    def _attach(agent: Any) -> None:
        if id(agent) in seen:
            return
        seen.add(id(agent))
        name = str(getattr(agent, "name", None) or type(agent).__name__)
        old = getattr(agent, "model", None)
        previous[name] = str(getattr(old, "model_id", old))
        agent.model = model
        managed = getattr(agent, "managed_agents", None) or {}
        if isinstance(managed, dict):
            managed = managed.values()
        for sub in managed or []:
            _attach(sub)

    for agent in targets:
        _attach(agent)

    print(f"[serve] attached {model_id} ({api_base}) to: {', '.join(previous) or 'nothing'}")
    return previous


def serve(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Serve an evolved STELLA policy over an OpenAI-compatible API.")
    ap.add_argument("--model", required=True, help="checkpoint dir or HF id")
    ap.add_argument("--lora", default="", help="LoRA adapter dir to serve on top of --model")
    ap.add_argument("--served_name", default=DEFAULT_MODEL_ID)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--print_only", action="store_true", help="print the command instead of running it")
    args, extra = ap.parse_known_args(argv)

    cmd = serve_command(
        args.model,
        served_name=args.served_name,
        port=args.port,
        host=args.host,
        lora=args.lora,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        extra=extra,
    )
    print("[serve] " + " ".join(shlex.quote(c) for c in cmd))
    if args.print_only:
        return 0
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[serve] vllm not found. Install it with: pip install vllm")
        return 1


if __name__ == "__main__":
    raise SystemExit(serve())
