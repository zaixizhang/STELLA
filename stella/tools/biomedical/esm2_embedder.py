"""
ESM2 Embedder Tool

Provides a reusable @tool that computes and caches ESM-2 embeddings for protein sequences.

Key features:
- Loads ESM-2 models via esm.pretrained using a provided model_name (e.g., "esm2_t33_650M_UR50D").
- Mean-pools final-layer token representations, excluding padding/BOS/EOS tokens.
- Per-sequence on-disk caching using SHA1 of sequence under {cache_dir}/{model_name}/{sha1}.npy.
- Robust device selection (auto/cuda/cpu) and error handling.
- Basic token validation/cleaning (replaces unknown tokens with 'X' and warns).
- Returns a dict with embeddings (np.ndarray float32), model_name, final layer index, and seq_hashes.

After creation: runs a small self-test if executed as a script and attempts to register the tool using
`add_tool_to_agents` if available in the runtime environment.
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings
from typing import List, Dict, Any

# Third-party imports (lazy-install fair-esm, smolagents if missing)

def _ensure_package(pkg: str, pip_name: str | None = None) -> None:
    try:
        __import__(pkg)
    except Exception:
        import subprocess
        pipn = pip_name or pkg
        print(f"[esm2_embedder] Installing missing dependency: {pipn} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pipn])


# Ensure dependencies
try:
    from smolagents import tool
except Exception:
    _ensure_package("smolagents")
    from smolagents import tool

try:
    import torch
except Exception as e:
    raise RuntimeError(
        "PyTorch is required to run ESM models. Please install a compatible version of torch first."
    ) from e

try:
    import esm  # from fair-esm
except Exception:
    _ensure_package("esm", "fair-esm")
    import esm

import numpy as np


def _select_device(device: str) -> str:
    d = device.lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if d not in {"cuda", "cpu"}:
        warnings.warn(f"Unknown device '{device}', defaulting to 'cpu'.")
        return "cpu"
    return d


def _clean_sequence(seq: str) -> tuple[str, int]:
    """Clean and validate a protein sequence.

    Returns (cleaned_sequence, num_replacements). Unrecognized tokens are replaced with 'X'.
    """
    allowed = set("ACDEFGHIKLMNPQRSTVWYBZXUOJ")  # common AA tokens + B,Z,X,U,O,J
    s = (seq or "").strip().upper()
    cleaned = []
    repl = 0
    for ch in s:
        if ch in allowed:
            cleaned.append(ch)
        else:
            cleaned.append('X')
            repl += 1
    return ("".join(cleaned), repl)


def _get_model_and_alphabet(model_name: str):
    # esm.pretrained.<name>() should return (model, alphabet)
    fn = getattr(esm.pretrained, model_name, None)
    if fn is None or not callable(fn):
        # Build a short list of available ESM2 models to help the user
        available = [n for n in dir(esm.pretrained) if n.startswith("esm2_")]
        raise ValueError(
            f"Invalid model_name '{model_name}'. Available esm2_* entries include: {available[:10]}..."
        )
    model, alphabet = fn()
    return model, alphabet


def _infer_final_layer(model, model_name: str) -> int:
    # ESM models usually expose num_layers attribute
    if hasattr(model, "num_layers"):
        return int(model.num_layers)
    # Fallback: try to parse from model_name like esm2_t33_650M_UR50D
    import re
    m = re.search(r"t(\d+)_", model_name)
    if m:
        return int(m.group(1))
    # conservative default
    return 33


@tool
def esm2_embedder(
    sequences: List[str],
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 8,
    device: str = "auto",
    cache_dir: str = "./esm2_cache",
) -> Dict[str, Any]:
    """Compute and cache ESM-2 embeddings for protein sequences.

    Args:
        sequences: List of amino acid sequences (strings).
        model_name: Name of the ESM2 model as in esm.pretrained (e.g., "esm2_t33_650M_UR50D").
        batch_size: Batch size for embedding computation.
        device: "auto", "cuda", or "cpu".
        cache_dir: Directory to store per-sequence cache (per model_name). Files are named by sha1 of sequence.

    Returns:
        A dict with keys:
            - embeddings: np.ndarray of shape (N, D), dtype float32
            - model_name: str
            - layer: int (final layer index used)
            - seq_hashes: list[str] corresponding to the SHA1 of each input sequence

    Notes:
        - Embeddings are mean-pooled token representations from the final layer, excluding padding/BOS/EOS tokens.
        - Unknown tokens are replaced by 'X'; a warning is issued if replacements occur.
    """
    # Validate inputs
    if not isinstance(sequences, list) or any(not isinstance(s, str) for s in sequences):
        raise TypeError("'sequences' must be a list of strings")
    if len(sequences) == 0:
        return {
            "embeddings": np.zeros((0, 0), dtype=np.float32),
            "model_name": model_name,
            "layer": 0,
            "seq_hashes": [],
        }

    dev = _select_device(device)

    # Prepare cache directory
    model_cache_dir = os.path.join(cache_dir, model_name)
    os.makedirs(model_cache_dir, exist_ok=True)

    # Clean sequences and compute hashes
    cleaned_sequences: List[str] = []
    seq_hashes: List[str] = []
    for idx, s in enumerate(sequences):
        cleaned, repl = _clean_sequence(s)
        if repl > 0:
            warnings.warn(
                f"Sequence at index {idx} contained {repl} invalid token(s); replaced with 'X'."
            )
        cleaned_sequences.append(cleaned)
        seq_hashes.append(hashlib.sha1(cleaned.encode("utf-8")).hexdigest())

    # Determine which sequences are cached
    cache_paths = [os.path.join(model_cache_dir, f"{h}.npy") for h in seq_hashes]
    cached_flags = [os.path.exists(p) for p in cache_paths]

    # Initialize model and alphabet only if needed
    embeddings_list: List[np.ndarray] = [None] * len(sequences)  # type: ignore

    need_indices = [i for i, c in enumerate(cached_flags) if not c]
    if len(need_indices) > 0:
        model, alphabet = _get_model_and_alphabet(model_name)
        model.eval()
        model = model.to(dev)
        batch_converter = alphabet.get_batch_converter()
        final_layer = _infer_final_layer(model, model_name)

        # Process in batches
        with torch.no_grad():
            for start in range(0, len(need_indices), batch_size):
                idxs = need_indices[start : start + batch_size]
                data = [(f"seq{i}", cleaned_sequences[i]) for i in idxs]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(dev)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                out = model(batch_tokens, repr_layers=[final_layer], return_contacts=False)
                reps = out["representations"][final_layer]
                for j, i in enumerate(idxs):
                    L = int(batch_lens[j].item())
                    # Exclude BOS (idx=0) and EOS (idx=L-1)
                    rep = reps[j, 1 : L - 1].mean(0).detach().cpu().numpy().astype(np.float32)
                    embeddings_list[i] = rep
                    # Save to cache
                    np.save(cache_paths[i], rep)
    else:
        # If nothing to compute, we still want to know final layer; load later from model if needed
        final_layer = None  # will infer below if necessary

    # Load cached ones and fill any remaining
    for i, is_cached in enumerate(cached_flags):
        if embeddings_list[i] is None:
            try:
                embeddings_list[i] = np.load(cache_paths[i]).astype(np.float32)
            except Exception as e:
                # If cache load fails, recompute this item ad-hoc
                model, alphabet = _get_model_and_alphabet(model_name)
                model.eval(); model = model.to(dev)
                batch_converter = alphabet.get_batch_converter()
                fl = _infer_final_layer(model, model_name)
                data = [(f"seq{i}", cleaned_sequences[i])]
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(dev)
                lens = (tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    out = model(tokens, repr_layers=[fl], return_contacts=False)
                    reps = out["representations"][fl]
                    L = int(lens[0].item())
                    rep = reps[0, 1 : L - 1].mean(0).detach().cpu().numpy().astype(np.float32)
                embeddings_list[i] = rep
                # refresh cache
                np.save(cache_paths[i], rep)
                if final_layer is None:
                    final_layer = fl

    # Determine final layer if still None
    if "final_layer" not in locals() or final_layer is None:
        # Load a small model instance solely to infer final layer
        try:
            model_tmp, _ = _get_model_and_alphabet(model_name)
            final_layer = _infer_final_layer(model_tmp, model_name)
            del model_tmp
        except Exception:
            final_layer = 33

    # Stack embeddings
    try:
        embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to stack embeddings: {e}")

    return {
        "embeddings": embeddings,
        "model_name": model_name,
        "layer": int(final_layer),
        "seq_hashes": seq_hashes,
    }


if __name__ == "__main__":
    # Quick self-test: embed 2 toy sequences and print shape
    toy_seqs = [
        "MKTFFVAGVILLLATVVAVASSS",
        "GAMGKKKKKKGKKKSSSSSSS",
    ]
    try:
        res = esm2_embedder(toy_seqs, batch_size=2)
        emb = res["embeddings"]
        print(f"[esm2_embedder self-test] embeddings shape: {emb.shape}")
        print(f"[esm2_embedder self-test] model: {res['model_name']}, layer: {res['layer']}")
    except Exception as e:
        print(f"[esm2_embedder self-test] Failed: {e}")

    # Best-effort: register tool with agents if an integration helper exists
    try:
        # Common project-level helper (if provided by the host repo)
        from add_tool_to_agents import add_tool_to_agents  # type: ignore
        add_tool_to_agents(esm2_embedder)
        print("[esm2_embedder] Registered tool via add_tool_to_agents.")
    except Exception as e:
        print(f"[esm2_embedder] Skipping add_tool_to_agents: {e}")