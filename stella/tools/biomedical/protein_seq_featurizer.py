# -*- coding: utf-8 -*-
"""
protein_seq_featurizer: Reusable protein sequence featurizer tool.

Features:
- Position-wise one-hot encoding (21 channels: 20 standard AAs + 'X' unknown)
- Biochemical deltas vs WT: mutation_count, BLOSUM62 sum/mean, hydrophobicity (Kyte-Doolittle) delta sum/mean,
  charge delta sum, charge flip count
- Optional pairwise interaction features within a sliding window, hashed into a fixed-size vector (default 4096)

Returns:
- Tuple[np.ndarray (float32), Dict[str, Any] metadata]

This module exposes a function decorated with @tool from smolagents for usage by agents.
If smolagents is not installed in the runtime, a no-op fallback decorator is used so the
module remains importable and testable. In agent environments, the true decorator will
be applied as expected.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import hashlib
import numpy as np

# Optional import for @tool; provide a robust fallback if unavailable
try:
    from smolagents import tool  # type: ignore
except Exception:  # pragma: no cover
    def tool(func=None, **kwargs):  # type: ignore
        """Fallback no-op decorator if smolagents is not available."""
        if func is None:
            def wrapper(f):
                return f
            return wrapper
        return func

# Constants and lookup tables
STANDARD_AA: str = "ACDEFGHIKLMNPQRSTVWY"  # order matters
UNKNOWN_AA: str = "X"
AA_LIST: List[str] = list(STANDARD_AA) + [UNKNOWN_AA]
AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_LIST)}
ONE_HOT_DIM: int = len(AA_LIST)  # 21
PAIRWISE_DIM: int = 4096  # fixed-size hashed vector length

# Kyte-Doolittle hydrophobicity scale
KD_HYDRO: Dict[str, float] = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Simplified net charge at pH 7: H is partially protonated (approx +0.1)
AA_CHARGE_PH7: Dict[str, float] = {
    'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.1,
    'A': 0.0, 'C': 0.0, 'F': 0.0, 'G': 0.0, 'I': 0.0, 'L': 0.0,
    'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'S': 0.0, 'T': 0.0,
    'V': 0.0, 'W': 0.0, 'Y': 0.0
}

# BLOSUM62 matrix as dict of dicts for the 20 standard amino acids
# Source: Standard BLOSUM62 (rounded integers)
BLOSUM62: Dict[str, Dict[str, int]] = {
    'A': {'A': 4,  'C': 0,  'D': -2, 'E': -1, 'F': -2, 'G': 0,  'H': -2, 'I': -1, 'K': -1, 'L': -1, 'M': -1, 'N': -2, 'P': -1, 'Q': -1, 'R': -1, 'S': 1,  'T': 0,  'V': 0,  'W': -3, 'Y': -2},
    'C': {'A': 0,  'C': 9,  'D': -3, 'E': -4, 'F': -2, 'G': -3, 'H': -3, 'I': -1, 'K': -3, 'L': -1, 'M': -1, 'N': -3, 'P': -3, 'Q': -3, 'R': -3, 'S': -1, 'T': -1, 'V': -1, 'W': -2, 'Y': -2},
    'D': {'A': -2, 'C': -3, 'D': 6,  'E': 2,  'F': -3, 'G': -1, 'H': -1, 'I': -3, 'K': -1, 'L': -4, 'M': -3, 'N': 1,  'P': -1, 'Q': 0,  'R': -2, 'S': 0,  'T': -1, 'V': -3, 'W': -4, 'Y': -3},
    'E': {'A': -1, 'C': -4, 'D': 2,  'E': 5,  'F': -3, 'G': -2, 'H': 0,  'I': -3, 'K': 1,  'L': -3, 'M': -2, 'N': 0,  'P': -1, 'Q': 2,  'R': 0,  'S': 0,  'T': -1, 'V': -2, 'W': -3, 'Y': -2},
    'F': {'A': -2, 'C': -2, 'D': -3, 'E': -3, 'F': 6,  'G': -3, 'H': -1, 'I': 0,  'K': -3, 'L': 0,  'M': 0,  'N': -3, 'P': -4, 'Q': -3, 'R': -3, 'S': -2, 'T': -2, 'V': -1, 'W': 1,  'Y': 3},
    'G': {'A': 0,  'C': -3, 'D': -1, 'E': -2, 'F': -3, 'G': 6,  'H': -2, 'I': -4, 'K': -2, 'L': -4, 'M': -3, 'N': 0,  'P': -2, 'Q': -2, 'R': -2, 'S': 0,  'T': -2, 'V': -3, 'W': -2, 'Y': -3},
    'H': {'A': -2, 'C': -3, 'D': -1, 'E': 0,  'F': -1, 'G': -2, 'H': 8,  'I': -3, 'K': -1, 'L': -3, 'M': -2, 'N': 1,  'P': -2, 'Q': 0,  'R': 0,  'S': -1, 'T': -2, 'V': -3, 'W': -2, 'Y': 2},
    'I': {'A': -1, 'C': -1, 'D': -3, 'E': -3, 'F': 0,  'G': -4, 'H': -3, 'I': 4,  'K': -3, 'L': 2,  'M': 1,  'N': -3, 'P': -3, 'Q': -3, 'R': -3, 'S': -2, 'T': -1, 'V': 3,  'W': -3, 'Y': -1},
    'K': {'A': -1, 'C': -3, 'D': -1, 'E': 1,  'F': -3, 'G': -2, 'H': -1, 'I': -3, 'K': 5,  'L': -2, 'M': -1, 'N': 0,  'P': -1, 'Q': 1,  'R': 2,  'S': 0,  'T': -1, 'V': -2, 'W': -3, 'Y': -2},
    'L': {'A': -1, 'C': -1, 'D': -4, 'E': -3, 'F': 0,  'G': -4, 'H': -3, 'I': 2,  'K': -2, 'L': 4,  'M': 2,  'N': -3, 'P': -3, 'Q': -2, 'R': -2, 'S': -2, 'T': -1, 'V': 1,  'W': -2, 'Y': -1},
    'M': {'A': -1, 'C': -1, 'D': -3, 'E': -2, 'F': 0,  'G': -3, 'H': -2, 'I': 1,  'K': -1, 'L': 2,  'M': 5,  'N': -2, 'P': -2, 'Q': 0,  'R': -1, 'S': -1, 'T': -1, 'V': 1,  'W': -1, 'Y': -1},
    'N': {'A': -2, 'C': -3, 'D': 1,  'E': 0,  'F': -3, 'G': 0,  'H': 1,  'I': -3, 'K': 0,  'L': -3, 'M': -2, 'N': 6,  'P': -2, 'Q': 0,  'R': 0,  'S': 1,  'T': 0,  'V': -3, 'W': -4, 'Y': -2},
    'P': {'A': -1, 'C': -3, 'D': -1, 'E': -1, 'F': -4, 'G': -2, 'H': -2, 'I': -3, 'K': -1, 'L': -3, 'M': -2, 'N': -2, 'P': 7,  'Q': -1, 'R': -2, 'S': -1, 'T': -1, 'V': -2, 'W': -4, 'Y': -3},
    'Q': {'A': -1, 'C': -3, 'D': 0,  'E': 2,  'F': -3, 'G': -2, 'H': 0,  'I': -3, 'K': 1,  'L': -2, 'M': 0,  'N': 0,  'P': -1, 'Q': 5,  'R': 1,  'S': 0,  'T': -1, 'V': -2, 'W': -2, 'Y': -1},
    'R': {'A': -1, 'C': -3, 'D': -2, 'E': 0,  'F': -3, 'G': -2, 'H': 0,  'I': -3, 'K': 2,  'L': -2, 'M': -1, 'N': 0,  'P': -2, 'Q': 1,  'R': 5,  'S': -1, 'T': -1, 'V': -3, 'W': -3, 'Y': -2},
    'S': {'A': 1,  'C': -1, 'D': 0,  'E': 0,  'F': -2, 'G': 0,  'H': -1, 'I': -2, 'K': 0,  'L': -2, 'M': -1, 'N': 1,  'P': -1, 'Q': 0,  'R': -1, 'S': 4,  'T': 1,  'V': -2, 'W': -3, 'Y': -2},
    'T': {'A': 0,  'C': -1, 'D': -1, 'E': -1, 'F': -2, 'G': -2, 'H': -2, 'I': -1, 'K': -1, 'L': -1, 'M': -1, 'N': 0,  'P': -1, 'Q': -1, 'R': -1, 'S': 1,  'T': 5,  'V': 0,  'W': -2, 'Y': -2},
    'V': {'A': 0,  'C': -1, 'D': -3, 'E': -2, 'F': -1, 'G': -3, 'H': -3, 'I': 3,  'K': -2, 'L': 1,  'M': 1,  'N': -3, 'P': -2, 'Q': -2, 'R': -3, 'S': -2, 'T': 0,  'V': 4,  'W': -3, 'Y': -1},
    'W': {'A': -3, 'C': -2, 'D': -4, 'E': -3, 'F': 1,  'G': -2, 'H': -2, 'I': -3, 'K': -3, 'L': -2, 'M': -1, 'N': -4, 'P': -4, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V': -3, 'W': 11, 'Y': 2},
    'Y': {'A': -2, 'C': -2, 'D': -3, 'E': -2, 'F': 3,  'G': -3, 'H': 2,  'I': -1, 'K': -2, 'L': -1, 'M': -1, 'N': -2, 'P': -3, 'Q': -1, 'R': -2, 'S': -2, 'T': -2, 'V': -1, 'W': 2,  'Y': 7},
}

BIO_DELTA_NAMES: List[str] = [
    'mutation_count',
    'blosum62_sum',
    'blosum62_mean',
    'hydrophobicity_delta_sum',
    'hydrophobicity_delta_mean',
    'charge_delta_sum',
    'charge_flip_count',
]


def _safe_aa(aa: str) -> str:
    """Return uppercase amino acid, 'X' if non-standard."""
    if not aa:
        return UNKNOWN_AA
    c = aa.upper()
    return c if c in AA_INDEX else UNKNOWN_AA


def _one_hot(seq: str) -> np.ndarray:
    """One-hot encode a protein sequence into shape (L, ONE_HOT_DIM) float32."""
    L = len(seq)
    arr = np.zeros((L, ONE_HOT_DIM), dtype=np.float32)
    for i, ch in enumerate(seq):
        aa = _safe_aa(ch)
        idx = AA_INDEX[aa]
        arr[i, idx] = 1.0
    return arr


def _bio_deltas(seq: str, wt: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute biochemical deltas vs WT.

    Returns
    - features: np.ndarray of length len(BIO_DELTA_NAMES)
    - info: Dict with intermediate counts for metadata
    """
    assert len(seq) == len(wt), "Sequence length must equal WT length"
    mut_count = 0
    bl_sum = 0.0
    bl_n = 0
    hyd_sum = 0.0
    hyd_n = 0
    charge_delta_sum = 0.0
    charge_flip_count = 0

    for a, b in zip(wt, seq):
        aa_w = _safe_aa(a)
        aa_s = _safe_aa(b)
        if aa_w == aa_s:
            continue
        mut_count += 1
        # BLOSUM62
        if aa_w in BLOSUM62 and aa_s in BLOSUM62[aa_w]:
            bl_sum += float(BLOSUM62[aa_w][aa_s])
            bl_n += 1
        # Hydrophobicity delta
        if aa_w in KD_HYDRO and aa_s in KD_HYDRO:
            hyd_sum += float(KD_HYDRO[aa_s] - KD_HYDRO[aa_w])
            hyd_n += 1
        # Charge delta
        q_w = AA_CHARGE_PH7.get(aa_w, 0.0)
        q_s = AA_CHARGE_PH7.get(aa_s, 0.0)
        charge_delta_sum += (q_s - q_w)
        # Charge flip count only if sign flips between positive and negative (ignore zero)
        if (q_w > 0 and q_s < 0) or (q_w < 0 and q_s > 0):
            charge_flip_count += 1

    bl_mean = (bl_sum / bl_n) if bl_n > 0 else 0.0
    hyd_mean = (hyd_sum / hyd_n) if hyd_n > 0 else 0.0

    feats = np.array([
        float(mut_count),
        float(bl_sum),
        float(bl_mean),
        float(hyd_sum),
        float(hyd_mean),
        float(charge_delta_sum),
        float(charge_flip_count),
    ], dtype=np.float32)

    info = {
        'mutated_positions': int(mut_count),
        'blosum_pairs_used': int(bl_n),
        'hydrop_pairs_used': int(hyd_n),
    }
    return feats, info


def _hash_pair_feature(i: int, j: int, aa_i: str, aa_j: str, dim: int = PAIRWISE_DIM) -> int:
    """Stable hash to an index in [0, dim).

    Uses blake2b for speed and stability across runs.
    """
    key = f"{i}-{j}-{aa_i}{aa_j}".encode('utf-8')
    h = hashlib.blake2b(key, digest_size=8).digest()
    # Convert first 8 bytes to int
    idx = int.from_bytes(h, byteorder='little', signed=False) % dim
    return idx


def _pairwise_vector(seq: str, window: int, dim: int = PAIRWISE_DIM) -> np.ndarray:
    """Compute hashed pairwise interaction vector within +/- window.

    For each pair (i, j) with 0 < (j - i) <= window, encode the tuple (i, j, aa_i, aa_j)
    into a stable hashed index and increment that position by 1.0.
    """
    L = len(seq)
    vec = np.zeros((dim,), dtype=np.float32)
    for i in range(L):
        aa_i = _safe_aa(seq[i])
        jmax = min(L, i + window + 1)
        for j in range(i + 1, jmax):
            aa_j = _safe_aa(seq[j])
            idx = _hash_pair_feature(i, j, aa_i, aa_j, dim)
            vec[idx] += 1.0
    return vec


@tool
def protein_seq_featurizer(
    sequences: List[str],
    wt_sequence: str,
    one_hot: bool = True,
    bio_deltas: bool = True,
    pairwise: bool = False,
    pairwise_window: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Feature-engineer protein sequences against a wild-type (WT) reference.

    This tool generates a fixed-length float32 feature vector per input sequence by
    concatenating selected components:
    - One-hot encoding (21 channels: 20 standard amino acids + 'X' for non-standard)
    - Biochemical deltas vs WT (mutation_count, BLOSUM62 sum/mean, hydrophobicity delta sum/mean,
      charge delta sum, charge flip count)
    - Hashed pairwise interaction features within a positional window

    Args:
        sequences: List of protein sequences (strings) to featurize. All must be same length as wt_sequence.
        wt_sequence: The wild-type (reference) sequence.
        one_hot: Include position-wise one-hot encoding of each sequence (default: True).
        bio_deltas: Include biochemical delta summary features vs WT (default: True).
        pairwise: Include hashed pairwise interaction features (default: False).
        pairwise_window: Positional window radius for pairwise interactions (default: 5).

    Returns:
        A tuple containing:
        - features: np.ndarray of shape (N, F) with dtype float32, where N = len(sequences), F depends on selected options.
        - metadata: Dict[str, Any] with details:
            - length: sequence length (L)
            - options: dict of feature flags
            - one_hot_dim: int (21) if enabled
            - pairwise_dim: int (4096) if enabled
            - bio_delta_names: ordered list of biochemical delta feature names
            - aa_index: mapping of amino acid symbol to one-hot index
            - feature_slices: dict with tuple (start, end) slices for each component within the feature vector
            - per_sequence_info: list of dicts with per-sequence computation info (e.g., counts used)
            - warnings: any collected warnings

    Raises:
        ValueError: If inputs are invalid (empty list, mismatched lengths, negative window).
    """
    # Input validation
    if sequences is None or not isinstance(sequences, list) or len(sequences) == 0:
        raise ValueError("'sequences' must be a non-empty list of strings.")
    if not isinstance(wt_sequence, str) or len(wt_sequence) == 0:
        raise ValueError("'wt_sequence' must be a non-empty string.")
    if not all(isinstance(s, str) for s in sequences):
        raise ValueError("All items in 'sequences' must be strings.")
    if any(len(s) != len(wt_sequence) for s in sequences):
        raise ValueError("All sequences must have the same length as 'wt_sequence'.")
    if not isinstance(pairwise_window, int) or pairwise_window < 0:
        raise ValueError("'pairwise_window' must be a non-negative integer.")

    L = len(wt_sequence)

    # Determine lengths of each component
    one_hot_len = L * ONE_HOT_DIM if one_hot else 0
    bio_len = len(BIO_DELTA_NAMES) if bio_deltas else 0
    pair_len = PAIRWISE_DIM if pairwise else 0

    total_len = one_hot_len + bio_len + pair_len
    if total_len == 0:
        raise ValueError("At least one of 'one_hot', 'bio_deltas', or 'pairwise' must be True.")

    N = len(sequences)
    features = np.zeros((N, total_len), dtype=np.float32)

    # Slices for metadata
    offset = 0
    slices: Dict[str, Tuple[int, int]] = {}
    if one_hot:
        slices['one_hot'] = (offset, offset + one_hot_len)
        offset += one_hot_len
    if bio_deltas:
        slices['bio_deltas'] = (offset, offset + bio_len)
        offset += bio_len
    if pairwise:
        slices['pairwise'] = (offset, offset + pair_len)
        offset += pair_len

    warnings: List[str] = []
    per_seq_info: List[Dict[str, Any]] = []

    wt_sequence = wt_sequence.upper()

    # Process each sequence
    for idx, seq in enumerate(sequences):
        s = seq.upper()
        # One-hot
        if one_hot:
            oh = _one_hot(s).reshape(-1).astype(np.float32)
            start, end = slices['one_hot']
            features[idx, start:end] = oh
        # Biochemical deltas
        if bio_deltas:
            bio_vec, info = _bio_deltas(s, wt_sequence)
            start, end = slices['bio_deltas']
            features[idx, start:end] = bio_vec
            per_seq_info.append(info)
        else:
            per_seq_info.append({})
        # Pairwise hashed vector
        if pairwise:
            pw = _pairwise_vector(s, window=pairwise_window, dim=PAIRWISE_DIM)
            start, end = slices['pairwise']
            features[idx, start:end] = pw

        # Collect warnings for non-standard AAs
        if any(_safe_aa(ch) == UNKNOWN_AA for ch in s):
            warnings.append(f"Sequence index {idx} contains non-standard amino acids; treated as 'X'.")

    metadata: Dict[str, Any] = {
        'length': L,
        'options': {
            'one_hot': one_hot,
            'bio_deltas': bio_deltas,
            'pairwise': pairwise,
            'pairwise_window': pairwise_window,
        },
        'one_hot_dim': ONE_HOT_DIM if one_hot else 0,
        'pairwise_dim': PAIRWISE_DIM if pairwise else 0,
        'bio_delta_names': BIO_DELTA_NAMES if bio_deltas else [],
        'aa_index': AA_INDEX,
        'feature_slices': slices,
        'per_sequence_info': per_seq_info,
        'warnings': warnings,
        'standard_aa': STANDARD_AA,
        'unknown_aa': UNKNOWN_AA,
        'dtype': 'float32',
        'total_features': total_len,
        'num_sequences': N,
    }

    return features.astype(np.float32, copy=False), metadata


if __name__ == "__main__":
    # Basic self-test to validate core functionality
    test_wt = "ACDEFGHIKLMNPQRSTVWY"  # length 20, all standard AAs
    test_sequences = [
        test_wt,
        "ACDEYGHIKLMNPQRSTVWY",  # F->Y mutation at pos 5
        "XCDEFGHIKLMNPQRSTVWX",  # Non-standard at ends
        "ACDEFGHIKLMNPQKSTVWY",  # R->K mutation
    ]
    feats, meta = protein_seq_featurizer(
        sequences=test_sequences,
        wt_sequence=test_wt,
        one_hot=True,
        bio_deltas=True,
        pairwise=True,
        pairwise_window=3,
    )
    print("[Self-test] features shape:", feats.shape)
    print("[Self-test] dtype:", feats.dtype)
    print("[Self-test] feature_slices:", meta.get('feature_slices'))
    print("[Self-test] bio_delta_names:", meta.get('bio_delta_names'))
    print("[Self-test] warnings (if any):", meta.get('warnings'))