"""Applicability domain: is this molecule like anything the model was trained on?

A model is only trustworthy near its training data. For each query molecule we
compute the highest Tanimoto similarity (on Morgan fingerprints) to any
training molecule. Below roughly 0.3 the molecule is outside the model's
experience and the prediction should be treated as a rough guess.

Implemented with NumPy on packed bit arrays so it needs no RDKit at runtime.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLD = 0.3


def pack(fingerprints: np.ndarray) -> np.ndarray:
    """Pack 0/1 fingerprints (n x bits) into bytes for compact storage (8x smaller)."""
    return np.packbits(np.asarray(fingerprints, dtype=np.uint8), axis=1)


def unpack(packed: np.ndarray, n_bits: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1, count=n_bits)


def max_tanimoto(query: np.ndarray, reference: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """For each query fingerprint, the highest Tanimoto similarity to any reference fingerprint.

    Tanimoto(A, B) = |A AND B| / |A OR B| = c / (a + b - c).
    """
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("query and reference must be 2-D with the same number of bits")
    r_counts = r.sum(axis=1)
    best = np.zeros(len(q), dtype=np.float32)
    for start in range(0, len(q), batch_size):
        block = q[start:start + batch_size]
        common = block @ r.T
        union = block.sum(axis=1, keepdims=True) + r_counts[None, :] - common
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, common / union, 0.0)
        best[start:start + batch_size] = sim.max(axis=1) if sim.size else 0.0
    return best


def in_domain(similarity: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    return np.asarray(similarity) >= threshold
