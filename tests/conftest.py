"""Shared fixtures.

`fake_featurizer` / `fake_scaffolder` stand in for RDKit so the training and
evaluation logic can be tested quickly and without chemistry dependencies.
Features are derived deterministically from each SMILES string, and the target
values in the synthetic datasets depend on those features, so models genuinely
have something to learn.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from admet.data import TOX21_ASSAYS

N_BITS = 64
N_DESC = 8


def _vector(smiles: str) -> np.random.Generator:
    seed = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def fake_featurizer(smiles: list[str], feature_set: str):
    valid = np.array([not s.startswith("INVALID") for s in smiles])
    rows = []
    for s in (s for s, ok in zip(smiles, valid) if ok):
        rng = _vector(s)
        bits = (rng.random(N_BITS) < 0.2).astype(np.float32)   # always drawn first, so descriptors
        desc = rng.normal(size=N_DESC).astype(np.float32)      # are identical whichever set is asked for
        parts = []
        if "morgan" in feature_set:
            parts.append(bits)
        if "rdkit2d" in feature_set:
            parts.append(desc)
        rows.append(np.concatenate(parts))
    return (np.vstack(rows) if rows else np.zeros((0, 0), np.float32)), valid


def fake_scaffolder(smiles: list[str]) -> list[str]:
    # Molecules are named "S<group>_<i>", so the group acts as the scaffold.
    return [s.split("_")[0] for s in smiles]


def _signal(smiles: str) -> float:
    rng = _vector(smiles)
    rng.random(N_BITS)                  # skip the fingerprint draw
    return float(rng.normal(size=N_DESC)[:3].sum())


@pytest.fixture
def synthetic_data(tmp_path):
    """Tiny ESOL- and Tox21-shaped datasets written where load_endpoint expects them."""
    rng = np.random.default_rng(0)
    smiles = [f"S{g}_{i}" for g in range(30) for i in range(rng.integers(2, 9))]
    smiles += ["INVALID_1", "INVALID_2"]
    signal = np.array([_signal(s) for s in smiles])

    esol = pd.DataFrame({"compound_id": range(len(smiles)), "smiles": smiles,
                         "logS": -3 + signal + rng.normal(scale=0.3, size=len(smiles))})
    esol.to_csv(tmp_path / "esol.csv", index=False)

    tox = pd.DataFrame({"mol_id": range(len(smiles)), "smiles": smiles})
    for k, assay in enumerate(TOX21_ASSAYS):
        y = (signal + rng.normal(scale=0.5, size=len(smiles)) > 1.0 + 0.05 * k).astype(float)
        y[rng.random(len(smiles)) < 0.1] = np.nan  # missing labels, as in real Tox21
        tox[assay] = y
    tox.to_csv(tmp_path / "tox21.csv.gz", index=False, compression="gzip")
    return tmp_path
