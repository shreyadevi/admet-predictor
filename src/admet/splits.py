"""Train / validation / test splits.

A **random split** puts near-identical molecules from the same chemical series
on both sides, so the test set mostly asks "can you recognise a close
relative?". That inflates scores.

A **scaffold split** keeps every molecule sharing a Bemis-Murcko scaffold in
the same partition, so the test set contains chemical series the model has
never seen. That is closer to the real use case - predicting properties of new
designs - and gives a more honest estimate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def _check_fractions(fractions: tuple[float, float, float]) -> None:
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1, got {fractions}")


def random_split(n: int, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> Split:
    _check_fractions(fractions)
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_valid = int(round(fractions[1] * n))
    return (np.sort(order[:n_train]),
            np.sort(order[n_train:n_train + n_valid]),
            np.sort(order[n_train + n_valid:]))


def scaffold_split_from_scaffolds(scaffolds: list[str], fractions=(0.8, 0.1, 0.1)) -> Split:
    """Deterministic scaffold split (the MoleculeNet / DeepChem procedure).

    Scaffold groups are sorted largest first (ties broken by first appearance)
    and assigned to train until it is full, then validation, then test. Large,
    common series therefore land in training, and the test set is made of the
    rarer scaffolds - the hard, realistic case.
    """
    _check_fractions(fractions)
    n = len(scaffolds)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, scaffold in enumerate(scaffolds):
        groups[scaffold].append(i)
    ordered = sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))

    train_cutoff = fractions[0] * n
    valid_cutoff = (fractions[0] + fractions[1]) * n
    train, valid, test = [], [], []
    for idx in ordered:
        if len(train) + len(idx) <= train_cutoff:
            train.extend(idx)
        elif len(train) + len(valid) + len(idx) <= valid_cutoff:
            valid.extend(idx)
        else:
            test.extend(idx)
    return np.array(sorted(train), dtype=int), np.array(sorted(valid), dtype=int), np.array(sorted(test), dtype=int)


def scaffold_split(smiles: list[str], fractions=(0.8, 0.1, 0.1)) -> Split:
    from .featurize import murcko_scaffolds

    return scaffold_split_from_scaffolds(murcko_scaffolds(smiles), fractions)


def make_split(kind: str, smiles: list[str], seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> Split:
    if kind == "random":
        return random_split(len(smiles), fractions, seed)
    if kind == "scaffold":
        return scaffold_split(smiles, fractions)
    raise ValueError(f"Unknown split {kind!r}; use 'random' or 'scaffold'")
