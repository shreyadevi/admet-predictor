import numpy as np
import pytest

from admet.splits import random_split, scaffold_split_from_scaffolds


def _check_partition(train, valid, test, n):
    combined = np.concatenate([train, valid, test])
    assert len(combined) == n
    assert len(np.unique(combined)) == n, "partitions overlap"


def test_random_split_sizes_and_disjoint():
    train, valid, test = random_split(1000, seed=1)
    _check_partition(train, valid, test, 1000)
    assert (len(train), len(valid), len(test)) == (800, 100, 100)


def test_random_split_is_reproducible_and_seed_dependent():
    a = random_split(200, seed=3)
    b = random_split(200, seed=3)
    c = random_split(200, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])


def test_scaffold_split_never_shares_a_scaffold_across_partitions():
    rng = np.random.default_rng(0)
    scaffolds = [f"scaf{rng.integers(0, 60)}" for _ in range(500)]
    train, valid, test = scaffold_split_from_scaffolds(scaffolds)
    _check_partition(train, valid, test, 500)
    groups = [{scaffolds[i] for i in part} for part in (train, valid, test)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_scaffold_split_puts_largest_series_in_training():
    scaffolds = ["big"] * 50 + ["mid"] * 20 + [f"rare{i}" for i in range(30)]
    train, valid, test = scaffold_split_from_scaffolds(scaffolds)
    train_scaffolds = {scaffolds[i] for i in train}
    assert {"big", "mid"} <= train_scaffolds
    assert all(scaffolds[i].startswith("rare") for i in test)


def test_scaffold_split_is_deterministic():
    scaffolds = [f"s{i % 17}" for i in range(300)]
    a = scaffold_split_from_scaffolds(scaffolds)
    b = scaffold_split_from_scaffolds(scaffolds)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("fractions", [(0.5, 0.5, 0.5), (0.8, 0.2), (-0.1, 0.6, 0.5)])
def test_invalid_fractions_rejected(fractions):
    with pytest.raises(ValueError):
        random_split(10, fractions=fractions)
