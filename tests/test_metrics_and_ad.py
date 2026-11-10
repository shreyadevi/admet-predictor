import math

import numpy as np
import pytest

from admet import applicability
from admet.metrics import classification_metrics, is_better, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = [-1.0, -2.0, -3.5, 0.2]
    m = regression_metrics(y, y)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)


def test_regression_metrics_known_rmse():
    m = regression_metrics([0, 0, 0, 0], [1, -1, 1, -1])
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)


def test_classification_metrics_perfect_ranking():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["active_rate"] == pytest.approx(0.5)


def test_classification_metrics_single_class_is_nan_not_error():
    m = classification_metrics([0, 0, 0], [0.1, 0.5, 0.9])
    assert math.isnan(m["roc_auc"])
    assert m["n_active"] == 0


def test_is_better_respects_metric_direction():
    assert is_better("regression", 0.5, 0.7)          # lower RMSE wins
    assert not is_better("regression", 0.9, 0.7)
    assert is_better("classification", 0.8, 0.7)      # higher AUC wins
    assert is_better("classification", 0.6, None)
    assert not is_better("classification", float("nan"), 0.6)


def test_tanimoto_identical_and_disjoint():
    ref = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    query = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    sim = applicability.max_tanimoto(query, ref)
    assert sim[0] == pytest.approx(1.0)
    assert sim[1] == pytest.approx(0.0)   # empty fingerprint: no similarity, no division error


def test_tanimoto_known_value_picks_nearest():
    ref = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=np.uint8)
    query = np.array([[1, 1, 0, 1]], dtype=np.uint8)
    # vs ref0: common 2, union 4 -> 0.5 ; vs ref1: common 1, union 3 -> 0.333
    assert applicability.max_tanimoto(query, ref)[0] == pytest.approx(0.5)


def test_tanimoto_batches_match_unbatched():
    rng = np.random.default_rng(0)
    ref = (rng.random((300, 128)) < 0.1).astype(np.uint8)
    query = (rng.random((70, 128)) < 0.1).astype(np.uint8)
    full = applicability.max_tanimoto(query, ref, batch_size=1000)
    batched = applicability.max_tanimoto(query, ref, batch_size=16)
    assert np.allclose(full, batched)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    bits = (rng.random((5, 2048)) < 0.05).astype(np.uint8)
    packed = applicability.pack(bits)
    assert packed.nbytes == bits.nbytes // 8
    assert np.array_equal(applicability.unpack(packed, 2048), bits)


def test_in_domain_threshold():
    assert list(applicability.in_domain(np.array([0.1, 0.3, 0.9]))) == [False, True, True]
