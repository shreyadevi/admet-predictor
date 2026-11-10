"""Evaluation metrics.

Regression: RMSE and MAE in the target's units, R^2, and Spearman rank
correlation (does the model order compounds correctly?).

Classification: ROC-AUC, plus PR-AUC, which matters more here because Tox21
assays are imbalanced (2-16% actives) and ROC-AUC can look good even when the
model finds few true actives. The active rate is reported as the PR-AUC
baseline a random model would get.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Rank correlation is undefined for a constant prediction (e.g. the mean baseline); report 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(y_true, y_pred).statistic
    rho = 0.0 if np.isnan(rho) else rho
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho),
        "n": int(len(y_true)),
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {
        "n": int(len(y_true)),
        "n_active": int(y_true.sum()),
        "active_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        # AUC is undefined when the test set has only one class.
        out.update(roc_auc=float("nan"), pr_auc=float("nan"), balanced_accuracy=float("nan"))
        return out
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, (y_prob >= threshold).astype(int)))
    return out


PRIMARY_METRIC = {"regression": ("rmse", "lower"), "classification": ("roc_auc", "higher")}


def is_better(task: str, candidate: float, incumbent: float | None) -> bool:
    metric, direction = PRIMARY_METRIC[task]
    if incumbent is None or np.isnan(incumbent):
        return not np.isnan(candidate)
    if np.isnan(candidate):
        return False
    return candidate < incumbent if direction == "lower" else candidate > incumbent
