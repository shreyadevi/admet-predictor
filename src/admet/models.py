"""Model configurations compared in the benchmark.

Each configuration pairs a feature set with an estimator. A naive baseline
(predict the training mean / active rate) is always included, so every score
can be read as "how much better than knowing nothing".
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    name: str
    estimator: str
    feature_set: str

    @property
    def label(self) -> str:
        return f"{self.estimator} | {self.feature_set}"


CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig("baseline", "baseline", "rdkit2d"),
    ModelConfig("linear_desc", "linear", "rdkit2d"),
    ModelConfig("rf_morgan", "random_forest", "morgan"),
    ModelConfig("rf_desc", "random_forest", "rdkit2d"),
    ModelConfig("gbm_desc", "gradient_boosting", "rdkit2d"),
    ModelConfig("gbm_combined", "gradient_boosting", "morgan+rdkit2d"),
)

CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}


def build_estimator(config: ModelConfig, task: str, seed: int = 0, fast: bool = False):
    """Create an unfitted scikit-learn estimator for a configuration."""
    n_trees = 100 if fast else 500
    # Tox21 trains 12 classifiers on ~7k molecules each; fewer, shallower trees
    # keep the saved model a manageable size with little loss in accuracy.
    n_class_trees = 100 if fast else 200
    n_iter = 150 if fast else 500
    regression = task == "regression"

    if config.estimator == "baseline":
        return DummyRegressor(strategy="mean") if regression else DummyClassifier(strategy="prior")

    if config.estimator == "linear":
        if regression:
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000),
        )

    if config.estimator == "random_forest":
        if regression:
            return RandomForestRegressor(n_estimators=n_trees, max_features=0.3, n_jobs=-1, random_state=seed)
        return RandomForestClassifier(
            n_estimators=n_class_trees,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    if config.estimator == "gradient_boosting":
        if regression:
            return HistGradientBoostingRegressor(
                max_iter=n_iter, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
            )
        return HistGradientBoostingClassifier(
            max_iter=n_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )

    raise ValueError(f"Unknown estimator {config.estimator!r}")


def predict_output(model, X, task: str):
    """Point predictions for regression, probability of 'active' for classification."""
    if task == "regression":
        return model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, 0] * 0.0  # training data had no actives
