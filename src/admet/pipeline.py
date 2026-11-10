"""Benchmarking and training.

* `benchmark` compares every model configuration on every endpoint, under both
  random and scaffold splits, over several random seeds.
* `train_final` picks the best configuration on the validation set, reports its
  held-out test performance, then refits on all data and saves the model.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import applicability
from .data import get_endpoint, load_endpoint
from .metrics import PRIMARY_METRIC, classification_metrics, is_better, regression_metrics
from .models import CONFIGS, ModelConfig, build_estimator, predict_output
from .splits import make_split, random_split, scaffold_split_from_scaffolds

Featurizer = Callable[[list[str], str], tuple[np.ndarray, np.ndarray]]
Scaffolder = Callable[[list[str]], list[str]]


class FeatureStore:
    """Computes each feature set once per dataset and reuses it across models and seeds."""

    def __init__(self, smiles: list[str], featurizer: Featurizer | None = None, scaffolder: Scaffolder | None = None):
        if featurizer is None or scaffolder is None:
            from . import featurize as fz

            featurizer = featurizer or fz.featurize
            scaffolder = scaffolder or fz.murcko_scaffolds
        self._featurizer = featurizer
        self._scaffolder = scaffolder
        self._cache: dict[str, np.ndarray] = {}

        # Validity is decided once, from the fingerprint pass.
        X, valid = featurizer(smiles, "morgan")
        self.valid = valid
        self.smiles = [s for s, ok in zip(smiles, valid) if ok]
        self._cache["morgan"] = X

    def get(self, feature_set: str) -> np.ndarray:
        if feature_set not in self._cache:
            parts = feature_set.split("+")
            if len(parts) > 1:
                self._cache[feature_set] = np.hstack([self.get(p) for p in parts])
            else:
                X, valid = self._featurizer(self.smiles, feature_set)
                if not valid.all():
                    raise RuntimeError("Molecules valid for fingerprints failed descriptor featurisation")
                self._cache[feature_set] = X
        return self._cache[feature_set]

    def scaffolds(self) -> list[str]:
        if "_scaffolds" not in self._cache:
            self._cache["_scaffolds"] = np.array(self._scaffolder(self.smiles), dtype=object)
        return list(self._cache["_scaffolds"])

    def split(self, kind: str, seed: int):
        if kind == "random":
            return random_split(len(self.smiles), seed=seed)
        if kind == "scaffold":
            return scaffold_split_from_scaffolds(self.scaffolds())
        return make_split(kind, self.smiles, seed)


def _fit_predict(config: ModelConfig, task: str, X, y, train_idx, eval_sets: dict, seed: int, fast: bool):
    """Fit on labelled training rows; return predictions for each evaluation subset."""
    labelled = ~np.isnan(y)
    tr = train_idx[labelled[train_idx]]
    model = build_estimator(config, task, seed=seed, fast=fast)
    y_train = y[tr] if task == "regression" else y[tr].astype(int)
    model.fit(X[tr], y_train)
    out = {}
    for name, idx in eval_sets.items():
        idx = idx[labelled[idx]]
        out[name] = (y[idx], predict_output(model, X[idx], task)) if len(idx) else (np.array([]), np.array([]))
    return model, out


def _score(task: str, y_true, y_pred) -> dict:
    if len(y_true) == 0:
        return {}
    return regression_metrics(y_true, y_pred) if task == "regression" else classification_metrics(y_true, y_pred)


def _run_store(endpoint, store: FeatureStore, labels, splits, seeds, configs, fast, log) -> list[dict]:
    rows = []
    for split_kind in splits:
        for seed in seeds:
            train_idx, valid_idx, test_idx = store.split(split_kind, seed)
            for config in configs:
                X = store.get(config.feature_set)
                for t, target in enumerate(endpoint.targets):
                    _, preds = _fit_predict(config, endpoint.task, X, labels[:, t], train_idx,
                                            {"valid": valid_idx, "test": test_idx}, seed, fast)
                    for subset, (y_true, y_pred) in preds.items():
                        rows.append({"endpoint": endpoint.name, "split": split_kind, "seed": seed,
                                     "config": config.name, "model": config.label, "target": target,
                                     "subset": subset, **_score(endpoint.task, y_true, y_pred)})
                log(f"[{endpoint.name}] split={split_kind} seed={seed} {config.label}: done")
    return rows


def _prepare(endpoint_name, featurizer, scaffolder, data_dir):
    endpoint = get_endpoint(endpoint_name)
    df = load_endpoint(endpoint_name, data_dir)
    store = FeatureStore(df["smiles"].tolist(), featurizer, scaffolder)
    labels = df.loc[store.valid, list(endpoint.targets)].to_numpy(dtype=float)
    return endpoint, df, store, labels


def benchmark(
    endpoints=("solubility", "tox21"),
    splits=("random", "scaffold"),
    seeds=(0, 1, 2),
    configs: tuple[ModelConfig, ...] = CONFIGS,
    fast: bool = False,
    featurizer: Featurizer | None = None,
    scaffolder: Scaffolder | None = None,
    data_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Return one row per (endpoint, split, seed, config, target, subset) with its metrics."""
    rows = []
    for endpoint_name in endpoints:
        endpoint, df, store, labels = _prepare(endpoint_name, featurizer, scaffolder, data_dir)
        log(f"[{endpoint_name}] {store.valid.sum()} valid molecules of {len(df)}")
        rows.extend(_run_store(endpoint, store, labels, splits, seeds, configs, fast, log))
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, subset: str = "test") -> pd.DataFrame:
    """Mean and standard deviation over seeds of each endpoint's headline metrics.

    For multi-assay endpoints, metrics are first averaged across assays within
    each seed, then summarised across seeds.
    """
    out = []
    df = results[results["subset"] == subset]
    for endpoint_name, ep_df in df.groupby("endpoint", sort=False):
        task = get_endpoint(endpoint_name).task
        metrics = ["rmse", "mae", "r2", "spearman"] if task == "regression" else ["roc_auc", "pr_auc", "balanced_accuracy"]
        per_seed = ep_df.groupby(["split", "config", "model", "seed"], sort=False)[metrics].mean().reset_index()
        agg = per_seed.groupby(["split", "config", "model"], sort=False)[metrics].agg(["mean", "std"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg = agg.reset_index()
        agg.insert(0, "endpoint", endpoint_name)
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def select_config(results: pd.DataFrame, endpoint_name: str, split: str = "scaffold") -> ModelConfig:
    """Best configuration by mean validation score (never by test score, which would leak)."""
    task = get_endpoint(endpoint_name).task
    metric, _ = PRIMARY_METRIC[task]
    val = results[(results["endpoint"] == endpoint_name) & (results["split"] == split) & (results["subset"] == "valid")]
    scores = val.groupby("config")[metric].mean()
    best_name, best_score = None, None
    for name, score in scores.items():
        if name == "baseline":
            continue
        if is_better(task, float(score), best_score):
            best_name, best_score = name, float(score)
    if best_name is None:
        raise RuntimeError(f"No configuration could be scored for {endpoint_name}")
    return next(c for c in CONFIGS if c.name == best_name)


def train_final(
    endpoint_name: str,
    out_dir: Path,
    split: str = "scaffold",
    fast: bool = False,
    featurizer: Featurizer | None = None,
    scaffolder: Scaffolder | None = None,
    data_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Select, evaluate and save the production model for one endpoint."""
    endpoint, _, store, labels = _prepare(endpoint_name, featurizer, scaffolder, data_dir)

    # 1. Model selection on the validation set.
    selection = pd.DataFrame(_run_store(endpoint, store, labels, [split], [0], CONFIGS, fast, lambda _: None))
    config = select_config(selection, endpoint_name, split)
    log(f"[{endpoint_name}] selected {config.label} on {split} validation set")

    # 2. Honest estimate: train on train+valid, score on the untouched test set.
    train_idx, valid_idx, test_idx = store.split(split, 0)
    fit_idx = np.concatenate([train_idx, valid_idx])
    X = store.get(config.feature_set)
    per_target, predictions = {}, {}
    for t, target in enumerate(endpoint.targets):
        _, preds = _fit_predict(config, endpoint.task, X, labels[:, t], fit_idx, {"test": test_idx}, 0, fast)
        y_true, y_pred = preds["test"]
        per_target[target] = _score(endpoint.task, y_true, y_pred)
        predictions[target] = {"y_true": y_true.tolist(), "y_pred": np.asarray(y_pred).tolist()}

    # 3. Production model: refit on every labelled molecule.
    all_idx = np.arange(len(store.smiles))
    models = {}
    for t, target in enumerate(endpoint.targets):
        model, _ = _fit_predict(config, endpoint.task, X, labels[:, t], all_idx, {}, 0, fast)
        models[target] = model

    fingerprints = store.get("morgan")
    artifact = {
        "endpoint": endpoint_name,
        "task": endpoint.task,
        "targets": list(endpoint.targets),
        "units": endpoint.units,
        "config": config.name,
        "feature_set": config.feature_set,
        "models": models,
        "train_fingerprints": applicability.pack(fingerprints > 0),
        "n_bits": int(fingerprints.shape[1]),
        "n_train": int(len(store.smiles)),
        "test_metrics": per_target,
        "split": split,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_dir / f"{endpoint_name}.joblib", compress=3)

    summary = {
        "endpoint": endpoint_name,
        "admet": endpoint.admet,
        "model": config.label,
        "split": split,
        "n_molecules": int(len(store.smiles)),
        "n_test": int(len(test_idx)),
        "test_metrics": per_target,
        "test_predictions": predictions,
    }
    if endpoint.task == "classification":
        aucs = [m["roc_auc"] for m in per_target.values() if m and not np.isnan(m.get("roc_auc", np.nan))]
        summary["mean_test_roc_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    log(f"[{endpoint_name}] saved {out_dir / f'{endpoint_name}.joblib'}")
    return summary


def _clean(obj):
    """Make an object strict-JSON safe: NaN/inf become null, NumPy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.floating | float):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2))
