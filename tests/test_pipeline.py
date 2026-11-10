"""End-to-end checks of benchmarking, model selection, training and reporting,
using synthetic data and a stand-in featurizer (no RDKit needed)."""

import json

import joblib
import numpy as np
import pandas as pd

from admet.models import CONFIGS_BY_NAME
from admet.pipeline import _clean, benchmark, save_json, summarize, train_final
from admet.report import benchmark_markdown, plot_benchmark, plot_parity

from .conftest import fake_featurizer, fake_scaffolder

CONFIGS = tuple(CONFIGS_BY_NAME[n] for n in ("baseline", "rf_morgan", "gbm_desc"))


def _run(data_dir, endpoints=("solubility", "tox21"), seeds=(0, 1)):
    return benchmark(endpoints, ("random", "scaffold"), seeds, configs=CONFIGS, fast=True,
                     featurizer=fake_featurizer, scaffolder=fake_scaffolder, data_dir=data_dir,
                     log=lambda _: None)


def test_benchmark_produces_rows_for_every_combination(synthetic_data):
    results = _run(synthetic_data)
    sol = results[results["endpoint"] == "solubility"]
    assert len(sol) == 2 * 2 * len(CONFIGS) * 1 * 2          # splits x seeds x configs x targets x subsets
    tox = results[results["endpoint"] == "tox21"]
    assert set(tox["target"]) == {
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    }
    assert results["rmse"].dropna().ge(0).all()


def test_models_beat_baseline_on_learnable_signal(synthetic_data):
    summary = summarize(_run(synthetic_data, endpoints=("solubility",)))
    random_split = summary[summary["split"] == "random"].set_index("config")
    assert random_split.loc["gbm_desc", "rmse_mean"] < random_split.loc["baseline", "rmse_mean"]


def test_summary_has_mean_and_std_per_config(synthetic_data):
    summary = summarize(_run(synthetic_data))
    assert {"rmse_mean", "rmse_std", "roc_auc_mean", "pr_auc_std"} <= set(summary.columns)
    assert len(summary) == 2 * 2 * len(CONFIGS)               # endpoints x splits x configs


def test_report_markdown_and_plots(synthetic_data, tmp_path):
    summary = summarize(_run(synthetic_data))
    md = benchmark_markdown(summary, n_seeds=2)
    assert "## solubility (Absorption)" in md and "## tox21 (Toxicity)" in md
    assert "± " in md
    paths = plot_benchmark(summary, tmp_path)
    assert [p.name for p in paths] == ["benchmark_solubility.png", "benchmark_tox21.png"]
    assert all(p.stat().st_size > 5000 for p in paths)
    parity = plot_parity([-1, -2, -3], [-1.2, -2.1, -2.5], "t", "log", tmp_path / "parity.png")
    assert parity.stat().st_size > 5000


def test_train_final_saves_a_loadable_artifact(synthetic_data, tmp_path):
    summary = train_final("tox21", tmp_path / "models", fast=True, featurizer=fake_featurizer,
                          scaffolder=fake_scaffolder, data_dir=synthetic_data, log=lambda _: None)
    artifact = joblib.load(tmp_path / "models" / "tox21.joblib")
    assert artifact["task"] == "classification"
    assert set(artifact["models"]) == set(artifact["targets"])
    assert artifact["config"] != "baseline"
    assert artifact["n_train"] == len(pd.read_csv(synthetic_data / "tox21.csv.gz")) - 2  # two invalid dropped
    assert 0.0 <= summary["mean_test_roc_auc"] <= 1.0
    # Stored fingerprints round-trip for the applicability-domain check
    from admet.applicability import unpack
    fps = unpack(artifact["train_fingerprints"], artifact["n_bits"])
    assert fps.shape == (artifact["n_train"], artifact["n_bits"])


def test_invalid_smiles_are_excluded(synthetic_data):
    results = _run(synthetic_data, endpoints=("solubility",), seeds=(0,))
    total = results[(results["config"] == "baseline") & (results["split"] == "random")]["n"].sum()
    n_valid = len(pd.read_csv(synthetic_data / "esol.csv")) - 2
    # valid + test subsets together cover 20% of valid molecules
    assert total == round(0.1 * n_valid) + (n_valid - round(0.8 * n_valid) - round(0.1 * n_valid))


def test_json_output_is_strict(tmp_path):
    save_json({"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(2)]}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": 1.5, "c": [2]}
    assert _clean({"k": float("inf")}) == {"k": None}
