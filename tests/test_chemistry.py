"""Tests that need RDKit: featurisation, scaffolds, real-data training, predictions and the API.
Skipped automatically when RDKit is not installed."""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from admet import featurize as fz  # noqa: E402
from admet.predict import Predictor, solubility_class  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_featurize_shapes_and_invalid_mask():
    X, valid = fz.featurize([ASPIRIN, "not_a_smiles", "CCO"], "morgan+rdkit2d")
    assert list(valid) == [True, False, True]
    assert X.shape == (2, fz.FINGERPRINT_BITS + len(fz.descriptor_names()))
    assert np.isfinite(X).all()


def test_fingerprint_is_binary():
    X, _ = fz.featurize([ASPIRIN], "morgan")
    assert set(np.unique(X)) <= {0.0, 1.0}
    assert X.sum() > 10


def test_salts_are_stripped_and_charges_neutralised():
    salt = fz.standardize("CC(=O)[O-].[Na+]")
    assert fz.canonical_smiles(salt) == "CC(=O)O"


def test_murcko_scaffolds():
    assert fz.murcko_scaffold("Cc1ccccc1") == "c1ccccc1"            # toluene -> benzene
    assert fz.murcko_scaffold("CCCCO") == ""                        # acyclic -> empty
    assert fz.murcko_scaffold(ASPIRIN) == fz.murcko_scaffold("OC(=O)c1ccccc1")


def test_solubility_bands():
    assert solubility_class(-1) == "high"
    assert solubility_class(-3) == "moderate"
    assert solubility_class(-5) == "low"
    assert solubility_class(-8) == "very low"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from admet.pipeline import train_final

    out = tmp_path_factory.mktemp("models")
    train_final("solubility", out, fast=True, log=lambda _: None)
    return out


def test_real_solubility_model_is_useful(model_dir):
    import joblib

    artifact = joblib.load(model_dir / "solubility.joblib")
    metrics = artifact["test_metrics"]["logS"]
    # The target's spread is ~2.1 log units; a useful model does much better than that.
    assert metrics["rmse"] < 1.5
    assert metrics["r2"] > 0.5


def test_predictor_output(model_dir):
    predictor = Predictor.load(model_dir)
    out = predictor.predict([ASPIRIN, "garbage("])
    good, bad = out
    assert good["valid"] and not bad["valid"]
    assert "error" in bad
    sol = good["solubility"]
    assert -6 < sol["value"] < 1
    assert sol["in_domain"] is True
    assert good["properties"]["lipinski_violations"] == 0
    assert 0 < good["properties"]["qed"] < 1


def test_api(model_dir, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ADMET_MODEL_DIR", str(model_dir))
    from api.app import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", json={"smiles": [ASPIRIN]})
        assert r.status_code == 200
        assert r.json()["results"][0]["solubility"]["units"] == "log10(mol/L)"
        assert client.post("/predict", json={"smiles": []}).status_code == 422
        assert client.post("/predict", json={"smiles": ["C"] * 101}).status_code == 422
