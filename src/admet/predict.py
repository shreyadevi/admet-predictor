"""Predict ADMET properties for new molecules with the saved models."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

from . import applicability
from .data import get_endpoint

DEFAULT_MODEL_DIR = Path(os.environ.get("ADMET_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))


def solubility_class(log_s: float) -> str:
    """Rough solubility bands commonly used in early drug discovery."""
    if log_s >= -2:
        return "high"
    if log_s >= -4:
        return "moderate"
    if log_s >= -6:
        return "low"
    return "very low"


def drug_likeness(mol) -> dict:
    """Physicochemical profile, Lipinski rule-of-five violations and QED score."""
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    props = {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Crippen.MolLogP(mol), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
    }
    violations = sum([
        props["molecular_weight"] > 500,
        props["logp"] > 5,
        props["h_bond_donors"] > 5,
        props["h_bond_acceptors"] > 10,
    ])
    props["lipinski_violations"] = int(violations)
    props["qed"] = round(QED.qed(mol), 3)
    return props


class Predictor:
    """Loads every saved endpoint model and predicts for lists of SMILES."""

    def __init__(self, artifacts: dict[str, dict], ad_threshold: float = applicability.DEFAULT_THRESHOLD):
        if not artifacts:
            raise ValueError("No models loaded")
        self.artifacts = artifacts
        self.ad_threshold = ad_threshold
        self._train_fps = {
            name: applicability.unpack(a["train_fingerprints"], a["n_bits"]) for name, a in artifacts.items()
        }

    @classmethod
    def load(cls, model_dir: Path | str = DEFAULT_MODEL_DIR, **kwargs) -> Predictor:
        model_dir = Path(model_dir)
        artifacts = {p.stem: joblib.load(p) for p in sorted(model_dir.glob("*.joblib"))}
        if not artifacts:
            raise FileNotFoundError(f"No *.joblib models in {model_dir}. Run `admet train` first.")
        return cls(artifacts, **kwargs)

    @property
    def endpoints(self) -> list[str]:
        return sorted(self.artifacts)

    def model_info(self) -> dict:
        return {
            name: {
                "model": a["config"],
                "feature_set": a["feature_set"],
                "targets": a["targets"],
                "n_train": a["n_train"],
                "trained_at": a["trained_at"],
                "test_metrics": a["test_metrics"],
            }
            for name, a in self.artifacts.items()
        }

    def predict(self, smiles: list[str]) -> list[dict]:
        from . import featurize as fz

        mols = [fz.standardize(s) for s in smiles]
        results = [{"input": s, "valid": m is not None} for s, m in zip(smiles, mols)]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        for i, result in enumerate(results):
            if mols[i] is None:
                result["error"] = "Could not parse SMILES"
            else:
                result["canonical_smiles"] = fz.canonical_smiles(mols[i])
                result["properties"] = drug_likeness(mols[i])
        if not valid_idx:
            return results

        valid_mols = [mols[i] for i in valid_idx]
        features: dict[str, np.ndarray] = {}
        fps = fz.featurize_mols(valid_mols, "morgan")

        for name, artifact in self.artifacts.items():
            fs = artifact["feature_set"]
            if fs not in features:
                features[fs] = fz.featurize_mols(valid_mols, fs)
            X = features[fs]
            similarity = applicability.max_tanimoto(fps > 0, self._train_fps[name])
            endpoint = get_endpoint(name)

            for row, i in enumerate(valid_idx):
                entry = {
                    "nearest_training_similarity": round(float(similarity[row]), 3),
                    "in_domain": bool(similarity[row] >= self.ad_threshold),
                }
                if endpoint.is_regression:
                    value = float(artifact["models"][endpoint.targets[0]].predict(X[row:row + 1])[0])
                    entry.update(value=round(value, 3), units=artifact["units"])
                    if name == "solubility":
                        entry["category"] = solubility_class(value)
                else:
                    probs = {}
                    for target, model in artifact["models"].items():
                        classes = list(model.classes_)
                        p = model.predict_proba(X[row:row + 1])[0]
                        probs[target] = round(float(p[classes.index(1)]) if 1 in classes else 0.0, 3)
                    entry["probabilities"] = probs
                    entry["flagged_assays"] = sorted(t for t, p in probs.items() if p >= 0.5)
                results[i][name] = entry
        return results
