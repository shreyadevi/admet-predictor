"""Turn SMILES strings into numbers a model can learn from.

Two complementary representations:

* **Morgan fingerprints (ECFP4)** - 2048 bits, each marking whether a circular
  substructure of radius 2 is present. Good at "this molecule contains a group
  like that one".
* **RDKit 2D descriptors** - ~200 computed physicochemical properties (logP,
  polar surface area, ring counts, ...). Good at global properties such as
  solubility.

RDKit is imported lazily so the rest of the package (splits, metrics, models)
can be used and tested without it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
FEATURE_SETS = ("morgan", "rdkit2d", "morgan+rdkit2d")

# Descriptors that are numerically unstable (Ipc spans many orders of magnitude)
# or duplicate information already captured elsewhere.
_EXCLUDED_DESCRIPTORS = {"Ipc"}
_DESCRIPTOR_CLIP = 1e6


@lru_cache(maxsize=1)
def _rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - exercised only without RDKit
        raise ImportError("RDKit is required for featurisation: pip install rdkit") from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, Descriptors, rdFingerprintGenerator, rdMolStandardize, MurckoScaffold


@lru_cache(maxsize=1)
def _morgan_generator():
    _, _, rdFingerprintGenerator, _, _ = _rdkit()
    return rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)


@lru_cache(maxsize=1)
def descriptor_names() -> list[str]:
    _, Descriptors, _, _, _ = _rdkit()
    return [name for name, _ in Descriptors._descList if name not in _EXCLUDED_DESCRIPTORS]


def standardize(smiles: str):
    """Parse and clean a SMILES string. Returns an RDKit Mol, or None if invalid.

    Keeps the largest fragment (drops salts and solvents) and neutralises
    charges, so the same compound drawn as different salt forms maps to the
    same structure.
    """
    Chem, _, _, rdMolStandardize, _ = _rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:  # standardisation failures fall back to the parsed molecule
        pass
    return mol


def canonical_smiles(mol) -> str:
    Chem, _, _, _, _ = _rdkit()
    return Chem.MolToSmiles(mol)


def morgan_fingerprint(mol) -> np.ndarray:
    return _morgan_generator().GetFingerprintAsNumPy(mol).astype(np.uint8)


def rdkit_descriptors(mol) -> np.ndarray:
    _, Descriptors, _, _, _ = _rdkit()
    functions = dict(Descriptors._descList)
    values = []
    for name in descriptor_names():
        try:
            values.append(float(functions[name](mol)))
        except Exception:
            values.append(np.nan)
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=_DESCRIPTOR_CLIP, neginf=-_DESCRIPTOR_CLIP)
    return np.clip(arr, -_DESCRIPTOR_CLIP, _DESCRIPTOR_CLIP)


def featurize_mols(mols, feature_set: str) -> np.ndarray:
    """Featurise already-parsed molecules (all must be valid)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {FEATURE_SETS}")
    blocks = []
    if "morgan" in feature_set:
        blocks.append(np.vstack([morgan_fingerprint(m) for m in mols]).astype(np.float32))
    if "rdkit2d" in feature_set:
        blocks.append(np.vstack([rdkit_descriptors(m) for m in mols]).astype(np.float32))
    return np.hstack(blocks)


def featurize(smiles: list[str], feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Featurise SMILES. Returns (X for valid molecules, boolean mask of which inputs were valid)."""
    mols = [standardize(s) for s in smiles]
    valid = np.array([m is not None for m in mols], dtype=bool)
    if not valid.any():
        return np.zeros((0, 0), dtype=np.float32), valid
    return featurize_mols([m for m in mols if m is not None], feature_set), valid


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold: the ring systems and the linkers between them.

    Molecules sharing a scaffold belong to the same chemical series. Acyclic
    molecules have an empty scaffold.
    """
    _, _, _, _, MurckoScaffold = _rdkit()
    mol = standardize(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def murcko_scaffolds(smiles: list[str]) -> list[str]:
    return [murcko_scaffold(s) for s in smiles]
