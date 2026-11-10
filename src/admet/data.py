"""Dataset registry: which ADMET endpoints exist and how to load them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("ADMET_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

TOX21_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


@dataclass(frozen=True)
class Endpoint:
    name: str
    admet: str            # which letter of ADMET this informs
    task: str             # "regression" or "classification"
    file: str
    targets: tuple[str, ...]
    description: str
    units: str = ""

    @property
    def is_regression(self) -> bool:
        return self.task == "regression"


ENDPOINTS: dict[str, Endpoint] = {
    "solubility": Endpoint(
        name="solubility",
        admet="Absorption",
        task="regression",
        file="esol.csv",
        targets=("logS",),
        units="log10(mol/L)",
        description="Aqueous solubility (ESOL, Delaney 2004). Poorly soluble drugs are poorly absorbed.",
    ),
    "tox21": Endpoint(
        name="tox21",
        admet="Toxicity",
        task="classification",
        file="tox21.csv.gz",
        targets=tuple(TOX21_ASSAYS),
        description="Tox21 panel: 12 in-vitro assays for nuclear-receptor and stress-response activity.",
    ),
}


def get_endpoint(name: str) -> Endpoint:
    try:
        return ENDPOINTS[name]
    except KeyError:
        raise KeyError(f"Unknown endpoint {name!r}; available: {sorted(ENDPOINTS)}") from None


def load_endpoint(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load an endpoint's table: a `smiles` column plus one column per target."""
    endpoint = get_endpoint(name)
    path = (data_dir or DATA_DIR) / endpoint.file
    df = pd.read_csv(path)
    missing = {"smiles", *endpoint.targets} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df["smiles"].astype(str).str.strip()
    return df[["smiles", *endpoint.targets]]
