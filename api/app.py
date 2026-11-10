"""HTTP API for ADMET predictions.

    uvicorn api.app:app --port 8080
    curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
         -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O"]}'

Interactive documentation is served at /docs.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from admet import __version__
from admet.predict import DEFAULT_MODEL_DIR, Predictor

MAX_MOLECULES = int(os.environ.get("ADMET_MAX_MOLECULES", "100"))
MAX_SMILES_LENGTH = 500

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup, not per request.
    state["predictor"] = Predictor.load(os.environ.get("ADMET_MODEL_DIR", DEFAULT_MODEL_DIR))
    yield
    state.clear()


app = FastAPI(
    title="ADMET Predictor",
    version=__version__,
    description=(
        "Predicts aqueous solubility and Tox21 toxicity-assay activity from a molecule's SMILES, "
        "with drug-likeness properties and an applicability-domain check. For research and "
        "education only; not for clinical or regulatory decisions."
    ),
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, examples=[["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]])

    @field_validator("smiles")
    @classmethod
    def check_size(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_MOLECULES:
            raise ValueError(f"At most {MAX_MOLECULES} molecules per request")
        too_long = [s for s in value if len(s) > MAX_SMILES_LENGTH]
        if too_long:
            raise ValueError(f"SMILES longer than {MAX_SMILES_LENGTH} characters are not accepted")
        return value


@app.get("/health")
def health() -> dict:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok", "endpoints": predictor.endpoints, "version": __version__}


@app.get("/models")
def models() -> dict:
    return state["predictor"].model_info()


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    results = state["predictor"].predict(request.smiles)
    return {"results": results, "n": len(results)}
