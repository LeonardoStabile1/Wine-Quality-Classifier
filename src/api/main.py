from fastapi import FastAPI, HTTPException
from pydantic import create_model
from typing import Any, List, Literal
import pandas as pd
import joblib
from pathlib import Path

from src.config.model import model_settings


ARTIFACT_PATH = Path(model_settings.model_path) / model_settings.model_name


if not ARTIFACT_PATH.exists():
    raise RuntimeError(f"Model artifacts not found at {ARTIFACT_PATH}")


artifacts = joblib.load(ARTIFACT_PATH)

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]


def build_models(feature_names: list[str]):
    fields = {}

    for feature in feature_names:
        if feature == "type":
            fields[feature] = (Literal["red", "white"], ...)
        else:
            fields[feature] = (float, ...)

    WineFeatures = create_model(
        "WineFeatures",
        **fields
    )

    PredictRequest = create_model(
        "PredictRequest",
        data=(List[WineFeatures], ...)
    )

    return PredictRequest


PredictRequest = build_models(features)


app = FastAPI(title="Wine Quality API")


@app.get("/")
async def health():
    return "OK"


@app.get("/ready")
def ready():
    if model is None or scaler is None:
        return {"status": "not_ready"}

    return {"status": "ready"}


@app.post("/predict")
def predict(payload: PredictRequest):
    df = pd.DataFrame(
        [item.model_dump() for item in payload.data]
    )

    if "type" in df.columns:
        df["type"] = df["type"].map({
            "red": 0,
            "white": 1
        })

    missing = set(features) - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}"
        )

    df = df.reindex(columns=features)

    X = scaler.transform(df)

    preds = model.predict(X)

    return {
        "predictions": preds.tolist()
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features": features
    }