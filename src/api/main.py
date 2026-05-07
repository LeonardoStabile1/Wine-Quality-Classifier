from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
import joblib
from pathlib import Path

from src.config.model import model_settings


class WineFeatures(BaseModel):
    density: float
    chlorides: float
    fixed_acidity: float
    free_sulfur_dioxide: float
    citric_acid: float
    total_sulfur_dioxide: float
    sulphates: float
    type: Literal["red", "white"]


class PredictRequest(BaseModel):
    data: List[WineFeatures]


app = FastAPI(title="Wine Quality API")


ARTIFACT_PATH = Path(model_settings.model_path) / model_settings.model_name


model = None
scaler = None
features = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, features

    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Model artifacts not found at {ARTIFACT_PATH}")

    artifacts = joblib.load(ARTIFACT_PATH)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["features"]


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
    if model is None or scaler is None or features is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([item.dict() for item in payload.data])

    df["type"] = df["type"].map({"red": 0, "white": 1})

    missing = set(features) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}"
        )

    df = df.reindex(columns=features)

    X = scaler.transform(df)
    preds = model.predict(X)

    return {"predictions": preds.tolist()}


@app.get("/model-info")
def model_info():
    if model is None:
        return {"status": "not_ready"}

    return {
        "model_type": type(model).__name__,
        "features": features
    }