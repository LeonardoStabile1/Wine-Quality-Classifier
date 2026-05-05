from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger

from src.data.ingestion import load_data
from src.data.transformation import split, apply_smote
from src.config.model import model_settings
from src.models.train import train_xgboost, save_artifacts, generate_reports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "wine_quality_red_white.csv"

MODEL_NAME = model_settings.model_name
MODEL_PATH = model_settings.model_path
REPORT_PATH = model_settings.report_path


def _validate_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        logger.error("Loaded dataframe is None")
        raise ValueError("Dataframe cannot be None")

    if df.empty:
        logger.error("Loaded dataframe is empty")
        raise ValueError("Dataframe cannot be empty")


def _validate_paths() -> None:
    if not isinstance(MODEL_NAME, str) or not MODEL_NAME:
        logger.error("Invalid MODEL_NAME: {}", MODEL_NAME)
        raise ValueError("MODEL_NAME must be a valid string")

    if not MODEL_PATH:
        logger.error("Invalid MODEL_PATH: {}", MODEL_PATH)
        raise ValueError("MODEL_PATH must be defined")

    if not REPORT_PATH:
        logger.error("Invalid REPORT_PATH: {}", REPORT_PATH)
        raise ValueError("REPORT_PATH must be defined")


def training_model() -> Tuple:
    """
    Execute the full training pipeline: data loading, preprocessing,
    model training, artifact persistence, and report generation.

    Returns:
        Tuple containing trained model, parameters, and evaluation outputs.
    """
    logger.info("Training pipeline started")

    _validate_paths()

    dataframe = load_data(DATA_PATH)
    _validate_dataframe(dataframe)

    X_train, X_test, y_train, y_test, scaler = split(dataframe)

    logger.info(
        "Data split completed | X_train={} X_test={}",
        X_train.shape, X_test.shape
    )

    X_train, y_train = apply_smote(X_train, y_train)

    logger.info("SMOTE applied | samples={}", len(X_train))

    model, params = train_xgboost(X_train, y_train)

    save_artifacts(MODEL_PATH, MODEL_NAME, model, params, scaler)

    y_pred = model.predict(X_test)

    report = generate_reports(REPORT_PATH, MODEL_NAME, y_test, y_pred)

    logger.info("Training pipeline finished successfully")

    return model, params, report


if __name__ == "__main__":
    training_model()
