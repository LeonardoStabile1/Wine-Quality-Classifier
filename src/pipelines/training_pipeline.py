import pandas as pd
from pathlib import Path

from src.data.ingestion import load_data
from src.data.transformation import split, apply_smote
from src.config.model import model_settings
from src.models.train import train_XGBoost, save_artifacts, generate_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "wine_quality_red_white.csv"
MODEL_NAME = model_settings.model_name
MODEL_PATH = model_settings.model_path
REPORT_PATH = model_settings.report_path

def training_model():
    dataframe = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = split(dataframe)
    X_train, y_train = apply_smote(X_train, y_train)
    model, params = train_XGBoost(X_train, y_train)
    save_artifacts(MODEL_PATH, MODEL_NAME, model, params, scaler)
    y_pred = model.predict(X_test)
    generate_reports(REPORT_PATH, MODEL_NAME, y_test, y_pred)

if __name__ == "__main__":
    training_model()
