import json
import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    train_xgboost,
    save_artifacts,
    generate_reports,
)


def make_classification_data(n=120):
    X = pd.DataFrame({
        "f1": np.random.rand(n),
        "f2": np.random.rand(n),
        "f3": np.random.rand(n),
    })
    y = np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))
    return X, y


def test_generate_reports_basic(tmp_path):
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])

    report = generate_reports(
        reports_path=str(tmp_path),
        model_name="test_model",
        y_true=y_true,
        y_pred=y_pred,
    )

    file_path = tmp_path / "test_model_report.json"

    assert file_path.exists()
    assert "accuracy" in report
    assert "classification_report" in report

    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["accuracy"] == report["accuracy"]


def test_generate_reports_invalid():
    with pytest.raises(ValueError):
        generate_reports("path", "model", None, None)

    with pytest.raises(ValueError):
        generate_reports("path", "model", [], [])

    with pytest.raises(ValueError):
        generate_reports("path", "model", [1, 2], [1])
