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


def test_train_xgboost_basic():
    X, y = make_classification_data()

    model, params = train_xgboost(X, y, n_iter=2, cv=2)

    assert model is not None
    assert isinstance(params, dict)


def test_train_xgboost_invalid_input():
    with pytest.raises(ValueError):
        train_xgboost(None, None)

    with pytest.raises(ValueError):
        train_xgboost(pd.DataFrame(), np.array([]))


def test_train_xgboost_mismatch():
    X, y = make_classification_data()
    with pytest.raises(ValueError):
        train_xgboost(X, y[:-1])


def test_save_artifacts_basic(tmp_path):
    dummy_model = {"a": 1}
    params = {"param": 1}

    save_artifacts(
        path=str(tmp_path),
        model_name="model.joblib",
        model=dummy_model,
        params=params,
        scaler=None,
    )

    file_path = tmp_path / "model.joblib"
    assert file_path.exists()


def test_save_artifacts_invalid():
    with pytest.raises(ValueError):
        save_artifacts("", "model.joblib", {}, {})

    with pytest.raises(ValueError):
        save_artifacts("path", "", {}, {})

    with pytest.raises(ValueError):
        save_artifacts("path", "model.joblib", None, {})

    with pytest.raises(ValueError):
        save_artifacts("path", "model.joblib", {}, "not_dict")


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
