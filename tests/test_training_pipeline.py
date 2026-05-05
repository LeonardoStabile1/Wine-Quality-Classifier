import pytest
import pandas as pd

from src.pipelines.training_pipeline import training_model


def test_training_pipeline_invalid_data(monkeypatch):
    monkeypatch.setattr("src.pipelines.training_pipeline.load_data", lambda _: None)

    with pytest.raises(ValueError):
        training_model()


def test_training_pipeline_empty_data(monkeypatch):
    monkeypatch.setattr("src.pipelines.training_pipeline.load_data", lambda _: pd.DataFrame())

    with pytest.raises(ValueError):
        training_model()
