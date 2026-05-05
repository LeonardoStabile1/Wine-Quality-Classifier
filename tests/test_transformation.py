import pandas as pd
import numpy as np

from src.data.transformation import (
    remove_outliers,
    feature_engineering,
    split,
    apply_smote
)


def make_sample_df():
    return pd.DataFrame({
        "fixed acidity": [7.0, 20.0],
        "volatile acidity": [0.7, 0.5],
        "citric acid": [0.0, 1.0],
        "residual sugar": [1.9, 2.0],
        "chlorides": [0.08, 0.5],
        "free sulfur dioxide": [11.0, 200.0],
        "total sulfur dioxide": [34.0, 400.0],
        "density": [0.9978, 1.05],
        "pH": [3.51, 3.0],
        "sulphates": [0.56, 2.0],
        "alcohol": [9.4, 10.0],
        "type": ["red", "white"],
        "quality": [5, 8]
    })


# -------------------------
# remove_outliers
# -------------------------

def test_remove_outliers_removes_extreme_values():
    df = make_sample_df()
    cleaned = remove_outliers(df)
    assert len(cleaned) == 1


def test_remove_outliers_removes_duplicates():
    df = pd.concat([make_sample_df().iloc[[0]], make_sample_df().iloc[[0]]])
    cleaned = remove_outliers(df)

    assert len(cleaned) == 1


def test_feature_engineering_type_encoding():
    df = make_sample_df()
    transformed = feature_engineering(df)

    assert set(transformed["type"].unique()) <= {0, 1}


def test_feature_engineering_quality_binning():
    df = make_sample_df()
    transformed = feature_engineering(df)
    assert set(transformed["quality"].unique()) <= {0, 1, 2}

def test_split_shapes():
    df = make_sample_df().loc[[0]].copy()

    df = pd.concat([df]*10, ignore_index=True)

    X_train, X_test, y_train, y_test, scaler = split(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_scaling():
    df = make_sample_df().loc[[0]].copy()
    df = pd.concat([df]*10, ignore_index=True)

    X_train, X_test, _, _, _ = split(df)
    assert np.allclose(X_train.mean(), 0, atol=1e-1)

