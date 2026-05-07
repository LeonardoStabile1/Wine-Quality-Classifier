import pytest
import pandas as pd

from src.data.transformation import (
    remove_outliers,
    feature_engineering,
    split,
    apply_smote,
)


def make_valid_df(n=100):
    return pd.DataFrame({
        "density": [0.99]*n,
        "chlorides": [0.1]*n,
        "fixed acidity": [7]*n,
        "free sulfur dioxide": [30]*n,
        "citric acid": [0.3]*n,
        "total sulfur dioxide": [150]*n,
        "sulphates": [0.5]*n,
        "type": ["red"]*(n//2) + ["white"]*(n - n//2),
        "quality": [3,5,7]*(n//3) + [5]*(n % 3)
    })


def test_remove_outliers_basic():
    df = make_valid_df()
    result = remove_outliers(df)
    assert not result.empty
    assert set(result.columns) == set(df.columns)


def test_remove_outliers_filters():
    df = make_valid_df()
    df.loc[0, "density"] = 2.0
    result = remove_outliers(df)
    assert (result["density"] <= 1.01).all()


def test_feature_engineering_basic():
    df = make_valid_df()
    result = feature_engineering(df)

    assert "type" in result
    assert "quality" in result
    assert set(result["type"].unique()).issubset({0, 1})
    assert set(result["quality"].unique()).issubset({0, 1, 2})


def test_feature_engineering_invalid_type():
    df = make_valid_df()
    df.loc[0, "type"] = "invalid"
    result = feature_engineering(df)
    assert result["type"].isna().sum() >= 1


def test_split_basic():
    df = make_valid_df()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split(df)

    assert not X_train.empty
    assert not X_val.empty
    assert not X_test.empty
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_split_invalid_train_size():
    df = make_valid_df()
    with pytest.raises(ValueError):
        split(df, train_size=1.5)


def test_apply_smote_basic():
    df = make_valid_df()
    df = feature_engineering(df)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_res, y_res = apply_smote(X, y)

    assert len(X_res) >= len(X)
    assert len(y_res) == len(X_res)


def test_apply_smote_empty():
    with pytest.raises(ValueError):
        apply_smote(pd.DataFrame(), pd.Series(dtype=int))


def test_missing_columns():
    df = pd.DataFrame({"a": [1,2,3]})
    with pytest.raises(ValueError):
        remove_outliers(df)