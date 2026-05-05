from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from loguru import logger


REQUIRED_COLUMNS = {
    "density",
    "chlorides",
    "fixed acidity",
    "free sulfur dioxide",
    "citric acid",
    "total sulfur dioxide",
    "sulphates",
    "type",
    "quality",
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error("Missing required columns: {}", missing)
        raise ValueError(f"Missing columns: {missing}")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and filter extreme values."""
    _validate_columns(df)

    initial_shape = df.shape
    df = df.copy().drop_duplicates()

    filters = (
        (df["density"] <= 1.01) &
        (df["chlorides"] < 0.3) &
        (df["fixed acidity"] < 12) &
        (df["free sulfur dioxide"] < 100) &
        (df["citric acid"] < 0.75) &
        (df["total sulfur dioxide"] < 300) &
        (df["sulphates"] < 1.25)
    )

    df = df.loc[filters]

    logger.info(
        "Outliers removed | before={} after={}",
        initial_shape, df.shape
    )

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and discretize the target."""
    _validate_columns(df)

    df = df.copy()

    if not set(df["type"].unique()).issubset({"red", "white"}):
        logger.warning("Unexpected values in 'type': {}", df["type"].unique())

    df["type"] = df["type"].map({"red": 0, "white": 1})

    df["quality"] = pd.cut(
        df["quality"],
        bins=[0, 4, 6, 10],
        labels=[0, 1, 2]
    ).astype("int64")

    logger.info("Feature engineering applied | shape={}", df.shape)

    return df


def split(
    df: pd.DataFrame,
    random_state: int = 42,
    train_size: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split data, remove outliers, and scale features."""
    _validate_columns(df)

    if not (0 < train_size < 1):
        logger.error("Invalid train_size: {}", train_size)
        raise ValueError("train_size must be between 0 and 1")

    df = feature_engineering(df)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    train_df = X_train.assign(quality=y_train)
    train_df = remove_outliers(train_df)

    X_train = train_df.drop(columns=["quality"])
    y_train = train_df["quality"]

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info(
        "Split and scaling completed | X_train={} X_test={}",
        X_train_scaled.shape, X_test_scaled.shape
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy="auto",
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance the training data."""
    if X_train.empty or y_train.empty:
        logger.error("Empty training data for SMOTE")
        raise ValueError("Training data cannot be empty")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE applied | before={} after={}",
        X_train.shape, X_res.shape
    )

    return X_res, y_res
