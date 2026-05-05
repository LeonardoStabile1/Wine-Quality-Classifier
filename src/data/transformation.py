import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def remove_outliers(df):
    """Remove duplicates and filter extreme values."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df[df["density"] <= 1.01]
    df = df[df["chlorides"] < 0.3]
    df = df[df["fixed acidity"] < 12]
    df = df[df["free sulfur dioxide"] < 100]
    df = df[df["citric acid"] < 0.75]
    df = df[df["total sulfur dioxide"] < 300]
    df = df[df["sulphates"] < 1.25]
    return df

def feature_engineering(df):
    """Encode categorical features and discretize the target."""
    df = df.copy()
    df["type"] = df["type"].map({"red": 0, "white": 1})
    df["quality"] = pd.cut(
        df["quality"],
        bins=[0, 4, 6, 10],
        labels=["Low", "Intermediate", "High"]
    )
    df["quality"] = df["quality"].map({"Low": 0, "Intermediate": 1, "High": 2})
    return df

def split(dataframe, random_state=42, train_size=0.8):
    """Split data, remove outliers, and scale features."""
    dataframe = feature_engineering(dataframe)
    X = dataframe.drop(columns=["quality"])
    y = dataframe["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    train_df = X_train.assign(quality=y_train)
    train_df = remove_outliers(train_df)
    X_train = train_df.drop(columns=["quality"])
    y_train = train_df["quality"]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    return X_train, X_test, y_train, y_test, scaler

def apply_smote(
    X_train,
    y_train,
    sampling_strategy="auto",
    k_neighbors=5,
    random_state=42
):
    """Apply SMOTE to balance the training data."""
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled