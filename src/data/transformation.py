import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["density"] <= 1.01]
    df = df[df["chlorides"] < 0.3]
    df = df[df["fixed acidity"] < 12]
    df = df[df["free sulfur dioxide"] < 100]
    df = df[df["citric acid"] < 0.75]
    df = df[df["total sulfur dioxide"] < 300]
    df = df[df["sulphates"] < 1.25]
    return df

def split(dataframe: pd.DataFrame, random_state: int = 42, train_size: float = 0.8):
    dataframe = feature_engineering(dataframe)

    X = dataframe.drop(columns=["quality"])
    y = dataframe["quality"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    train_df = X_train.assign(quality=y_train)
    train_df = remove_outliers(train_df)

    X_train = train_df.drop(columns=["quality"])
    y_train = train_df["quality"]

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler