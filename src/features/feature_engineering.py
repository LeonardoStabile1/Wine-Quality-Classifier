import pandas as pd


def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering transformations to the input dataset.

    This function performs:
    - Encoding of the 'type' column: {'red': 0, 'white': 1}
    - Binning of the 'quality' column into three categories:
        * Low (0–4)
        * Intermediate (5–6)
        * High (7–10)
    - Conversion of binned quality labels into ordinal integers:
        {'Low': 0, 'Intermediate': 1, 'High': 2}

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset containing at least 'type' and 'quality' columns.

    Returns
    -------
    pd.DataFrame
        A transformed copy of the input DataFrame with encoded features.

    Raises
    ------
    KeyError
        If required columns ('type', 'quality') are not present in the input.

    Notes
    -----
    - The original DataFrame is not modified.
    - Assumes 'quality' values are within the range [0, 10].
    """
    df = dataframe.copy()
    df["type"] = df["type"].map({"red": 0, "white": 1})
    df["quality"] = pd.cut(
        df["quality"],
        bins=[0, 4, 6, 10],
        labels=["Low", "Intermediate", "High"]
    )
    df["quality"] = df["quality"].map({"Low": 0, "Intermediate": 1, "High": 2})
    return df