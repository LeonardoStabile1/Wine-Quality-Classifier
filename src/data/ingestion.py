from typing import Optional
from pathlib import Path
import pandas as pd

def load_data(file_path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and return it as a pandas DataFrame.

    Args:
        file_path (Optional[str]): Path to the CSV file.

    Returns:
        Optional[pd.DataFrame]: A DataFrame if the file is valid and contains data;
        otherwise None (e.g., invalid path, file does not exist, empty data,
        or parsing/decoding errors).
    """
    if not file_path:
        return None

    path = Path(file_path)

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        if df.empty or df.shape[1] == 0:
            return None

        return df

    except (pd.errors.ParserError, UnicodeDecodeError):
        return None