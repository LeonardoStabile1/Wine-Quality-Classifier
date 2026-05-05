from typing import Optional
from pathlib import Path
import pandas as pd
from loguru import logger


def load_data(file_path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and return it as a pandas DataFrame.
    """
    if not file_path:
        logger.warning("No file path provided.")
        return None

    path = Path(file_path)

    if not path.exists():
        logger.error("File does not exist: {}", path)
        return None

    try:
        df = pd.read_csv(path)

        if df.empty or df.shape[1] == 0:
            logger.warning("Loaded file is empty or has no columns: {}", path)
            return None

        logger.info("Data loaded successfully from {}", path)
        return df

    except pd.errors.ParserError as e:
        logger.exception("CSV parsing failed for {}: {}", path, e)
        return None

    except UnicodeDecodeError as e:
        logger.exception("Encoding error while reading {}: {}", path, e)
        return None

    except Exception as e:
        logger.exception("Unexpected error while loading {}: {}", path, e)
        return None
