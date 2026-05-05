import pandas as pd
from pathlib import Path
import pytest

from src.data.ingestion import load_data


def test_load_data_valid_file(tmp_path):
    file = tmp_path / "test.csv"
    df_expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_expected.to_csv(file, index=False)

    df = load_data(str(file))

    assert df is not None
    pd.testing.assert_frame_equal(df, df_expected)


def test_load_data_file_not_exists():
    df = load_data("file_not_exists.csv")
    assert df is None


def test_load_data_empty_path():
    assert load_data("") is None
    assert load_data(None) is None  # type: ignore


def test_load_data_invalid_csv(tmp_path):
    file = tmp_path / "bad.csv"
    file.write_bytes(b"\x00\x00\x00\x00")

    df = load_data(str(file))

    assert df is None