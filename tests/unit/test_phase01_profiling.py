"""Unit tests for src.eda.phase01_profiling (pure helpers, synthetic data)."""

import pandas as pd

from src.eda.phase01_profiling import (
    detect_date_range,
    detect_keys,
    dtype_summary,
    profile_dataframe,
)


def test_dtype_summary_counts_dtypes():
    df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"]})
    s = dtype_summary(df)
    # pandas >=2.0 reports string columns as "str", older as "object".
    assert "int" in s and "float" in s and ("str" in s or "object" in s)


def test_detect_date_range_finds_pub_date():
    df = pd.DataFrame({"pub_date": ["2024-01-01", "2024-06-01"], "v": [1, 2]})
    col, dmin, dmax = detect_date_range(df)
    assert col == "pub_date"
    assert dmin < dmax


def test_detect_date_range_no_date_col():
    df = pd.DataFrame({"x": [1, 2]})
    col, dmin, dmax = detect_date_range(df)
    assert col is None


def test_detect_keys_prefers_id_then_url():
    df = pd.DataFrame({"id": [1], "url": ["http://x"], "title": ["t"]})
    assert detect_keys(df) == ("id", "url")

    df2 = pd.DataFrame({"article_url": ["http://x"], "title": ["t"]})
    assert detect_keys(df2) == (None, "article_url")


def test_profile_dataframe_shape():
    df = pd.DataFrame({"id": [1, 2, 3], "pub_date": ["2024-01-01", "2024-02-01", "2024-03-01"]})
    row = profile_dataframe("test", df)
    assert row["table"] == "test"
    assert row["row_count"] == 3
    assert row["col_count"] == 2
    assert row["primary_key"] == "id"
    assert row["date_col"] == "pub_date"
    assert row["memory_mb"] >= 0
