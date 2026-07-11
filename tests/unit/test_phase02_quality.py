"""Unit tests for src.eda.phase02_quality (pure helpers, synthetic data)."""

import numpy as np
import pandas as pd

from src.eda.phase02_quality import (
    date_gap_count,
    duplicates_for,
    invalid_values_report,  # noqa: F401 (import sanity)
    missingness_by_stock,
    missingness_for,
    price_invalid,
)


# ---- missingness_for ----
def test_missingness_for_computes_pct():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    m = missingness_for(df, "t")
    a_row = m[m["column"] == "a"].iloc[0]
    assert a_row["n_missing"] == 1
    assert a_row["pct"] == round(1 / 3 * 100, 2)
    assert len(m) == 2


def test_missingness_for_empty():
    assert missingness_for(pd.DataFrame(), "t").empty


# ---- missingness_by_stock ----
def test_missingness_by_stock_aggregates_cells():
    frames = {"VCB": pd.DataFrame({"a": [1, None], "b": [1, 1]})}
    ms = missingness_by_stock(frames)
    assert len(ms) == 1
    row = ms.iloc[0]
    assert row["ticker"] == "VCB"
    assert row["n_missing_cells"] == 1
    assert row["pct_missing"] == round(1 / 4 * 100, 2)


# ---- date_gap_count ----
def test_date_gap_count_detects_gaps():
    # Mon 2024-01-01 ... Fri 2024-01-05 = 5 business days; provide only 2.
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-05"]})
    assert date_gap_count(df) == 3  # 5 expected - 2 present


def test_date_gap_count_no_gap():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    assert date_gap_count(df) == 0


def test_date_gap_count_missing_col():
    assert date_gap_count(pd.DataFrame({"x": [1]})) == 0
    assert date_gap_count(pd.DataFrame()) == 0


# ---- duplicates_for (NaN excluded) ----
def test_duplicates_for_counts_on_key():
    df = pd.DataFrame({"url": ["a", "a", "b"], "title": ["x", "x", "y"]})
    assert duplicates_for(df, ["url"]) == 1
    assert duplicates_for(df, ["title"]) == 1


def test_duplicates_for_excludes_nan_keys():
    # Three NaN urls must NOT be counted as duplicates of each other.
    df = pd.DataFrame({"url": ["a", "b", np.nan, np.nan, np.nan]})
    assert duplicates_for(df, ["url"]) == 0


def test_duplicates_for_missing_cols():
    assert duplicates_for(pd.DataFrame({"x": [1, 2]}), ["url"]) == 0


# ---- price_invalid (tz-safe, numeric coercion, NaN-aware) ----
def test_price_invalid_flags_all_issues():
    df = pd.DataFrame(
        {
            "open": [10, 10, 10, 10],
            "high": [11, 8, 11, 11],  # row 1: high < low
            "low": [9, 9, 9, 9],
            "close": [10, 10, 10, 10],
            "volume": [100, -5, 100, "abc"],  # row 1: negative; row 3: unparseable
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2099-01-01"],  # row 3: future
        }
    )
    inv = price_invalid(df)
    assert inv["negative_volume"] == 1
    assert inv["unparseable_volume"] == 1  # 'abc' coerced to NaN
    assert inv["high_lt_low"] == 1
    assert inv["future_dates"] == 1


def test_price_invalid_tz_aware_dates_do_not_crash():
    # tz-aware pub_date must not raise TypeError vs tz-naive now().
    df = pd.DataFrame({"volume": [1], "high": [2.0], "low": [1.0], "pub_date": ["2024-01-01T00:00:00+07:00"]})
    inv = price_invalid(df)  # should not raise
    assert inv["future_dates"] == 0


def test_price_invalid_empty():
    assert price_invalid(pd.DataFrame()) == {
        "negative_volume": 0,
        "unparseable_volume": 0,
        "high_lt_low": 0,
        "future_dates": 0,
    }
