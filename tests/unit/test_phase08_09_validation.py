"""Tests for src.eda.phase08_feature_validation and src.eda.phase09_leakage."""

import numpy as np
import pandas as pd
import pytest

from src.eda import phase08_feature_validation as p8
from src.eda import phase09_leakage as p9
from src.eda.phase08_feature_validation import (
    collinear_groups,
    duplicate_columns,
    missingness,
    near_zero_variance,
    train_test_drift,
)
from src.eda.phase09_leakage import dates_monotonic_per_group, target_leakage_flags


# ============ phase08 ============
def test_missingness_pct():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    m = missingness(df).set_index("column")
    assert m.loc["a", "pct_missing"] == round(1 / 3 * 100, 2)
    assert m.loc["b", "pct_missing"] == 0.0


def test_near_zero_variance_flags_constant():
    df = pd.DataFrame({"const": [5, 5, 5, 5], "varied": [1, 2, 3, 4]})
    assert near_zero_variance(df) == ["const"]


def test_duplicate_columns_detected():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [9, 8, 7]})
    groups = duplicate_columns(df)
    assert [["a", "b"]] == groups or [["b", "a"]] == groups


def test_collinear_groups():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 100))
    df = pd.DataFrame({"x": x, "x_dup": x * 2 + 0.001, "unrelated": rng.normal(0, 1, 100)})
    groups = collinear_groups(df, threshold=0.9)
    flat = [c for g in groups for c in g]
    assert "x" in flat and "x_dup" in flat and "unrelated" not in flat


def test_train_test_drift_runs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=200, freq="D"),
        "a": list(rng.normal(0, 1, 100)) + list(rng.normal(5, 1, 100)),  # clear drift at midpoint
    })
    d = train_test_drift(df, "date", "2023-04-10")
    a_row = d[d["column"] == "a"].iloc[0]
    assert a_row["drift"] is True or a_row["drift"] == True  # noqa: E712


# ============ phase09 ============
def test_dates_monotonic_per_group():
    df = pd.DataFrame({"ticker": ["A"] * 3 + ["B"] * 3, "date": pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03"] * 2)})
    assert dates_monotonic_per_group(df, "ticker", "date")
    bad = pd.DataFrame({"ticker": ["A"] * 3, "date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])})
    assert not dates_monotonic_per_group(bad, "ticker", "date")


def test_target_leakage_flags_perfect_correlation():
    rng = np.random.default_rng(0)
    feat = pd.Series(rng.normal(0, 1, 200))
    df = pd.DataFrame({"sentiment_mean": feat, "pk_t+1": feat + rng.normal(0, 0.01, 200)})  # ~perfect
    flags = target_leakage_flags(df, ["pk_t+1"], threshold=0.95)
    assert any(f["feature"] == "sentiment_mean" for f in flags)


def test_target_leakage_none_when_uncorrelated():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sentiment_mean": rng.normal(0, 1, 200), "pk_t+1": rng.normal(0, 1, 200)})
    assert target_leakage_flags(df, ["pk_t+1"], threshold=0.95) == []


# ============ real-data smoke ============
def test_real_phase08_run_smoke():
    written = p8.run_phase()
    if not written:
        pytest.skip("no feature matrix (run phase03+phase07 first)")
    names = {p.name for p in written}
    assert {"feature_report.csv", "collinearity.json", "drop_recommendations.json"} <= names


def test_real_phase09_leakage_list_smoke():
    written = p9.run_phase()
    if not written:
        pytest.skip("no feature matrix")
    names = {p.name for p in written}
    assert {"leakage_list.md", "leakage_checks.json"} <= names
    # the explicit leakage list must mention targets + split
    text = next(p for p in written if p.name == "leakage_list.md").read_text(encoding="utf-8")
    assert "fixed" in text.lower() and "split" in text.lower()
