"""Tests for src.eda.phase18_event_study_by_type (Story 14-2)."""

import numpy as np
import pandas as pd
import pytest

from src.eda import phase18_event_study_by_type as P18


def test_market_benchmark_returns_empty_input():
    assert P18.market_benchmark_returns({}).empty


def test_market_benchmark_returns_equal_weighted_mean():
    idx = pd.date_range("2024-01-01", periods=3)
    a = pd.Series([0.01, 0.02, 0.03], index=idx)
    b = pd.Series([0.03, 0.00, 0.01], index=idx)
    mkt = P18.market_benchmark_returns({"A": a, "B": b})
    assert mkt.iloc[0] == pytest.approx(0.02)
    assert mkt.iloc[1] == pytest.approx(0.01)


def test_event_type_window_metrics_includes_car():
    pk = pd.Series([0.01] * 21)
    lr = pd.Series([0.0] * 21)
    abnormal = pd.Series([0.001] * 21)  # constant positive abnormal return
    m = P18.event_type_window_metrics(10, pk, lr, abnormal, horizons=(1, 5))
    assert list(m["horizon"]) == [1, 5]
    # post_car over horizon h = h * 0.001 (constant abnormal return summed)
    assert m.loc[m["horizon"] == 5, "post_car"].iloc[0] == pytest.approx(0.005)
    assert m.loc[m["horizon"] == 1, "post_car"].iloc[0] == pytest.approx(0.001)


def test_event_type_window_metrics_edge_event_returns_none():
    pk = pd.Series([0.01] * 5)
    lr = pd.Series([0.0] * 5)
    abnormal = pd.Series([0.0] * 5)
    m = P18.event_type_window_metrics(0, pk, lr, abnormal, horizons=(1,))
    assert m["pre_car"].iloc[0] is None


def test_event_days_by_type_filters_ticker_and_type():
    exploded = pd.DataFrame({
        "ticker": ["VCB", "VCB", "FPT"],
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")],
        "event_earnings": [1, 0, 1],
        "event_ma": [0, 1, 0],
    })
    days = P18.event_days_by_type(exploded, "VCB", "earnings")
    assert days == [pd.Timestamp("2024-01-02")]


def test_event_days_by_type_missing_column_returns_empty():
    assert P18.event_days_by_type(pd.DataFrame({"ticker": ["VCB"], "date": [pd.Timestamp("2024-01-02")]}), "VCB", "earnings") == []


def test_event_days_by_type_empty_frame_returns_empty():
    assert P18.event_days_by_type(pd.DataFrame(), "VCB", "earnings") == []


# ============ real-data smoke ============
@pytest.mark.slow  # per-ticker/event-type event-day loop over the full ~1.45M-row corpus
def test_real_phase18_run_smoke():
    written = P18.run_phase()
    if not written:
        pytest.skip("no price_metrics / sentiment article data available")
    assert any("event_study_by_type.csv" in str(p) for p in written)
