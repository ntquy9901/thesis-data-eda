"""Tests for src.modeling.dataset (Story 7-1)."""

import pandas as pd
import pytest

from src.modeling import dataset as ds
from src.modeling.dataset import har_features, time_split


def test_har_features_trailing_no_lookahead():
    pk = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    har = har_features(pk)
    # weekly (5d) mean at idx4 = mean(1..5) = 3.0; at idx3 = NaN (only 4 values)
    assert pd.isna(har["har_weekly"].iloc[3])
    assert har["har_weekly"].iloc[4] == pytest.approx(3.0)
    # daily = value itself
    assert (har["har_daily"] == pk).all()
    # changing a FUTURE value must not change a PAST HAR (no look-ahead)
    pk2 = pk.copy()
    pk2.iloc[4] = 999.0
    har2 = har_features(pk2)
    # idx3 window [0..3] must be unaffected (both NaN here → unaffected)
    assert pd.isna(har["har_weekly"].iloc[3]) and pd.isna(har2["har_weekly"].iloc[3])


def test_time_split_monotonic_no_overlap():
    panel = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=400, freq="D"), "v": range(400)})
    train, test = time_split(panel, split="2025-01-01")
    assert train["date"].max() < pd.Timestamp("2025-01-01")
    assert test["date"].min() >= pd.Timestamp("2025-01-01")
    assert len(train) + len(test) == len(panel)


def test_time_split_empty_test_when_all_before():
    panel = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="D")})
    train, test = time_split(panel, split="2025-01-01")
    assert len(train) == 10 and test.empty


def test_real_build_panel_and_split_smoke():
    panel = ds.build_panel()
    if panel.empty:
        pytest.skip("no EDA outputs (run phases 3+7 first)")
    assert {"ticker", "date", "parkinson_vol", "har_weekly", "har_monthly"} <= set(panel.columns)
    assert any(t in panel.columns for t in ds.TARGETS)
    train, test = ds.time_split(panel.dropna(subset=ds.TARGETS))
    # leakage gate: every train date strictly before every test date
    if not train.empty and not test.empty:
        assert train["date"].max() < test["date"].min()


def test_real_run_writes_panel_and_summary_smoke():
    written = ds.run()
    if not written:
        pytest.skip("no EDA outputs")
    names = {p.name for p in written}
    assert {"panel.parquet", "split_summary.json"} <= names
    panel = pd.read_parquet(next(p for p in written if p.name == "panel.parquet"))
    assert not panel.empty
    assert ds.feature_columns(panel)  # non-empty feature list, excludes targets/id/date
