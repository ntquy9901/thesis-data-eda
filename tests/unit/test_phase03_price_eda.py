"""Tests for src.eda.phase03_price_eda (pure helpers + runners + real-data smoke)."""

import numpy as np
import pandas as pd
import pytest

from src.eda import common
from src.eda import phase03_price_eda as p3
from src.eda.phase03_price_eda import (
    add_returns,
    average_true_range,
    detect_outliers,
    parkinson_targets,
    parkinson_volatility,
    price_metrics,
    realized_volatility,
    regime_changes,
    regime_flags,
    rolling_stats,
    volatility_clustering_test,
    volatility_targets,
)


# ============ leakage-safe targets (CRITICAL) ============
def test_volatility_targets_use_only_future_returns():
    lr = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    # rv_t+1 at index t = |lr[t+1]|
    t1 = volatility_targets(lr, horizons=(1,))
    assert abs(t1.loc[0, "rv_t+1"] - 0.02) < 1e-9
    assert abs(t1.loc[3, "rv_t+1"] - 0.05) < 1e-9
    assert pd.isna(t1.loc[4, "rv_t+1"])  # no future → NaN

    # rv_t+2 at t = sqrt(lr[t+1]^2 + lr[t+2]^2)
    t2 = volatility_targets(lr, horizons=(2,))
    assert abs(t2.loc[0, "rv_t+2"] - np.sqrt(0.02**2 + 0.03**2)) < 1e-9
    assert pd.isna(t2.loc[3, "rv_t+2"]) and pd.isna(t2.loc[4, "rv_t+2"])


def test_volatility_targets_no_lookahead_into_target_row():
    """Changing lr[t] must NOT change rv_t+h at index t (proves no look-ahead)."""
    base = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    altered = base.copy()
    altered.iloc[1] = 0.99  # change lr at index 1
    tb = volatility_targets(base, horizons=(1, 2))
    ta = volatility_targets(altered, horizons=(1, 2))
    # rv at index 1 must be identical (uses lr[2..], not lr[1])
    assert abs(tb.loc[1, "rv_t+1"] - ta.loc[1, "rv_t+1"]) < 1e-9
    assert abs(tb.loc[1, "rv_t+2"] - ta.loc[1, "rv_t+2"]) < 1e-9
    # but rv at index 0 DOES depend on lr[1]
    assert abs(ta.loc[0, "rv_t+1"] - 0.99) < 1e-9


# ============ core metrics ============
def test_add_returns_basic():
    df = pd.DataFrame({"date": ["2024-01-02", "2024-01-01"], "close": [110, 100]})
    out = add_returns(df)
    assert list(out["date"]) == ["2024-01-01", "2024-01-02"]  # sorted
    assert pd.isna(out["log_returns"].iloc[0])
    assert abs(out["log_returns"].iloc[1] - np.log(110 / 100)) < 1e-9


def test_average_true_range_handles_flat():
    s = pd.Series([100] * 6)
    atr = average_true_range(s, s, s, window=3)
    # high==low==close → TR=0 → ATR=0
    assert atr.dropna().eq(0).all()


def test_realized_volatility_known():
    lr = pd.Series([0.0, 0.1, 0.1])  # window 2
    rv = realized_volatility(lr, 2)
    assert pd.isna(rv.iloc[0])  # only 1 element in window
    assert abs(rv.iloc[1] - 0.1) < 1e-9  # sqrt(0^2 + 0.1^2)
    assert abs(rv.iloc[2] - np.sqrt(0.02)) < 1e-9  # sqrt(0.1^2 + 0.1^2)


# ============ Parkinson volatility (baseline-aligned target) ============
def test_parkinson_volatility_known_value():
    # H/L = 110/100 → ln(1.1)^2 / (4 ln2)
    high = pd.Series([110.0])
    low = pd.Series([100.0])
    pv = parkinson_volatility(high, low)
    assert abs(pv.iloc[0] - (np.log(1.1) ** 2) / (4 * np.log(2))) < 1e-12


def test_parkinson_volatility_masks_invalid():
    high = pd.Series([110.0, 100.0, 0.0])  # row1 high<low; row2 high=0
    low = pd.Series([100.0, 110.0, 100.0])
    pv = parkinson_volatility(high, low)
    assert pd.notna(pv.iloc[0])
    assert pd.isna(pv.iloc[1])  # high<low → NaN
    assert pd.isna(pv.iloc[2])  # high=0 → NaN


def test_parkinson_targets_leakage_safe():
    pv = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    tgt = parkinson_targets(pv, horizons=(1,))
    assert abs(tgt.loc[0, "pk_t+1"] - 0.02) < 1e-12  # future value at t=0
    assert pd.isna(tgt.loc[4, "pk_t+1"])  # no future → NaN


def test_price_metrics_has_parkinson_columns():
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=40, freq="D"),
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}
    )
    m = price_metrics(df)
    assert {"parkinson_vol", "pk_t+1", "pk_t+5", "pk_t+10", "pk_t+22"} <= set(m.columns)


def test_price_metrics_columns_and_nan_tail():
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=40, freq="D"),
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}
    )
    m = price_metrics(df)
    assert {"returns", "log_returns", "atr_14", "realized_vol_5d", "rv_t+1", "rv_t+10", "rv_t+22"} <= set(m.columns)
    assert m["rv_t+10"].isna().sum() >= 10  # NaN tail


# ============ diagnostics ============
def test_detect_outliers_flags_tail():
    s = pd.Series([-1, 0, 0, 0, 0, 0, 0, 0, 0, 5.0])
    mask = detect_outliers(s, threshold=2.0)
    assert mask.iloc[-1]  # 5.0 is the clear outlier
    assert not mask.iloc[0]  # -1 is within 2σ
    assert not mask.iloc[1]


def test_detect_outliers_zero_std():
    assert not detect_outliers(pd.Series([5, 5, 5])).any()


def test_rolling_stats_shape():
    rs = rolling_stats(pd.Series(range(100), dtype=float), windows=(20,))
    assert "mean_20" in rs.columns and "std_20" in rs.columns
    assert rs["mean_20"].isna().sum() == 19


def test_regime_flags_empty():
    assert regime_flags(pd.Series([np.nan, np.nan])).isna().all()


def test_volatility_clustering_insufficient():
    out = volatility_clustering_test(pd.Series([0.01] * 5))
    assert out["clustering"] == "insufficient_data"


def test_volatility_clustering_constant_is_insufficient():
    # constant squared returns have zero variance → must not falsely report "yes"
    out = volatility_clustering_test(pd.Series([0.01] * 30))
    assert out["clustering"] == "insufficient_data"


# ============ review-driven fixes ============
def test_add_returns_masks_nonfinite_from_zero_price():
    # close=0 → log return would be -inf; must be NaN, not ±inf
    df = pd.DataFrame({"date": ["d1", "d2", "d3"], "close": [10.0, 0.0, 10.0]})
    out = add_returns(df)
    assert not np.isinf(out["log_returns"]).any()
    assert pd.isna(out["log_returns"].iloc[1])  # the zero-price bar masked


def test_price_metrics_raises_on_missing_columns():
    df = pd.DataFrame({"date": [1, 2], "close": [10.0, 11.0]})  # no OHLC
    with pytest.raises(ValueError, match="missing"):
        price_metrics(df)


def test_regime_changes_detects_transitions():
    # half low-vol, half high-vol → at least one transition in the labeled region
    rv = pd.concat([pd.Series([0.01] * 80), pd.Series([0.5] * 80)], ignore_index=True)
    ch = regime_changes(rv, window=60, n_bins=3)
    assert len(ch) >= 1
    assert "regime" in ch.columns


# ============ integration runner (monkeypatched) ============
@pytest.fixture
def redirected(tmp_path, monkeypatch):
    price = tmp_path / "VCB_ohlcv.csv"
    rng = np.random.default_rng(0)
    n = 80
    rows = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "open": 100.0 + rng.normal(0, 1, n),
        "high": 102.0 + rng.normal(0, 1, n),
        "low": 98.0 + rng.normal(0, 1, n),
        "close": 100.0 + rng.normal(0, 1, n),
        "volume": rng.integers(100, 1000, n),
    })
    rows.to_csv(price, index=False, encoding="utf-8")
    monkeypatch.setattr(p3, "EDA_TICKERS", ["VCB"])
    monkeypatch.setattr(p3, "PRICE_DATA_DIR", tmp_path)
    out = tmp_path / "eda_output"
    monkeypatch.setattr(common, "EDA_OUTPUT_DIR", out)
    return out


def test_run_phase_writes_artifacts(redirected):
    written = p3.run_phase()
    names = {p.name for p in written}
    assert "price_metrics_VCB.parquet" in names
    assert "acf_pacf_VCB.png" in names
    assert "outliers_VCB.csv" in names
    assert "corr_heatmap.png" in names
    assert "rolling_vol.png" in names
    assert "findings.md" in names
    # parquet readable
    pq = redirected / "price" / "price_metrics_VCB.parquet"
    df = pd.read_parquet(pq)
    assert "rv_t+10" in df.columns


# ============ real-data sample smoke (per CLAUDE.md Testing quality rules) ============
def test_real_data_vcb_metrics_smoke():
    from config import PRICE_DATA_DIR

    path = PRICE_DATA_DIR / "VCB_ohlcv.csv"
    if not path.exists():
        pytest.skip("VCB real price data not available")
    df = pd.read_csv(path, encoding="utf-8")
    m = price_metrics(df)
    assert len(m) == len(df)
    # UTF-8 + date parse ran clean; targets have NaN tail (no look-ahead leakage)
    assert m["rv_t+10"].isna().sum() >= 10
    assert m["rv_t+1"].notna().sum() > 0
    # ATR finite on interior rows
    assert m["atr_14"].dropna().shape[0] > 100
