"""Tests for src.modeling.significance (Epic 9)."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import significance as sig
from src.modeling.significance import (
    _ablation_block,
    bootstrap_delta,
    diebold_mariano,
    event_abnormal_ttest,
)


def test_diebold_mariano_identical_errors_not_significant():
    e = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2, -0.05, 0.15] * 5)
    d = diebold_mariano(e, e)
    assert d["dm_pvalue"] > 0.05  # identical → not significant


def test_diebold_mariano_clearly_different():
    rng = np.random.default_rng(0)
    e1 = rng.normal(0, 1, 500)  # larger errors
    e2 = rng.normal(0, 0.1, 500)  # much smaller errors
    d = diebold_mariano(e1, e2)
    assert d["dm_pvalue"] < 0.01  # model1 clearly worse
    assert d["dm_stat"] > 0  # positive = e1 larger loss


def test_bootstrap_delta_ci_contains_zero_when_equal():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    pred = y.copy()  # both perfect → Δ = 0
    b = bootstrap_delta(y, pred, pred, n_boot=200)
    assert b["n_boot"] == 200
    assert b["delta_rmse_ci"][0] <= 0 <= b["delta_rmse_ci"][1]


def test_event_abnormal_ttest_parses(tmp_path):
    csv = tmp_path / "event_study.csv"
    pd.DataFrame({"horizon": [1] * 30 + [5] * 30, "abnormal_vol": [0.001] * 30 + [0.0] * 30}).to_csv(csv)
    out = event_abnormal_ttest(csv)
    assert 1 in out and 5 in out
    assert isinstance(out[1]["pvalue"], float)


def _fake_panel_for_ablation(n_per_period: int = 150) -> pd.DataFrame:
    """Synthetic (ticker, date) panel with real PRICE_FEATURES + target columns, enough rows
    on both sides of SPLIT_DATE to fit price vs price+X models."""
    from src.modeling.baseline import PRICE_FEATURES
    from src.modeling.dataset import SPLIT_DATE, TARGETS

    rng = np.random.default_rng(0)
    train_dates = pd.date_range("2020-01-01", periods=n_per_period)
    test_dates = pd.date_range(pd.Timestamp(SPLIT_DATE), periods=n_per_period)
    dates = train_dates.append(test_dates)
    n = len(dates)
    df = pd.DataFrame({"ticker": ["AAA"] * n, "date": dates})
    for c in PRICE_FEATURES:
        df[c] = rng.normal(0, 1, n)
    for c in TARGETS:
        df[c] = rng.uniform(0, 0.01, n)
    return df


def test_ablation_block_ridge_and_gbm_run_without_error():
    panel = _fake_panel_for_ablation()
    lines: list[str] = []
    out = _ablation_block(panel, ["price"], "ridge", lines, "Test section")
    assert "price" in out
    assert any("Test section" in line for line in lines)
    # each target present in the fake panel got a DM+bootstrap entry
    from src.modeling.dataset import TARGETS

    for t in TARGETS:
        assert t in out["price"]
        assert "dm" in out["price"][t] and "bootstrap" in out["price"][t]


def test_ablation_block_gbm_model_type_used():
    panel = _fake_panel_for_ablation()
    lines: list[str] = []
    out = _ablation_block(panel, ["price"], "gbm", lines, "GBM section")
    assert "price" in out and out["price"]


def test_real_significance_run_smoke():
    written = sig.run()
    if not written:
        pytest.skip("no panel")
    names = {p.name for p in written}
    assert {"significance.json", "significance_report.md"} <= names
