"""Tests for src.modeling.significance (Epic 9)."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import significance as sig
from src.modeling.significance import (
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


def test_real_significance_run_smoke():
    written = sig.run()
    if not written:
        pytest.skip("no panel")
    names = {p.name for p in written}
    assert {"significance.json", "significance_report.md"} <= names
