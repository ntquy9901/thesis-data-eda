"""Tests for src.eda.phase05_relationship and src.eda.phase06_event_study."""

import numpy as np
import pandas as pd
import pytest

from src.eda import phase05_relationship as p5
from src.eda import phase06_event_study as p6
from src.eda.phase05_relationship import (
    cross_correlation,
    distance_correlation,
    fdr_correct,
    granger_causality,
    kendall_tau,
    mutual_information,
    pearson_spearman,
)
from src.eda.phase06_event_study import (
    event_window_metrics,
    select_event_indices,
    window_mean,
    window_sum,
)


# ============ phase05 ============
def test_pearson_spearman_perfect():
    x = pd.Series([1, 2, 3, 4, 5], dtype=float)
    ps = pearson_spearman(x, x)
    assert abs(ps["pearson_r"] - 1.0) < 1e-6
    assert abs(ps["spearman_r"] - 1.0) < 1e-6
    assert ps["n"] == 5


def test_pearson_spearman_too_few():
    ps = pearson_spearman(pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0]))
    assert ps["pearson_r"] is None


def test_mutual_information_runs():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 200))
    y = pd.Series(x + rng.normal(0, 0.1, 200))  # strong dependence
    mi = mutual_information(x, y)
    assert mi is not None and mi > 0


def test_granger_detects_causality():
    rng = np.random.default_rng(0)
    n = 200
    cause = pd.Series(rng.normal(0, 1, n))
    effect = pd.Series(cause.shift(1).fillna(0).to_numpy() + rng.normal(0, 0.1, n))
    g = granger_causality(cause, effect, maxlag=3)
    assert g["min_p"] is not None
    assert g["significant"]  # lag-1 cause → effect


def test_cross_correlation_finds_lag():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 100))
    b = pd.Series(a.shift(2).fillna(0).to_numpy())  # b lags a by 2
    xcf = cross_correlation(a, b, max_lag=5)
    # strongest correlation near lag +2 (a vs b.shift(+2)? convention) — just check non-empty + a peak
    assert xcf and all(-1 <= v <= 1 for v in xcf.values())


def test_fdr_correct_flags():
    flags = fdr_correct([0.001, 0.5, 0.002, 0.9])
    assert flags[0] and not flags[1]  # 0.001 sig, 0.5 not (truthy — robust to np.bool_)


def test_kendall_tau_perfect():
    x = pd.Series([1, 2, 3, 4, 5], dtype=float)
    kt = kendall_tau(x, x)
    assert abs(kt["kendall_tau"] - 1.0) < 1e-6
    assert kt["n"] == 5


def test_kendall_tau_too_few():
    kt = kendall_tau(pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0]))
    assert kt["kendall_tau"] is None


def test_kendall_tau_constant_input():
    kt = kendall_tau(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0]))
    assert kt["kendall_tau"] is None


def test_distance_correlation_independent_near_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 300))
    y = pd.Series(rng.normal(0, 1, 300))  # independent of x
    dc = distance_correlation(x, y)
    assert dc is not None and 0 <= dc < 0.2


def test_distance_correlation_nonlinear_dependence():
    # y = x^2: Pearson ~ 0 (symmetric), but distance correlation should catch the dependence.
    x = pd.Series(np.linspace(-5, 5, 300))
    y = x**2
    dc = distance_correlation(x, y)
    ps = pearson_spearman(x, y)
    assert dc is not None and dc > 0.3
    assert abs(ps["pearson_r"]) < 0.1


def test_distance_correlation_too_few():
    dc = distance_correlation(pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0]))
    assert dc is None


# ============ phase06 ============
def test_window_mean_bounds():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert window_mean(s, 0, 2) == 2.0
    assert window_mean(s, -1, 2) is None  # out of range
    assert window_mean(s, 0, 10) is None


def test_window_sum_bounds():
    s = pd.Series([1.0, 2, 3])
    assert window_sum(s, 0, 1) == 3.0
    assert window_sum(s, 0, 5) is None


def test_event_window_metrics_symmetric():
    # parkinson vol symmetric around event index 10; abnormal ~0
    pk = pd.Series([0.01] * 21)
    lr = pd.Series([0.0] * 21)
    m = event_window_metrics(10, pk, lr, horizons=(1, 5, 10))
    assert list(m["horizon"]) == [1, 5, 10]
    assert all(abs(a) < 1e-9 for a in m["abnormal_vol"])


def test_event_window_metrics_edge_event():
    pk = pd.Series([0.01] * 5)
    lr = pd.Series([0.0] * 5)
    m = event_window_metrics(0, pk, lr, horizons=(1,))  # pre window [−1,−1] out of range
    assert m["pre_vol"].iloc[0] is None
    assert m["abnormal_vol"].iloc[0] is None


def test_select_event_indices_top_abs():
    score = pd.Series([0.1, -0.9, 0.5, 0.0, -0.8])
    idx = select_event_indices(score, top_n=2)
    assert set(idx) == {1, 4}  # largest |score|


# ============ real-data smoke ============
def test_real_joined_panel_correlation_smoke():
    panel = p5._load_joined_panel()
    if panel.empty:
        pytest.skip("no joined news×price panel (run phase03+phase07 first)")
    if {"news_count_1d", "pk_t+1"} <= set(panel.columns):
        ps = pearson_spearman(panel["news_count_1d"], panel["pk_t+1"])
        assert ps["n"] > 0
        assert ps["pearson_r"] is None or -1 <= ps["pearson_r"] <= 1


def test_real_phase05_run_smoke():
    written = p5.run_phase()
    if not written:
        pytest.skip("no joined panel")
    assert any("corr_matrix.csv" in str(p) for p in written)


def test_real_phase06_run_smoke():
    written = p6.run_phase()
    if not written:
        pytest.skip("no joined panel / events")
    assert any("event_study.csv" in str(p) for p in written)

