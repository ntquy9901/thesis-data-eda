"""Tests for src.eda.phase12_embedding_price_correlation (Story 11-3)."""

import numpy as np
import pandas as pd
import pytest

from src.eda import phase12_embedding_price_correlation as P


def _synthetic_panel(n=200, seed=0):
    """emb_0: linear relationship with log_returns. emb_1: monotonic-but-not-linear
    (non-linear-only) relationship with log_returns via a cubic transform."""
    rng = np.random.default_rng(seed)
    emb_0 = rng.normal(size=n)
    emb_1 = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    log_returns = 0.8 * emb_0 + noise  # strong linear signal
    # emb_1 monotonic cubic relation with noise on a SEPARATE target so its
    # Pearson-vs-Spearman comparison isn't diluted by the emb_0 linear term.
    pk_t1 = np.sign(emb_1) * np.abs(emb_1) ** 3 + rng.normal(scale=0.02, size=n)
    return pd.DataFrame({
        "ticker": ["VCB"] * n,
        "date": pd.date_range("2019-01-01", periods=n, freq="D"),
        "emb_0": emb_0, "emb_1": emb_1,
        "log_returns": log_returns, "pk_t+1": pk_t1,
    })


def test_compute_correlations_linear_signal_detected():
    panel = _synthetic_panel()
    corr = P.compute_correlations(panel, ["emb_0", "emb_1"])
    row = corr[(corr.feature == "emb_0") & (corr.target == "log_returns")].iloc[0]
    assert row["pearson_r"] > 0.5
    assert row["fdr_pearson"] is True or row["fdr_pearson"] == True  # noqa: E712


def test_summarize_linear_vs_nonlinear_counts():
    panel = _synthetic_panel()
    corr = P.compute_correlations(panel, ["emb_0", "emb_1"])
    summary = P.summarize(corr)
    assert summary["linear_significant_count"] >= 1
    assert "top5_abs_pearson_r_per_target" in summary


def test_summarize_nonlinear_only_bucket():
    """Direct unit test of the summarize() bucketing logic: a dim with fdr_spearman=True but
    fdr_pearson=False must land in nonlinear_only_significant_count, not linear_significant_count."""
    corr = pd.DataFrame([
        {"feature": "emb_0", "target": "log_returns", "pearson_r": 0.6, "pearson_p": 0.001,
         "spearman_r": 0.6, "spearman_p": 0.001, "mi": 0.1, "fdr_pearson": True, "fdr_spearman": True},
        {"feature": "emb_1", "target": "pk_t+1", "pearson_r": 0.05, "pearson_p": 0.4,
         "spearman_r": 0.3, "spearman_p": 0.01, "mi": 0.05, "fdr_pearson": False, "fdr_spearman": True},
    ])
    summary = P.summarize(corr)
    assert summary["linear_significant_count"] == 1
    assert summary["nonlinear_only_significant_count"] == 1


def test_summarize_empty_corr():
    summary = P.summarize(pd.DataFrame())
    assert "note" in summary


def test_load_joined_panel_missing_files_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    assert P._load_joined_panel().empty


def test_load_joined_panel_empty_adv_returns_empty(monkeypatch, tmp_path):
    (tmp_path / "modeling").mkdir()
    pd.DataFrame(columns=["ticker", "date", "emb_0"]).to_parquet(tmp_path / "modeling" / "advanced_news_features.parquet")
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    assert P._load_joined_panel().empty


def test_load_joined_panel_no_price_files_returns_empty(monkeypatch, tmp_path):
    (tmp_path / "modeling").mkdir()
    (tmp_path / "price").mkdir()
    pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"]), "emb_0": [0.1]}).to_parquet(
        tmp_path / "modeling" / "advanced_news_features.parquet"
    )
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    assert P._load_joined_panel().empty


def test_compute_correlations_skips_feature_not_in_panel():
    panel = pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"]), "log_returns": [0.01]})
    corr = P.compute_correlations(panel, ["emb_missing"])
    assert corr.empty


def test_run_phase_empty_panel_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P, "_load_joined_panel", lambda: pd.DataFrame())
    assert P.run_phase() == []


def test_run_phase_empty_corr_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    panel = pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"])})  # no emb_* cols
    monkeypatch.setattr(P, "_load_joined_panel", lambda: panel)
    assert P.run_phase() == []


def test_real_run_phase_smoke():
    """Real EDA artifacts (if present) -> no crash, sane schema."""
    written = P.run_phase()
    if not written:
        pytest.skip("no advanced_news_features / price_metrics artifacts yet")
    names = {p.name for p in written}
    assert {"embedding_price_corr.csv", "embedding_price_corr_summary.json"} <= names
