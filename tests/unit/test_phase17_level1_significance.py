"""Tests for src.eda.phase17_level1_significance (Story 14-1)."""

import numpy as np
import pandas as pd
import pytest

from src.eda import phase17_level1_significance as P17


def _fake_panel(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "positive_score": rng.uniform(0, 1, n),
        "negative_score": rng.uniform(0, 1, n),
        "pk_t+1": rng.uniform(0, 0.01, n),
        "pk_t+5": rng.uniform(0, 0.01, n),
    })


def test_compute_level1_correlations_has_all_five_stats():
    panel = _fake_panel()
    corr = P17.compute_level1_correlations(panel, ["positive_score", "negative_score"])
    expected_cols = {"pearson_r", "spearman_r", "kendall_tau", "mi", "dcor", "fdr_pearson", "fdr_spearman", "fdr_kendall"}
    assert expected_cols <= set(corr.columns)
    assert len(corr) == 2 * 2  # 2 features x 2 targets present in fake panel


def test_compute_level1_correlations_missing_feature_skipped():
    panel = _fake_panel()
    corr = P17.compute_level1_correlations(panel, ["not_a_real_feature"])
    assert corr.empty


def test_summarize_empty():
    s = P17.summarize(pd.DataFrame())
    assert "note" in s


def test_summarize_flags_likely_useless_and_nonlinear_candidates():
    corr = pd.DataFrame([
        {"feature": "a", "target": "t1", "pearson_r": 0.5, "mi": 0.5, "dcor": 0.5,
         "fdr_pearson": True, "fdr_spearman": True, "fdr_kendall": True},
        {"feature": "b", "target": "t1", "pearson_r": 0.01, "mi": 0.0, "dcor": 0.0,
         "fdr_pearson": False, "fdr_spearman": False, "fdr_kendall": False},
        {"feature": "c", "target": "t1", "pearson_r": 0.005, "mi": 0.05, "dcor": 0.2,
         "fdr_pearson": False, "fdr_spearman": True, "fdr_kendall": False},
    ])
    s = P17.summarize(corr)
    assert s["n_feature_target_pairs"] == 3
    assert s["likely_useless_mi_near_zero_count"] == 1
    assert s["nonlinear_candidate_pearson_near_zero_mi_positive_count"] == 1
    assert s["nonlinear_candidates"][0]["feature"] == "c"


def test_load_joined_panel_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P17, "EDA_OUTPUT_DIR", tmp_path)
    assert P17._load_joined_panel().empty


# ============ real-data smoke ============
@pytest.mark.slow  # MI/dcor over the full sentiment_features x price panel (~1.45M-row corpus)
def test_real_phase17_run_smoke():
    written = P17.run_phase()
    if not written:
        pytest.skip("no sentiment_features.parquet / price_metrics (run sentiment_scores + phase03 first)")
    assert any("level1_corr.csv" in str(p) for p in written)
