"""Tests for src.eda.phase13_novelty_correlation (Story 12-1)."""

import pandas as pd
import pytest

from src.eda import phase13_novelty_correlation as P


def test_load_joined_panel_empty_novelty_returns_empty(monkeypatch):
    monkeypatch.setattr(P, "novelty_daily", lambda group, window_days: pd.DataFrame())
    assert P._load_joined_panel().empty


def test_load_joined_panel_no_price_files_returns_empty(monkeypatch, tmp_path):
    novelty = pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"]), "novelty_mean": [0.5]})
    monkeypatch.setattr(P, "novelty_daily", lambda group, window_days: novelty)
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    assert P._load_joined_panel().empty


def test_compute_novelty_correlations_basic():
    panel = pd.DataFrame({
        "novelty_mean": [0.1, 0.5, 0.9, 0.3, 0.7],
        "log_returns": [0.01, 0.02, 0.03, 0.015, 0.025],
    })
    corr = P.compute_novelty_correlations(panel)
    assert len(corr) == 1
    assert corr.iloc[0]["feature"] == "novelty_mean"
    assert "fdr_pearson" in corr.columns


def test_run_phase_empty_panel_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P, "_load_joined_panel", lambda: pd.DataFrame())
    assert P.run_phase() == []


def test_run_phase_empty_corr_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    # panel HAS a target column but is missing novelty_mean -> compute_novelty_correlations
    # must guard this (not raise KeyError) and return empty.
    panel = pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"]), "log_returns": [0.01]})
    monkeypatch.setattr(P, "_load_joined_panel", lambda: panel)
    assert P.run_phase() == []


def test_compute_novelty_correlations_missing_feature_column_returns_empty():
    panel = pd.DataFrame({"log_returns": [0.01, 0.02, 0.03]})  # no novelty_mean column
    assert P.compute_novelty_correlations(panel).empty


def test_real_run_phase_smoke():
    written = P.run_phase()
    if not written:
        pytest.skip("no news/price artifacts yet")
    assert written[0].name == "novelty_price_corr.csv"
