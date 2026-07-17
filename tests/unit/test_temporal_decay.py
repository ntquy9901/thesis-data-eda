"""Tests for src.eda.phase15_temporal_decay_correlation (Story 12-3)."""

import pandas as pd
import pytest

from src.eda import phase15_temporal_decay_correlation as P


def test_load_joined_panel_empty_decayed_returns_empty(monkeypatch):
    monkeypatch.setattr(P, "decayed_embedding_features", lambda group, halflife_days: pd.DataFrame())
    assert P._load_joined_panel().empty


def test_load_joined_panel_no_price_files_returns_empty(monkeypatch, tmp_path):
    decayed = pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"]), "emb_decay_0": [0.1]})
    monkeypatch.setattr(P, "decayed_embedding_features", lambda group, halflife_days: decayed)
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    assert P._load_joined_panel().empty


def test_compute_decay_correlations_basic():
    panel = pd.DataFrame({
        "emb_decay_0": [0.1, 0.5, 0.9, 0.3, 0.7],
        "emb_decay_1": [0.2, 0.4, 0.6, 0.1, 0.9],
        "log_returns": [0.01, 0.02, 0.03, 0.015, 0.025],
    })
    corr = P.compute_decay_correlations(panel)
    assert len(corr) == 2
    assert set(corr["feature"]) == {"emb_decay_0", "emb_decay_1"}


def test_run_phase_empty_panel_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P, "_load_joined_panel", lambda: pd.DataFrame())
    assert P.run_phase() == []


def test_run_phase_empty_corr_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    panel = pd.DataFrame({"ticker": ["VCB"], "date": pd.to_datetime(["2020-01-01"])})  # no emb_decay_* cols
    monkeypatch.setattr(P, "_load_joined_panel", lambda: panel)
    assert P.run_phase() == []


def test_real_run_phase_smoke():
    written = P.run_phase()
    if not written:
        pytest.skip("no news/price artifacts yet")
    assert written[0].name == "decay_price_corr.csv"
