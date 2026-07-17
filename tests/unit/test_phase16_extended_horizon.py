"""Tests for src.eda.phase16_extended_horizon_correlation."""

import pandas as pd
import pytest

from src.eda import phase16_extended_horizon_correlation as P


def test_load_joined_panel_no_adv_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    assert P._load_joined_panel().empty


def test_load_joined_panel_no_price_files_returns_empty(monkeypatch, tmp_path):
    (tmp_path / "modeling").mkdir()
    pd.DataFrame({"ticker": ["ZZZ"], "date": pd.to_datetime(["2020-01-01"]), "emb_0": [0.1]}).to_parquet(
        tmp_path / "modeling" / "advanced_news_features.parquet"
    )
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P, "EDA_TICKERS", ["ZZZ"])  # no matching OHLCV file on disk
    assert P._load_joined_panel().empty


def test_compute_extended_correlations_basic():
    panel = pd.DataFrame({
        "emb_0": [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.6],
        "pk_t+15": [0.01, 0.02, 0.03, 0.015, 0.025, 0.012, 0.022],
        "pk_t+20": [0.02, 0.01, 0.04, 0.02, 0.03, 0.018, 0.028],
    })
    corr = P.compute_extended_correlations(panel)
    assert len(corr) == 2
    assert set(corr["target"]) == {"pk_t+15", "pk_t+20"}


def test_compute_extended_correlations_empty_features():
    panel = pd.DataFrame({"pk_t+15": [0.01, 0.02, 0.03]})  # no emb_* columns
    assert P.compute_extended_correlations(panel).empty


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
    written = P.run_phase()
    if not written:
        pytest.skip("no news/price artifacts yet")
    assert written[0].name == "extended_horizon_corr.csv"
