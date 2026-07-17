"""Tests for src.eda.phase14_uncertainty_index (Story 12-2)."""

import pandas as pd
import pytest

from src.eda import phase14_uncertainty_index as P


def test_is_uncertain_all_three_categories_present():
    text = "Kinh tế Việt Nam đối mặt rủi ro do chính sách mới của Ngân hàng Nhà nước"
    assert P.is_uncertain(text) is True


def test_is_uncertain_missing_uncertainty_category():
    text = "Kinh tế Việt Nam tăng trưởng nhờ chính sách hỗ trợ doanh nghiệp"  # no risk/uncertainty term
    assert P.is_uncertain(text) is False


def test_is_uncertain_missing_policy_category():
    text = "Kinh tế tăng trưởng nhưng nhà đầu tư lo ngại biến động thị trường"  # no policy term
    assert P.is_uncertain(text) is False


def test_is_uncertain_missing_econ_category():
    text = "Chính sách của chính phủ gây lo ngại bất ổn"  # no econ term
    assert P.is_uncertain(text) is False


def test_is_uncertain_nfd_unicode_still_matches():
    """Vietnamese text in NFD (decomposed) form must still match NFC keyword literals."""
    import unicodedata

    text_nfc = "Kinh tế Việt Nam đối mặt rủi ro do chính sách mới của Ngân hàng Nhà nước"
    text_nfd = unicodedata.normalize("NFD", text_nfc)
    assert P.is_uncertain(text_nfd) is True


def test_build_uncertainty_index_no_sources_returns_empty(monkeypatch):
    monkeypatch.setattr(P, "discover_source_files", lambda: {})
    assert P.build_uncertainty_index().empty


def test_build_uncertainty_index_aggregation(monkeypatch):
    from pathlib import Path

    news = pd.DataFrame({
        "title": [
            "Kinh tế Việt Nam đối mặt rủi ro do chính sách mới của Ngân hàng Nhà nước",
            "Doanh nghiệp báo lãi quý này",
        ],
        "lead": ["", ""],
        "pub_date": pd.to_datetime(["2020-01-01T08:00:00", "2020-01-01T08:00:00"]),
        "source": ["cafef", "cafef"],
    })
    monkeypatch.setattr(P, "discover_source_files", lambda: {"cafef": Path("cafef_articles.csv")})
    monkeypatch.setattr(P, "load_source", lambda source, path: news)
    idx = P.build_uncertainty_index()
    if idx.empty:
        pytest.skip("no trading calendar available in this environment")
    assert (idx["n_articles"] == 2).any()


def test_load_joined_panel_empty_index_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "build_uncertainty_index", lambda: pd.DataFrame())
    assert P._load_joined_panel().empty


def test_load_joined_panel_averages_across_tickers_not_duplicates(monkeypatch, tmp_path):
    """Regression test for the pseudo-replication bug: joining must produce ONE row per date
    (market-wide average target), not one row per (ticker, date)."""
    idx = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "n_articles": [2], "n_uncertain": [1], "uncertainty_ratio": [0.5]})
    monkeypatch.setattr(P, "build_uncertainty_index", lambda: idx)
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    (tmp_path / "price").mkdir(parents=True)
    for ticker, val in [("VCB", 0.01), ("FPT", 0.03)]:
        pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "log_returns": [val]}).to_parquet(
            tmp_path / "price" / f"price_metrics_{ticker}.parquet"
        )
    monkeypatch.setattr(P, "EDA_TICKERS", ["VCB", "FPT"])
    panel = P._load_joined_panel()
    assert len(panel) == 1  # NOT 2 (one row per ticker) — averaged into a single market-wide row
    assert panel.iloc[0]["log_returns"] == pytest.approx(0.02)  # mean(0.01, 0.03)


def test_compute_uncertainty_correlations_basic():
    panel = pd.DataFrame({
        "uncertainty_ratio": [0.1, 0.5, 0.9, 0.3, 0.7],
        "pk_t+1": [0.001, 0.002, 0.003, 0.0015, 0.0025],
    })
    corr = P.compute_uncertainty_correlations(panel)
    assert len(corr) == 1
    assert corr.iloc[0]["feature"] == "uncertainty_ratio"


def test_compute_uncertainty_correlations_missing_feature_column_returns_empty():
    panel = pd.DataFrame({"pk_t+1": [0.001, 0.002, 0.003]})  # no uncertainty_ratio column
    assert P.compute_uncertainty_correlations(panel).empty


def test_summarize_empty_corr():
    summary = P.summarize(pd.DataFrame())
    assert "note" in summary


def test_summarize_basic():
    corr = pd.DataFrame([
        {"feature": "uncertainty_ratio", "target": "pk_t+1", "fdr_pearson": True, "fdr_spearman": True},
        {"feature": "uncertainty_ratio", "target": "pk_t+5", "fdr_pearson": False, "fdr_spearman": True},
    ])
    summary = P.summarize(corr)
    assert summary["linear_significant_count"] == 1
    assert summary["nonlinear_only_significant_count"] == 1


def test_run_phase_empty_index_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P, "build_uncertainty_index", lambda: pd.DataFrame())
    assert P.run_phase() == []


def test_run_phase_writes_index_even_if_no_price_match(monkeypatch, tmp_path):
    (tmp_path / "uncertainty").mkdir(parents=True)
    idx = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "n_articles": [1], "n_uncertain": [1], "uncertainty_ratio": [1.0]})
    monkeypatch.setattr(P, "EDA_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(P, "build_uncertainty_index", lambda: idx)
    monkeypatch.setattr(P, "_load_joined_panel", lambda: pd.DataFrame())
    written = P.run_phase()
    assert len(written) == 1
    assert written[0].name == "uncertainty_index.csv"


def test_real_run_phase_smoke():
    written = P.run_phase()
    if not written:
        pytest.skip("no news/price artifacts yet")
    assert written[0].name == "uncertainty_index.csv"
