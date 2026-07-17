"""Tests for src.dashboard.data (loaders) + app import smoke (Epic 10)."""

import pandas as pd
import pytest

from src.dashboard import data as D


def test_load_panel_smoke():
    df = D.load_panel()
    if df.empty:
        pytest.skip("no panel.parquet (run modeling dataset first)")
    assert {"ticker", "date"}.issubset(df.columns)


def test_available_tickers_smoke():
    tk = D.available_tickers()
    if not tk:
        pytest.skip("no price_metrics")
    assert all(isinstance(t, str) for t in tk)
    assert len(tk) >= 1


def test_load_metrics_smoke():
    m = D.load_metrics()
    if m.empty:
        pytest.skip("no metrics.csv")
    assert {"target", "model", "feature_set", "r2"}.issubset(m.columns)


def test_load_significance_smoke():
    s = D.load_significance()
    if not s:
        pytest.skip("no significance.json")
    assert "per_target" in s


def test_headline_metrics_smoke():
    hm = D.headline_metrics()
    if not hm:
        pytest.skip("no metrics/significance")
    # one entry per target with the expected keys
    first = next(iter(hm.values()))
    assert {"r2_price", "rmse_price", "dir_acc", "dm_pvalue"} <= set(first)


def test_load_news_embedding_source_stats_smoke():
    df = D.load_news_embedding_source_stats()
    if df.empty:
        pytest.skip("no source_stats.csv")
    assert "source" in df.columns


def test_load_news_embedding_coverage_smoke():
    df = D.load_news_embedding_coverage()
    if df.empty:
        pytest.skip("no embedding_coverage.csv")
    assert "group" in df.columns


def test_load_embedding_price_corr_smoke():
    df = D.load_embedding_price_corr()
    if df.empty:
        pytest.skip("no embedding_price_corr.csv")
    assert {"feature", "target"} <= set(df.columns)


def test_load_extended_horizon_corr_smoke():
    df = D.load_extended_horizon_corr()
    if df.empty:
        pytest.skip("no extended_horizon_corr.csv")
    assert {"feature", "target"} <= set(df.columns)


def test_load_articles_list_khach_quan_smoke():
    df = D.load_articles_list("khach_quan", source="cafef", limit=10)
    if df.empty:
        pytest.skip("no cafef_articles.csv")
    assert "title" in df.columns
    assert len(df) <= 10


def test_load_articles_list_tong_hop_smoke():
    df = D.load_articles_list("tong_hop", limit=10)
    if df.empty:
        pytest.skip("no tong_hop-group articles found")
    assert "title" in df.columns
    assert set(df["source"]) <= {"ssi", "vndirect", "vnstock", "vietstock", "vsdc"}


def test_load_articles_list_groups_are_mutually_exclusive():
    """Regression test: khach_quan and tong_hop must never share a source (the original
    file-based split had 'tong_hop' as the union of all sources, so every khach_quan article
    trivially also appeared in tong_hop — this is the fix for that user-reported issue)."""
    from src.features.news_embeddings import GROUP_SOURCES

    assert set(GROUP_SOURCES["khach_quan"]).isdisjoint(GROUP_SOURCES["tong_hop"])


def test_load_articles_list_missing_file_returns_empty(monkeypatch):
    monkeypatch.setattr("src.data.discover_news.discover_source_files", lambda: {})
    assert D.load_articles_list("tong_hop").empty


def test_loaders_with_missing_base_return_empty(tmp_path):
    # no artifacts under an empty base dir → loaders must not raise
    assert D.load_panel(tmp_path).empty
    assert D.load_metrics(tmp_path).empty
    assert D.load_significance(tmp_path) == {}
    assert D.available_tickers(tmp_path) == []
    assert D.load_json("news/x.json", tmp_path) == {}
    assert D.load_text("report/y.md", tmp_path) == ""
    assert D.load_news_embedding_source_stats(tmp_path).empty
    assert D.load_news_embedding_coverage(tmp_path).empty
    assert D.load_embedding_price_corr(tmp_path).empty
    assert D.load_extended_horizon_corr(tmp_path).empty


def test_app_imports():
    # Streamlit app module must import cleanly (page builders are thin compositions)
    import src.dashboard.app as app  # noqa: F401

    assert set(app.PAGES.keys()) == {
        "Overview", "Price EDA", "News EDA", "News Embedding", "Embedding Correlation",
        "Đọc tin tức", "Modeling", "Significance",
    }


def test_page_news_embedding_no_data_warns(monkeypatch):
    """When no source_stats.csv exists, the page must warn (not crash)."""
    from streamlit.testing.v1 import AppTest

    import src.dashboard.data as D

    monkeypatch.setattr(D, "load_news_embedding_source_stats", lambda *a, **kw: pd.DataFrame())
    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("News Embedding")
    at.run()
    assert not at.exception
    assert any("No source_stats.csv" in w.value for w in at.warning)


def test_page_embedding_correlation_no_data_warns(monkeypatch):
    """When embedding_price_corr.csv is missing/invalid, the page must warn (not crash)."""
    from streamlit.testing.v1 import AppTest

    import src.dashboard.data as D

    monkeypatch.setattr(D, "load_embedding_price_corr", lambda *a, **kw: pd.DataFrame())
    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("Embedding Correlation")
    at.run()
    assert not at.exception
    assert any("embedding_price_corr.csv" in w.value for w in at.warning)


def test_app_runs_all_pages_headless():
    """Exercise every page via streamlit's headless AppTest (covers app.py logic)."""
    import os

    if not os.path.isdir("eda_output/price"):
        pytest.skip("no artifacts for dashboard")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception, f"default page raised: {at.exception}"
    for page in ["Price EDA", "News EDA", "News Embedding", "Embedding Correlation",
                 "Đọc tin tức", "Modeling", "Significance", "Overview"]:
        at.sidebar.radio(key="page").set_value(page)
        at.run()
        assert not at.exception, f"page {page} raised: {at.exception}"
