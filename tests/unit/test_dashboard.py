"""Tests for src.dashboard.data (loaders) + app import smoke (Epic 10)."""

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


def test_loaders_with_missing_base_return_empty(tmp_path):
    # no artifacts under an empty base dir → loaders must not raise
    assert D.load_panel(tmp_path).empty
    assert D.load_metrics(tmp_path).empty
    assert D.load_significance(tmp_path) == {}
    assert D.available_tickers(tmp_path) == []
    assert D.load_json("news/x.json", tmp_path) == {}
    assert D.load_text("report/y.md", tmp_path) == ""


def test_app_imports():
    # Streamlit app module must import cleanly (page builders are thin compositions)
    import src.dashboard.app as app  # noqa: F401

    assert set(app.PAGES.keys()) == {"Overview", "Price EDA", "News EDA", "Modeling", "Significance"}


def test_app_runs_all_pages_headless():
    """Exercise every page via streamlit's headless AppTest (covers app.py logic)."""
    import os

    if not os.path.isdir("eda_output/price"):
        pytest.skip("no artifacts for dashboard")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/dashboard/app.py", default_timeout=60)
    at.run()
    assert not at.exception, f"default page raised: {at.exception}"
    for page in ["Price EDA", "News EDA", "Modeling", "Significance", "Overview"]:
        at.sidebar.radio(key="page").set_value(page)
        at.run()
        assert not at.exception, f"page {page} raised: {at.exception}"
