"""Tests for src.eda.phase11_news_embedding_eda (Story 11-1)."""

import numpy as np
import pandas as pd
import pytest

from src.eda import phase11_news_embedding_eda as P


def _fake_emb_frame(n=6, group="tong_hop", source="cafef", dim=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": ["VCB"] * n,
        "date": pd.date_range("2020-01-01", periods=n),
        "source": [source] * n,
        **{f"emb_{i}": rng.normal(size=n) for i in range(dim)},
    })


def test_embedding_coverage_basic():
    emb_by_group = {"tong_hop": _fake_emb_frame(), "khach_quan": pd.DataFrame()}
    cov = P._embedding_coverage(emb_by_group)
    assert set(cov["group"]) == {"tong_hop"}
    assert cov.iloc[0]["n_embedded_rows"] == 6


def test_embedding_coverage_all_empty():
    cov = P._embedding_coverage({"tong_hop": pd.DataFrame(), "khach_quan": pd.DataFrame()})
    assert cov.empty


def test_group_similarity_two_groups():
    emb_by_group = {"tong_hop": _fake_emb_frame(seed=0), "khach_quan": _fake_emb_frame(seed=1)}
    sim = P._group_similarity(emb_by_group, sample=5)
    assert sim["n_groups"] == 2
    assert "within_tong_hop" in sim and "within_khach_quan" in sim
    assert "across_groups" in sim


def test_group_similarity_empty_groups():
    sim = P._group_similarity({"tong_hop": pd.DataFrame(), "khach_quan": pd.DataFrame()})
    assert sim["n_groups"] == 0
    assert "across_groups" not in sim


def test_plot_group_scatter_empty_returns_false(tmp_path):
    ok = P._plot_group_scatter({"tong_hop": pd.DataFrame(), "khach_quan": pd.DataFrame()}, tmp_path / "x.png")
    assert ok is False
    assert not (tmp_path / "x.png").exists()


def test_plot_group_scatter_writes_file(tmp_path):
    emb_by_group = {"tong_hop": _fake_emb_frame(seed=0), "khach_quan": _fake_emb_frame(seed=1)}
    out = tmp_path / "scatter.png"
    ok = P._plot_group_scatter(emb_by_group, out)
    assert ok is True
    assert out.exists()


def test_source_stats_smoke():
    """Real crawl_data CSVs (dynamically discovered) -> no crash, sane schema. Does NOT assert
    an exact source set — new crawl files are expected to appear over time (see discover_news.py)."""
    stats = P._source_stats()
    if stats.empty:
        pytest.skip("no news sources discovered")
    assert {"source", "group", "n_articles"} <= set(stats.columns)
    assert stats["group"].isin(["khach_quan", "tong_hop", "unclassified"]).all()


def test_source_stats_tags_correct_group():
    stats = P._source_stats()
    if stats.empty:
        pytest.skip("no source CSVs")
    by_source = stats.set_index("source")["group"]
    if "cafef" in by_source.index:
        assert by_source["cafef"] == "khach_quan"
    if "ssi" in by_source.index:
        assert by_source["ssi"] == "tong_hop"


def test_source_stats_missing_source_file_skips(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(P, "discover_source_files", lambda: {"cafef": Path("cafef_articles.csv")})

    def _fake_load_source(source, path):
        raise FileNotFoundError(source)

    monkeypatch.setattr(P, "load_source", _fake_load_source)
    stats = P._source_stats()
    assert stats.empty


def test_source_stats_tags_unclassified_source(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(P, "discover_source_files", lambda: {"mysterysource": Path("x.csv")})
    monkeypatch.setattr(P, "load_source", lambda s, p: pd.DataFrame({"title": ["a"], "lead": ["b"]}))
    stats = P._source_stats()
    assert stats.iloc[0]["group"] == "unclassified"


def test_group_similarity_sample_zero_yields_none_mean_cos():
    """sample=0 -> empty sampled arrays -> _mean_cos's a/b-empty guard returns None."""
    emb_by_group = {"tong_hop": _fake_emb_frame(), "khach_quan": _fake_emb_frame(seed=1)}
    sim = P._group_similarity(emb_by_group, sample=0)
    assert sim["within_tong_hop"] is None
    assert sim["across_groups"] is None


def test_real_run_phase_smoke():
    written = P.run_phase()
    if not written:
        pytest.skip("no news data")
    names = {p.name for p in written}
    assert {"source_stats.csv", "embedding_coverage.csv", "group_similarity.json"} <= names
