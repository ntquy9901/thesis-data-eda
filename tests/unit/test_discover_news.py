"""Tests for src.data.discover_news (dynamic news-file discovery + schema normalization)."""

import pandas as pd
import pytest

from src.data import discover_news as DN


def test_infer_source_name_old_schema():
    from pathlib import Path

    assert DN._infer_source_name(Path("cafef_articles.csv")) == "cafef"
    assert DN._infer_source_name(Path("ssi_articles.csv")) == "ssi"


def test_infer_source_name_new_schema():
    from pathlib import Path

    assert DN._infer_source_name(Path("news_unenriched_vnexpress_records.csv")) == "vnexpress"
    assert DN._infer_source_name(Path("vietstock_records.csv")) == "vietstock"


def test_discover_source_files_skips_denylist_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(DN, "CRAWL_DATA_ROOT", tmp_path)
    # a denylisted duplicate/backup file
    pd.DataFrame({"title": ["a"], "pub_date": ["2020-01-01"]}).to_csv(tmp_path / "data.csv", index=False)
    # a rolling snapshot (skipped by prefix)
    pd.DataFrame({"title": ["a"], "pub_date": ["2020-01-01"]}).to_csv(tmp_path / "objective_v2026-07-15.csv", index=False)
    # a legitimate old-schema source
    pd.DataFrame({"title": ["a"], "pub_date": ["2020-01-01"], "lead": ["x"], "url": ["u1"]}).to_csv(
        tmp_path / "cafef_articles.csv", index=False
    )
    # a non-news file (wrong schema)
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(tmp_path / "unrelated.csv", index=False)

    found = DN.discover_source_files()
    assert set(found.keys()) == {"cafef"}


def test_discover_source_files_finds_new_tier_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(DN, "CRAWL_DATA_ROOT", tmp_path)
    objective_dir = tmp_path / "objective"
    objective_dir.mkdir()
    pd.DataFrame({
        "title": ["a"], "publish_time": ["2026-07-08T08:57:53Z"], "source_tier": ["tier2"],
        "source": ["vnexpress"], "url": ["u1"], "raw_text": ["noi dung"],
    }).to_csv(objective_dir / "news_unenriched_vnexpress_records.csv", index=False)

    found = DN.discover_source_files()
    assert "vnexpress" in found


def test_discover_source_files_missing_root_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(DN, "CRAWL_DATA_ROOT", tmp_path / "does_not_exist")
    assert DN.discover_source_files() == {}


def test_load_source_old_schema_normalizes_columns(tmp_path):
    p = tmp_path / "cafef_articles.csv"
    pd.DataFrame({
        "title": ["Tin A"], "pub_date": ["01/02/2020"], "lead": ["Lead A"], "url": ["u1"],
    }).to_csv(p, index=False)
    df = DN.load_source("cafef", p)
    assert df.iloc[0]["source"] == "cafef"
    assert pd.notna(df.iloc[0]["pub_date"])
    assert df.iloc[0]["lead"] == "Lead A"


def test_load_source_new_schema_uses_raw_text_as_lead(tmp_path):
    p = tmp_path / "news_unenriched_vnexpress_records.csv"
    pd.DataFrame({
        "title": ["Tin B"], "publish_time": ["2026-07-08T08:57:53Z"], "source_tier": ["tier2"],
        "url": ["u2"], "raw_text": ["Noi dung day du"],
    }).to_csv(p, index=False)
    df = DN.load_source("vnexpress", p)
    assert df.iloc[0]["lead"] == "Noi dung day du"
    assert pd.notna(df.iloc[0]["pub_date"])


def test_load_all_sources_no_log(tmp_path, monkeypatch):
    monkeypatch.setattr(DN, "CRAWL_DATA_ROOT", tmp_path)
    monkeypatch.setattr(DN, "PROCESSING_LOG_PATH", tmp_path / "log.md")
    pd.DataFrame({"title": ["a"], "pub_date": ["2020-01-01"], "lead": ["x"], "url": ["u1"]}).to_csv(
        tmp_path / "cafef_articles.csv", index=False
    )
    out = DN.load_all_sources(log=False)
    assert "cafef" in out
    assert not (tmp_path / "log.md").exists()


def test_load_all_sources_writes_log(tmp_path, monkeypatch):
    monkeypatch.setattr(DN, "CRAWL_DATA_ROOT", tmp_path)
    log_path = tmp_path / "log.md"
    monkeypatch.setattr(DN, "PROCESSING_LOG_PATH", log_path)
    pd.DataFrame({"title": ["a"], "pub_date": ["2020-01-01"], "lead": ["x"], "url": ["u1"]}).to_csv(
        tmp_path / "cafef_articles.csv", index=False
    )
    DN.load_all_sources(log=True)
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "cafef" in content
    assert "cafef_articles.csv" in content


def test_log_processing_appends(tmp_path, monkeypatch):
    log_path = tmp_path / "log.md"
    monkeypatch.setattr(DN, "PROCESSING_LOG_PATH", log_path)
    DN.log_processing([{"source": "cafef", "path": "/x/cafef_articles.csv", "n_rows": 10}])
    DN.log_processing([{"source": "ssi", "path": "/x/ssi_articles.csv", "n_rows": 5}])
    content = log_path.read_text(encoding="utf-8")
    assert content.count("## ") == 2
    assert "cafef" in content and "ssi" in content


def test_discover_source_files_real_smoke():
    """Real crawl_data directory -> no crash, sane schema, includes at least one known source."""
    found = DN.discover_source_files()
    if not found:
        pytest.skip("no crawl_data available")
    assert all(hasattr(p, "exists") for p in found.values())
