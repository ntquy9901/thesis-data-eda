"""Integration tests for phase runners (monkeypatched paths → tmp CSVs).

These cover the I/O runner functions that unit tests of pure helpers miss,
pushing diff-coverage of src/eda past the 80% gate.
"""

import json

import pandas as pd
import pytest

from src.eda import common
from src.eda import phase01_profiling as p1
from src.eda import phase02_quality as p2

NEWS_CSV = "source,title,pub_date,url\ncafef,Hello,2024-01-01,http://a\ncafef,Hello,2024-01-02,http://a\n"
PRICE_CSV = "date,open,high,low,close,volume\n2024-01-01,10,11,9,10,100\n2024-01-02,10,8,9,10,-5\n"
DXY_CSV = "date,dxy,source,collected_at\n2024-01-01,104.0,fx,2024-01-01\n"


@pytest.fixture
def redirected(tmp_path, monkeypatch):
    """Point NEWS/MACRO/price/output paths at tmp_path with tiny CSVs."""
    news = tmp_path / "news.csv"
    news.write_text(NEWS_CSV, encoding="utf-8")
    price = tmp_path / "VCB_ohlcv.csv"
    price.write_text(PRICE_CSV, encoding="utf-8")
    dxy = tmp_path / "dxy.csv"
    dxy.write_text(DXY_CSV, encoding="utf-8")

    monkeypatch.setattr(p1, "NEWS_FILES", {"news": news})
    monkeypatch.setattr(p1, "MACRO_FILES", {"dxy": dxy})
    monkeypatch.setattr(p1, "EDA_TICKERS", ["VCB"])
    monkeypatch.setattr(p1, "PRICE_DATA_DIR", tmp_path)
    monkeypatch.setattr(p2, "NEWS_FILES", {"news": news})
    monkeypatch.setattr(p2, "EDA_TICKERS", ["VCB"])
    monkeypatch.setattr(p2, "PRICE_DATA_DIR", tmp_path)
    out_root = tmp_path / "eda_output"
    monkeypatch.setattr(common, "EDA_OUTPUT_DIR", out_root)
    monkeypatch.setattr(p1, "EDA_OUTPUT_DIR", out_root, raising=False)
    return out_root


def test_profile_table_reads_all_sources(redirected):
    df = p1.profile_table()
    tables = set(df["table"])
    assert {"news", "VCB_price", "dxy"} == tables
    vcb = df[df["table"] == "VCB_price"].iloc[0]
    assert vcb["row_count"] == 2
    assert vcb["primary_key"] == "date"


def test_phase01_run_phase_writes_csv(redirected):
    out = p1.run_phase()
    assert out.exists()
    written = pd.read_csv(out)
    assert "table" in written.columns and len(written) == 3


def test_missingness_report_covers_news_and_price(redirected):
    rep = p2.missingness_report()
    # news has no missing; price has no missing → all pct 0, but rows exist
    assert set(rep["table"]) == {"news", "VCB_price"}
    assert (rep["pct"] == 0.0).all()


def test_duplicate_report_detects_dup_url(redirected):
    rep = p2.duplicate_report()
    assert rep["news_by_url"] == 1  # http://a appears twice
    assert rep["news_by_title"] == 1
    assert rep["VCB_price_by_date"] == 0


def test_invalid_values_report_flags_negative_volume(redirected):
    rep = p2.invalid_values_report()
    assert rep["VCB"]["negative_volume"] == 1


def test_by_stock_and_date(redirected):
    by_stock, by_date = p2.by_stock_and_date()
    assert len(by_stock) == 1 and by_stock.iloc[0]["ticker"] == "VCB"
    assert by_date.iloc[0]["ticker"] == "VCB"


def test_phase02_run_phase_writes_five_artifacts(redirected):
    paths = p2.run_phase()
    names = sorted(p.name for p in paths)
    assert names == [
        "duplicate_report.json",
        "invalid_values.json",
        "missingness_by_date.csv",
        "missingness_by_stock.csv",
        "missingness_report.csv",
    ]
    # JSON is valid
    assert isinstance(json.loads((redirected / "quality" / "duplicate_report.json").read_text()), dict)
