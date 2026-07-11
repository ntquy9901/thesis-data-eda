"""Tests for src.eda.phase04_news_eda and src.eda.phase07_sparse_news."""


import numpy as np
import pandas as pd
import pytest

from src.eda import phase04_news_eda as p4
from src.eda import phase07_sparse_news as p7
from src.eda.phase04_news_eda import (
    classify_publish_time,
    effective_trading_date,
    normalize_pub_date,
    np_searchsorted,
)
from src.eda.phase07_sparse_news import days_since_last_news, sparse_features


# ============ phase04: date normalization ============
def test_normalize_pub_date_per_source_dayfirst():
    # cafef + news_articles are ISO; ssi + vndirect are DD/MM (verified on real data)
    iso = normalize_pub_date(pd.Series(["2024-07-01"]), "cafef_articles")
    dmy = normalize_pub_date(pd.Series(["01/07/2024"]), "ssi_articles")  # dayfirst → 1 July
    assert iso.iloc[0].month == 7 and iso.iloc[0].day == 1
    assert dmy.iloc[0].month == 7 and dmy.iloc[0].day == 1  # NOT 7 January


def test_normalize_pub_date_bad_values_naT():
    s = normalize_pub_date(pd.Series(["not-a-date", None]), "ssi_articles")
    assert s.isna().all()


def test_np_searchsorted_first_ge():
    cal = np.array(["2024-01-01", "2024-01-03", "2024-01-06"], dtype="datetime64[D]")
    vals = np.array(["2024-01-01", "2024-01-02", "2024-01-10"], dtype="datetime64[D]")
    idx = np_searchsorted(cal, vals)
    assert list(idx) == [0, 1, 3]  # 01-01→0, 01-02→1 (next is 01-03), 01-10→3 (past end)


def test_effective_trading_date_rolls_forward():
    news = pd.Series(pd.to_datetime(["2024-01-01 10:00+07:00", "2024-01-02 16:00+07:00"]))
    trading = ["2024-01-01", "2024-01-03", "2024-01-05"]
    eff = effective_trading_date(news, trading)
    # news1: trading day, before 15:00 → 2024-01-01
    assert eff.iloc[0].normalize() == pd.Timestamp("2024-01-01")
    # news2: after 15:00 on 01-02 (non-trading) → roll to next trading 01-03
    assert eff.iloc[1].normalize() == pd.Timestamp("2024-01-03")


def test_effective_trading_date_propagates_nat():
    # NaT inputs must NOT be silently mapped to the last trading date
    news = pd.Series([pd.NaT, pd.to_datetime("2024-01-01 10:00+07:00")])
    eff = effective_trading_date(news, ["2024-01-01", "2024-06-01"])
    assert pd.isna(eff.iloc[0])
    assert eff.iloc[1].normalize() == pd.Timestamp("2024-01-01")


def test_classify_publish_time_buckets():
    news = pd.Series(pd.to_datetime(["2024-01-01 08:00+07:00", "2024-01-01 10:00+07:00",
                                     "2024-01-01 20:00+07:00", "2024-01-06 10:00+07:00"]))  # Sat
    pt = classify_publish_time(news)
    assert pt["session"].iloc[0] == "before_market"
    assert pt["session"].iloc[1] == "during_market"
    assert pt["session"].iloc[2] == "after_market"
    assert pt["is_weekend"].iloc[3] == True  # noqa: E712


# ============ phase07: sparse news ============
def test_days_since_last_news_basic():
    counts = pd.Series([0, 1, 0, 0, 2, 0])  # news at idx1, idx4
    dsl = days_since_last_news(counts)
    assert pd.isna(dsl.iloc[0])  # before first news
    assert dsl.iloc[1] == 0
    assert dsl.iloc[2] == 1 and dsl.iloc[3] == 2
    assert dsl.iloc[4] == 0
    assert dsl.iloc[5] == 1


def test_sparse_features_never_masks_no_news_as_zero_sentiment():
    td = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D"))
    counts = {pd.Timestamp("2024-01-02"): 2}  # news only on day 2
    sent = {pd.Timestamp("2024-01-02"): 0.5}
    df = sparse_features(counts, sent, td)
    assert list(df["news_available"]) == [0, 1, 0, 0, 0]
    assert df.loc[pd.Timestamp("2024-01-02"), "sentiment_mean"] == 0.5
    # No-news days have NaN sentiment (NOT 0) — the key EDA rule
    assert pd.isna(df.loc[pd.Timestamp("2024-01-01"), "sentiment_mean"])
    assert pd.isna(df.loc[pd.Timestamp("2024-01-03"), "sentiment_mean"])
    # rolling windows
    assert df.loc[pd.Timestamp("2024-01-04"), "news_count_3d"] == 2  # [02,03,04]
    # coverage_ratio_5d = fraction of last 5 trading days with news (rolling, min_periods=1)
    assert "coverage_ratio_5d" in df.columns
    assert df.loc[pd.Timestamp("2024-01-02"), "coverage_ratio_5d"] == 0.5  # mean of [0,1]
    assert df.loc[pd.Timestamp("2024-01-05"), "coverage_ratio_5d"] == 0.2  # 1 of 5


# ============ integration + real-data smoke ============
@pytest.fixture
def redirected(tmp_path, monkeypatch):
    news = tmp_path / "news.csv"
    news.write_text(
        "source,title,pub_date,url\n"
        "ssi,VCB tăng mạnh,2024-01-02 09:30+07:00,http://a\n"
        "cafef,FPT giảm,02/01/2024 10:00,http://b\n"
        "ssi,VCB lãi,2024-01-03 10:00+07:00,http://c\n",
        encoding="utf-8",
    )
    price = tmp_path / "VCB_ohlcv.csv"
    price.write_text("date,open,high,low,close,volume\n2024-01-01,1,1,1,1,1\n2024-01-02,1,1,1,1,1\n2024-01-03,1,1,1,1,1\n2024-01-04,1,1,1,1,1\n", encoding="utf-8")
    monkeypatch.setattr(p4, "NEWS_FILES", {"news": news})
    monkeypatch.setattr(p4, "EDA_TICKERS", ["VCB"])
    monkeypatch.setattr(p4, "PRICE_DATA_DIR", tmp_path)
    monkeypatch.setattr(p7, "EDA_TICKERS", ["VCB"])
    out = tmp_path / "eda_output"
    monkeypatch.setattr("src.eda.common.EDA_OUTPUT_DIR", out)
    return out


def test_phase04_run_writes_artifacts(redirected):
    written = p4.run_phase()
    names = {p.name for p in written}
    assert {"coverage_report.csv", "publish_time.json", "news_per_stock.csv",
            "sentiment_summary.json", "topics.json", "source_report.json"} <= names
    cov = pd.read_csv(redirected / "news" / "coverage_report.csv")
    assert (cov["metric"] == "total_articles").any()


def test_phase07_run_writes_panel(redirected):
    written = p7.run_phase()
    assert len(written) == 1 and written[0].name == "sparse_news_features.parquet"
    panel = pd.read_parquet(written[0])
    assert {"ticker", "trading_date", "news_count_1d", "news_count_5d",
            "news_available", "days_since_last_news", "sentiment_mean"} <= set(panel.columns)


# ============ real-data sample smoke (per CLAUDE.md Testing quality rules) ============
def test_real_news_loads_and_effective_date_smoke():
    from config import CRAWL_DATA_ROOT

    p = CRAWL_DATA_ROOT / "news_articles.csv"
    if not p.exists():
        pytest.skip("real news data not available")
    df = pd.read_csv(p, encoding="utf-8", nrows=200)
    dt = p4.normalize_pub_date(df["pub_date"], "news_articles")
    assert dt.notna().mean() > 0.5  # most rows parse under ISO/default
    # effective_trading_date runs without error on a tiny calendar
    eff = p4.effective_trading_date(dt, ["2024-01-01", "2024-06-01"])
    assert eff.notna().any()
