"""Tests for src.modeling.features (Story 8-2)."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import features as F
from src.modeling.features import aggregate_articles, topic_flags


def test_topic_flags_matches_category():
    f = topic_flags("Lợi nhuận quý của VCB tăng mạnh")
    assert f["topic_earnings_count"] == 1
    assert f["topic_dividend_count"] == 0


def test_topic_flags_macro():
    f = topic_flags("Fed tăng lãi suất, lạm phát giảm")
    assert f["topic_macro_count"] == 1


def test_aggregate_articles_basic():
    rows = pd.DataFrame({
        "sentiment": [0.5, -0.3, 0.1],
        "topic_earnings_count": [1, 0, 1],
        "topic_dividend_count": [0, 0, 0],
        "topic_ma_count": [0, 0, 0],
        "topic_management_count": [0, 0, 0],
        "topic_regulation_count": [0, 0, 0],
        "topic_macro_count": [0, 1, 0],
        "topic_sector_count": [0, 0, 0],
    })
    a = aggregate_articles(rows)
    assert a["event_weighted_count"] == pytest.approx(0.9)  # |0.5|+|0.3|+|0.1|
    assert a["neg_news_count"] == 1 and a["pos_news_count"] == 2
    assert a["topic_earnings_count"] == 2 and a["topic_macro_count"] == 1
    assert a["abs_sentiment"] == pytest.approx(abs((0.5 - 0.3 + 0.1) / 3))


def test_aggregate_articles_empty_is_nan():
    a = aggregate_articles(pd.DataFrame())
    assert np.isnan(a["event_weighted_count"])
    assert np.isnan(a["abs_sentiment"])


def test_real_build_advanced_features_smoke():
    df = F.build_advanced_features()
    if df.empty:
        pytest.skip("no news data")
    assert "ticker" in df.columns and "date" in df.columns
    assert {"event_weighted_count", "abs_sentiment"} <= set(df.columns)
    # no-news rows must be NaN (not 0) for sentiment-like features
    nn = df[df["event_weighted_count"].isna()]
    assert nn["abs_sentiment"].isna().all()


def test_real_run_writes_parquet_smoke():
    written = F.run()
    if not written:
        pytest.skip("no news")
    assert written[0].name == "advanced_news_features.parquet"
