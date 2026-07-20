"""Tests for src.features.sentiment_scores (Story 14-1)."""

import unicodedata

import pandas as pd
import pytest

from src.features import sentiment_scores as S


def test_category_score_fraction():
    # matches "tăng mạnh", "giá tăng", "tích cực" -> 3 distinct keywords out of len(POSITIVE_KW)
    assert S.category_score("giá tăng mạnh và tích cực", S.POSITIVE_KW) == pytest.approx(3 / len(S.POSITIVE_KW))


def test_category_score_no_match_is_zero():
    assert S.category_score("hôm nay trời đẹp", S.POSITIVE_KW) == 0.0


def test_category_score_empty_keywords_is_zero():
    assert S.category_score("anything", []) == 0.0


def test_article_sentiment_scores_all_five_keys():
    scores = S.article_sentiment_scores("Cổ phiếu tăng trưởng mạnh, kỳ vọng tích cực")
    assert set(scores) == set(S.SENTIMENT_SCORE_COLS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert scores["positive_score"] > 0


def test_article_sentiment_scores_fear_negative_uncertainty():
    scores = S.article_sentiment_scores("Công ty phá sản, bán tháo, hoảng loạn, rủi ro cao")
    assert scores["fear_score"] > 0
    assert scores["negative_score"] > 0
    assert scores["uncertainty_score"] > 0
    assert scores["positive_score"] == 0.0


def test_article_sentiment_scores_nfd_unicode_still_matches():
    """Vietnamese text in NFD (decomposed) form must still match NFC keyword literals."""
    text_nfc = "Cổ phiếu tăng trưởng mạnh"
    text_nfd = unicodedata.normalize("NFD", text_nfc)
    assert S.article_sentiment_scores(text_nfd)["positive_score"] > 0


def test_article_event_flags_multiple_categories():
    flags = S.article_event_flags("Doanh nghiệp công bố cổ tức và lợi nhuận quý này")
    assert flags["event_earnings"] == 1
    assert flags["event_dividend"] == 1
    assert flags["event_ma"] == 0
    assert set(flags) == set(S.EVENT_TYPE_COLS)


def test_build_article_sentiment_empty_news_returns_empty():
    assert S.build_article_sentiment(pd.DataFrame()).empty


def test_explode_tickers_empty_returns_empty():
    assert S._explode_tickers(pd.DataFrame()).empty


def test_explode_tickers_matches_mentioned_ticker():
    df = pd.DataFrame({
        "_text": ["VCB tăng trưởng mạnh, kỳ vọng tích cực"],
        "date": [pd.Timestamp("2024-01-02")],
        **{c: [0.5] for c in S.SENTIMENT_SCORE_COLS},
        **{c: [0] for c in S.EVENT_TYPE_COLS},
    })
    out = S._explode_tickers(df)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "VCB"


def test_explode_tickers_no_ticker_mention_dropped():
    df = pd.DataFrame({
        "_text": ["Thị trường chung tăng nhẹ hôm nay"],
        "date": [pd.Timestamp("2024-01-02")],
        **{c: [0.5] for c in S.SENTIMENT_SCORE_COLS},
        **{c: [0] for c in S.EVENT_TYPE_COLS},
    })
    assert S._explode_tickers(df).empty


def test_build_daily_sentiment_features_no_sources_returns_empty(monkeypatch):
    monkeypatch.setattr(S, "discover_source_files", lambda: {})
    assert S.build_daily_sentiment_features(ticker_universe=["VCB"]).empty


# ============ real-data smoke ============
@pytest.mark.slow  # processes the full ~1.45M-row crawl_data corpus (minutes, not seconds)
def test_real_run_smoke():
    written = S.run()
    if not written:
        pytest.skip("no discovered news sources")
    assert any("sentiment_features.parquet" in str(p) for p in written)
