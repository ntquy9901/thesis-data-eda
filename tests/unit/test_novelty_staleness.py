"""Tests for src.features.news_embeddings novelty/decay functions (Story 12-1, 12-3)."""

import pandas as pd
import pytest

import src.features.news_embeddings as ne


def _raw_frame(ticker_dates_vecs, dim=4):
    """ticker_dates_vecs: list of (ticker, date_str, vector)."""
    rows = []
    for ticker, date_str, vec in ticker_dates_vecs:
        rec = {"ticker": ticker, "date": pd.Timestamp(date_str), "source": "cafef"}
        rec.update({f"raw_{i}": vec[i] for i in range(dim)})
        rows.append(rec)
    return pd.DataFrame(rows)


def _mock_daily_calendar(monkeypatch, start="2019-01-01", end="2019-02-01"):
    """Isolate decayed_embedding_features from the real (file-based) trading calendar: a plain
    every-calendar-day range is deterministic and sufficient for these unit tests."""
    days = pd.date_range(start, end, freq="D")
    monkeypatch.setattr(ne, "_trading_calendar", lambda: days)


def test_compute_novelty_scores_first_article_is_maximally_novel(monkeypatch):
    """No prior articles for a ticker -> novelty = 1.0."""
    raw = _raw_frame([("VCB", "2020-01-01", [1.0, 0.0, 0.0, 0.0])])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    out = ne.compute_novelty_scores("tong_hop")
    assert out.iloc[0]["novelty"] == pytest.approx(1.0)


def test_compute_novelty_scores_repeated_article_is_stale(monkeypatch):
    """An article identical to a very recent one (same ticker, within window) -> novelty ~ 0."""
    raw = _raw_frame([
        ("VCB", "2020-01-01", [1.0, 0.0, 0.0, 0.0]),
        ("VCB", "2020-01-02", [1.0, 0.0, 0.0, 0.0]),  # identical vector, 1 day later
    ])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    out = ne.compute_novelty_scores("tong_hop", window_days=5)
    second = out[out["date"] == pd.Timestamp("2020-01-02")].iloc[0]
    assert second["novelty"] == pytest.approx(0.0, abs=1e-6)


def test_compute_novelty_scores_same_day_duplicate_is_stale(monkeypatch):
    """Two identical articles published the SAME day must also detect each other as a rehash
    (not just distinct-day duplicates) -> novelty ~ 0 for both."""
    raw = _raw_frame([
        ("VCB", "2020-01-01", [1.0, 0.0, 0.0, 0.0]),
        ("VCB", "2020-01-01", [1.0, 0.0, 0.0, 0.0]),  # identical vector, SAME day
    ])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    out = ne.compute_novelty_scores("tong_hop", window_days=5)
    assert (out["novelty"].abs() < 1e-6).all()


def test_compute_novelty_scores_outside_window_is_novel(monkeypatch):
    """An identical article outside the window (too old) does not count -> novelty = 1.0."""
    raw = _raw_frame([
        ("VCB", "2020-01-01", [1.0, 0.0, 0.0, 0.0]),
        ("VCB", "2020-01-20", [1.0, 0.0, 0.0, 0.0]),  # 19 days later, window=5
    ])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    out = ne.compute_novelty_scores("tong_hop", window_days=5)
    second = out[out["date"] == pd.Timestamp("2020-01-20")].iloc[0]
    assert second["novelty"] == pytest.approx(1.0)


def test_compute_novelty_scores_empty_raw_returns_empty(monkeypatch):
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: pd.DataFrame())
    assert ne.compute_novelty_scores("tong_hop").empty


def test_novelty_daily_aggregates_mean(monkeypatch):
    raw = _raw_frame([
        ("VCB", "2020-01-01", [1.0, 0.0, 0.0, 0.0]),
        ("VCB", "2020-01-01", [0.0, 1.0, 0.0, 0.0]),  # 2 articles, same day
    ])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    daily = ne.novelty_daily("tong_hop")
    assert len(daily) == 1
    assert daily.iloc[0]["novelty_mean"] == pytest.approx(1.0)  # both are first articles


def test_novelty_daily_empty(monkeypatch):
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: pd.DataFrame())
    assert ne.novelty_daily("tong_hop").empty


# ---------- decayed_embedding_features (Story 12-3) ----------
def test_decayed_embedding_features_weights_recent_more(monkeypatch):
    """A recent article should dominate the decayed mean more than an old one (checked pre-PCA
    by bypassing _reduce, so the raw weighted-sum math is verified directly)."""
    raw = _raw_frame([
        ("VCB", "2019-01-01", [1.0, 0.0, 0.0, 0.0]),
        ("VCB", "2019-01-10", [0.0, 1.0, 0.0, 0.0]),  # 9 days later, halflife=5 -> recent dominates
    ])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    monkeypatch.setattr(ne, "_reduce", lambda df, dim=None: df)  # bypass PCA to inspect raw_* directly
    _mock_daily_calendar(monkeypatch)
    out = ne.decayed_embedding_features("tong_hop", halflife_days=5, lookback_days=20, dim=4)
    last_day = out[out["date"] == pd.Timestamp("2019-01-10")].iloc[0]
    # weight(age=9) = 0.5**(9/5) ~ 0.287; weight(age=0) = 1.0; normalized recent weight ~ 0.777
    assert last_day["raw_1"] > last_day["raw_0"]
    assert last_day["raw_1"] == pytest.approx(0.7767, abs=1e-3)


def test_decayed_embedding_features_covers_non_publication_trading_days(monkeypatch):
    """The core Story 12-3 fix: a trading day with NO same-day article, but with a recent
    article still inside the lookback window, must still get a real (non-NaN) decayed value —
    not be silently absent (that was the bug found in code-review round 1: the old
    implementation only emitted rows for a ticker's OWN article-publish dates)."""
    raw = _raw_frame([
        ("VCB", "2019-01-01", [1.0, 0.0, 0.0, 0.0]),
        ("VCB", "2019-01-10", [0.0, 1.0, 0.0, 0.0]),
    ])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    monkeypatch.setattr(ne, "_reduce", lambda df, dim=None: df)
    _mock_daily_calendar(monkeypatch)
    out = ne.decayed_embedding_features("tong_hop", halflife_days=5, lookback_days=20, dim=4)
    # 2019-01-05 has NO article of its own, but 2019-01-01's article is within the 20-day
    # lookback -> must still appear with a real (non-NaN) decayed value.
    mid = out[out["date"] == pd.Timestamp("2019-01-05")]
    assert len(mid) == 1
    assert mid.iloc[0][[c for c in out.columns if c.startswith("raw_")]].notna().all()
    # every calendar day in [2019-01-01, 2019-01-10] should be covered (10 days)
    assert len(out) == 10


def test_decayed_embedding_features_halflife_zero_raises(monkeypatch):
    raw = _raw_frame([("VCB", "2019-01-01", [1.0, 0.0, 0.0, 0.0])])
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: raw)
    with pytest.raises(ValueError, match="halflife_days"):
        ne.decayed_embedding_features("tong_hop", halflife_days=0)


def test_decayed_embedding_features_empty_raw(monkeypatch):
    monkeypatch.setattr(ne, "load_or_build_raw", lambda group: pd.DataFrame())
    assert ne.decayed_embedding_features("tong_hop").empty
