"""Phase 7 — Sparse News Analysis (per EDA Guide).

Builds a ``(ticker, trading_date)`` panel of news-availability features that
respect the EDA-Guide rule: never mask "no news" as sentiment=0.

Features per (ticker, trading_date):
- ``news_count_1d`` / ``news_count_3d`` / ``news_count_5d`` — trailing counts on
  the trading-calendar index (position-based rolling = calendar-correct).
- ``days_since_last_news`` — trading-day distance to the most recent news.
- ``news_available`` — 1 if ``news_count_1d`` > 0 else 0.
- ``sentiment_mean`` — mean sentiment of news on that date; **NaN when
  news_available=0** (no news ≠ neutral news).

Output: ``eda_output/news/sparse_news_features.parquet``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import ensure_output_dirs, phase_output_dir
from src.eda.phase04_news_eda import (  # noqa: F401 (reuse)
    _load_news_frame,
    _sentiment_summary,
    _trading_calendar,
    effective_trading_date,
)

ROLLING_WINDOWS = (1, 3, 5)


# ---------- pure helpers (unit-tested) ----------
def days_since_last_news(counts: pd.Series) -> pd.Series:
    """Trading-day distance to the most recent nonzero count. 0 on a news day.

    Leading days before any news are NaN. Pure.
    """
    arr = counts.to_numpy()
    out = np.full(len(arr), np.nan)
    last = -1
    for i, c in enumerate(arr):
        if c > 0:
            last = i
            out[i] = 0
        elif last >= 0:
            out[i] = i - last
    return pd.Series(out, index=counts.index)


def sparse_features(counts_by_date: dict, sentiment_by_date: dict, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the sparse-news feature frame for one ticker.

    ``counts_by_date`` / ``sentiment_by_date`` map normalized dates → values.
    ``trading_dates`` is the sorted trading calendar (DatetimeIndex).
    """
    counts = pd.Series(0, index=trading_dates, dtype=int)
    for d, c in counts_by_date.items():
        d = pd.Timestamp(d).normalize()
        if d in counts.index:
            counts.loc[d] = c

    df = pd.DataFrame(index=trading_dates)
    df["news_count_1d"] = counts
    df["news_count_3d"] = counts.rolling(3, min_periods=1).sum().astype(int)
    df["news_count_5d"] = counts.rolling(5, min_periods=1).sum().astype(int)
    df["news_available"] = (counts > 0).astype(int)
    df["coverage_ratio_5d"] = (counts > 0).astype(int).rolling(5, min_periods=1).mean()
    df["days_since_last_news"] = days_since_last_news(counts)

    sent = pd.Series(np.nan, index=trading_dates, dtype=float)
    for d, v in sentiment_by_date.items():
        d = pd.Timestamp(d).normalize()
        if d in sent.index:
            sent.loc[d] = v
    df["sentiment_mean"] = sent.where(df["news_available"] == 1)  # NaN where no news
    return df


# ---------- phase runner ----------
def _ticker_news(news: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {ticker: news rows mentioning it} with effective_trading_date set."""
    if news.empty:
        return {}
    try:
        from src.sprint1.task1_3_vietnamese_nlp import VietnameseNLPProcessor

        proc = VietnameseNLPProcessor()
        tickers_per_doc = proc.extract_stock_tickers(news["_text"].tolist())
        sentiments = proc.calculate_sentiment_vietnamese(news["_text"].tolist())
    except Exception:
        import re

        from config import VN30_TICKERS

        pat = re.compile(r"\b(" + "|".join(VN30_TICKERS) + r")\b")
        tickers_per_doc = [pat.findall(t) for t in news["_text"].tolist()]
        sentiments = [{"score": 0.0}] * len(news)

    out: dict[str, pd.DataFrame] = {}
    for i, tk in enumerate(tickers_per_doc):
        for t in set(tk):
            out.setdefault(t, []).append(
                {"eff_date": news.iloc[i]["eff_date"], "sentiment": sentiments[i].get("score", 0.0)}
            )
    return {k: pd.DataFrame(v) for k, v in out.items() if v}


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("news")
    news = _load_news_frame()
    if news.empty:
        return []
    trading = _trading_calendar()
    td_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(trading)).dt.normalize().sort_values())
    news["eff_date"] = effective_trading_date(news["pub_date_dt"], trading)

    per_ticker = _ticker_news(news)
    panels = []
    for ticker in EDA_TICKERS:
        rows = per_ticker.get(ticker)
        if rows is None or rows.empty:
            counts_by_date, sent_by_date = {}, {}
        else:
            counts_by_date = rows.groupby(rows["eff_date"].dt.normalize()).size().to_dict()
            sent_by_date = (
                rows.dropna(subset=["sentiment"])
                .groupby(rows["eff_date"].dt.normalize())["sentiment"]
                .mean()
                .to_dict()
            )
        df = sparse_features(counts_by_date, sent_by_date, td_index)
        df.insert(0, "ticker", ticker)
        df.index.name = "trading_date"
        panels.append(df.reset_index())

    if not panels:
        return []
    panel = pd.concat(panels, ignore_index=True)
    out = outdir / "sparse_news_features.parquet"
    panel.to_parquet(out, index=False)
    return [out]


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
