"""Story 8-2 — Advanced news features (richer signal than daily counts/mean).

Per-(ticker, effective_trading_date) features aggregated from article-level data:
- ``event_weighted_count`` — Σ |sentiment| over the day's news (strong news counts more)
- ``abs_sentiment`` — |mean sentiment| (sentiment strength regardless of sign)
- ``sentiment_std`` — dispersion of sentiment across the day's articles
- ``neg_news_count`` / ``pos_news_count`` — counts by sign
- ``topic_<category>_count`` — articles matching each EDA-Guide category
  (earnings/dividend/ma/management/regulation/macro/sector) via keyword flags

NaN where a (ticker, date) has no news (consistent with the Phase-7 rule).
Output: ``eda_output/modeling/advanced_news_features.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.eda.phase04_news_eda import (
    TOPIC_CATEGORIES,
    _load_news_frame,
    _trading_calendar,
    effective_trading_date,
)

ADV_FEATURES = [
    "event_weighted_count", "abs_sentiment", "sentiment_std",
    "neg_news_count", "pos_news_count",
    "topic_earnings_count", "topic_dividend_count", "topic_ma_count",
    "topic_management_count", "topic_regulation_count", "topic_macro_count", "topic_sector_count",
]


# ---------- pure helpers (unit-tested) ----------
def topic_flags(text: str) -> dict[str, int]:
    """1/0 flag per EDA-Guide category for one article (keyword match)."""
    t = str(text).lower()
    return {f"topic_{cat}_count": int(any(kw in t for kw in kws)) for cat, kws in TOPIC_CATEGORIES.items()}


def aggregate_articles(rows: pd.DataFrame) -> dict:
    """Aggregate one (ticker, date)'s article rows → advanced feature dict."""
    if rows.empty:
        return dict.fromkeys(ADV_FEATURES, np.nan)
    sent = rows["sentiment"]
    n = len(rows)
    out = {
        "event_weighted_count": float(sent.abs().sum()),
        "abs_sentiment": float(sent.mean().item() if pd.notna(sent.mean()) else np.nan),
        "sentiment_std": float(sent.std(ddof=0)) if n > 1 else 0.0,
        "neg_news_count": int((sent < 0).sum()),
        "pos_news_count": int((sent > 0).sum()),
    }
    out["abs_sentiment"] = abs(out["abs_sentiment"]) if pd.notna(out["abs_sentiment"]) else np.nan
    for cat in TOPIC_CATEGORIES:
        out[f"topic_{cat}_count"] = int(rows[f"topic_{cat}_count"].sum())
    return out


# ---------- runner ----------
def _article_table() -> pd.DataFrame:
    """One row per (article × mentioned ticker) with eff_date, sentiment, topic flags."""
    news = _load_news_frame()
    if news.empty:
        return pd.DataFrame()
    trading = _trading_calendar()
    news["eff_date"] = effective_trading_date(news["pub_date_dt"], trading)

    try:
        from src.sprint1.task1_3_vietnamese_nlp import VietnameseNLPProcessor

        proc = VietnameseNLPProcessor()
        tickers_per_doc = proc.extract_stock_tickers(news["_text"].tolist())
        sentiments = [r.get("score", 0.0) for r in proc.calculate_sentiment_vietnamese(news["_text"].tolist())]
    except Exception:  # regex + zero-sentiment fallback
        import re

        from config import VN30_TICKERS

        pat = re.compile(r"\b(" + "|".join(VN30_TICKERS) + r")\b")
        tickers_per_doc = [pat.findall(t) for t in news["_text"].tolist()]
        sentiments = [0.0] * len(news)

    flags = news["_text"].apply(topic_flags).tolist()
    records = []
    for i, tk in enumerate(tickers_per_doc):
        for t in set(tk):
            rec = {"ticker": t, "eff_date": news.iloc[i]["eff_date"], "sentiment": sentiments[i]}
            rec.update(flags[i])
            records.append(rec)
    return pd.DataFrame(records)


def build_advanced_features() -> pd.DataFrame:
    """Build the advanced-news panel (ticker, date) × ADV_FEATURES."""
    arts = _article_table()
    if arts.empty:
        return pd.DataFrame()
    arts = arts.dropna(subset=["eff_date"])
    td = pd.DatetimeIndex(pd.to_datetime(pd.Series(_trading_calendar())).dt.normalize().sort_values())

    # aggregate article rows per (ticker, normalized eff_date) — manual loop (robust)
    records = []
    eff_norm = arts["eff_date"].dt.normalize()
    for (ticker, date), rows in arts.groupby([arts["ticker"], eff_norm]):
        rec = {"ticker": ticker, "date": date}
        rec.update(aggregate_articles(rows.drop(columns=["ticker", "eff_date"], errors="ignore")))
        records.append(rec)
    agg = pd.DataFrame(records)

    out_frames = []
    for ticker in EDA_TICKERS:
        sub = agg[agg["ticker"] == ticker]
        if sub.empty:
            df = pd.DataFrame(np.nan, index=td, columns=ADV_FEATURES)
        else:
            sub = sub.copy()
            sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
            df = sub.set_index("date").reindex(td)[ADV_FEATURES]
        df.insert(0, "ticker", ticker)
        df.index.name = "date"
        out_frames.append(df.reset_index())
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def run() -> list:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    df = build_advanced_features()
    if df.empty:
        return []
    out = outdir / "advanced_news_features.parquet"
    df.to_parquet(out, index=False)
    return [out]


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
