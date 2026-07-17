"""Story 8-2 / 11-1 — Advanced news features (richer signal than daily counts/mean).

Per-(ticker, effective_trading_date) features aggregated from article-level data:
- ``emb_0``..``emb_{PCA_DIM-1}`` — mean-pooled PhoBERT embedding (PCA-reduced) over
  the day's news, from the "tong_hop" (consolidated) group (see ``src/features/news_embeddings.py``)
- ``topic_<category>_count`` — articles matching each EDA-Guide category
  (earnings/dividend/ma/management/regulation/macro/sector) via keyword flags

Sentiment (keyword dict) was dropped (Story 11-1): it loses information relative to
embeddings and is no longer used as a modeling feature.

NaN where a (ticker, date) has no news (consistent with the Phase-7 rule).
Output: ``eda_output/modeling/advanced_news_features.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.eda.phase04_news_eda import TOPIC_CATEGORIES, _trading_calendar
from src.features.news_embeddings import (  # noqa: F401 (re-export)
    PCA_DIM,
    build_group_embeddings,
    topic_flags,
)

EMB_FEATURES = [f"emb_{i}" for i in range(PCA_DIM)]
ADV_FEATURES = EMB_FEATURES + [
    "topic_earnings_count", "topic_dividend_count", "topic_ma_count",
    "topic_management_count", "topic_regulation_count", "topic_macro_count", "topic_sector_count",
]


def aggregate_articles(rows: pd.DataFrame) -> dict:
    """Aggregate one (ticker, date)'s article rows → advanced feature dict (mean embedding + topic counts)."""
    if rows.empty:
        return dict.fromkeys(ADV_FEATURES, np.nan)
    # PCA may have produced fewer than PCA_DIM components (small train set) -> pad with NaN
    out = {c: float(rows[c].mean()) if c in rows.columns else np.nan for c in EMB_FEATURES}
    for cat in TOPIC_CATEGORIES:
        out[f"topic_{cat}_count"] = int(rows[f"topic_{cat}_count"].sum())
    return out


# ---------- runner ----------
def _article_table() -> pd.DataFrame:
    """One row per (article x mentioned ticker) with eff_date, embedding dims, topic flags (tong_hop group)."""
    emb = build_group_embeddings("tong_hop")
    if emb.empty:
        return pd.DataFrame()
    return emb.rename(columns={"date": "eff_date"})


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
