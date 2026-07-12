"""Story 7-1 — Modeling dataset: HAR + news features + targets, leakage-safe split.

Builds a per-(ticker, date) modeling panel with:
- HAR features on Parkinson vol: ``har_daily`` (1d), ``har_weekly`` (5d),
  ``har_monthly`` (22d) — all TRAILING rolling means (no look-ahead).
- News features (from the sparse-news panel): news_count_1d/3d/5d,
  days_since_last_news, sentiment_mean.
- Targets: pk_t+1 / pk_t+5 / pk_t+10 (future Parkinson, leakage-safe).

Time-based split: train ≤ ``SPLIT_DATE`` (exclusive), test ≥ ``SPLIT_DATE``.
No random shuffling. Reads EDA outputs only.

Output: ``eda_output/modeling/panel.parquet`` + ``split_summary.json``.
"""

from __future__ import annotations

import json

import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs

HAR_WINDOWS = {"har_daily": 1, "har_weekly": 5, "har_monthly": 22}
NEWS_FEATURES = ["news_count_1d", "news_count_3d", "news_count_5d", "days_since_last_news", "sentiment_mean"]
PRICE_FEATURES = ["atr_14", "realized_vol_5d", "realized_vol_20d", "parkinson_vol", "log_returns"]
TARGETS = ["pk_t+1", "pk_t+5", "pk_t+10"]
SPLIT_DATE = "2025-01-01"  # train < 2025, test >= 2025 (per Phase 9 leakage policy)
MODELING_DIR = "modeling"


# ---------- pure helpers (unit-tested) ----------
def har_features(parkinson: pd.Series) -> pd.DataFrame:
    """HAR daily/weekly/monthly TRAILING rolling means of Parkinson vol.

    ``har_daily`` is the 1-day mean (== the value itself). All windows are
    trailing (past only) → no look-ahead.
    """
    out = {}
    for name, w in HAR_WINDOWS.items():
        out[name] = parkinson.rolling(w, min_periods=w).mean()
    return pd.DataFrame(out)


def time_split(panel: pd.DataFrame, split: str = SPLIT_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel into train (< split) and test (>= split) by date. No shuffle."""
    dates = pd.to_datetime(panel["date"])
    train = panel[dates < pd.Timestamp(split)].copy()
    test = panel[dates >= pd.Timestamp(split)].copy()
    return train, test


# ---------- phase runner ----------
def build_panel() -> pd.DataFrame:
    """Join price metrics + sparse news + HAR features per (ticker, date)."""
    frames = []
    for ticker in EDA_TICKERS:
        pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
        if not pq.exists():
            continue
        price = pd.read_parquet(pq)
        price["date"] = pd.to_datetime(price["date"]).dt.normalize()
        price["ticker"] = ticker

        # HAR features (trailing means of Parkinson)
        har = har_features(price["parkinson_vol"])
        price = pd.concat([price.reset_index(drop=True), har.reset_index(drop=True)], axis=1)

        keep = ["ticker", "date"] + [c for c in (PRICE_FEATURES + list(HAR_WINDOWS) + TARGETS) if c in price.columns]
        frames.append(price[keep])

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)

    news_path = EDA_OUTPUT_DIR / "news" / "sparse_news_features.parquet"
    if news_path.exists():
        news = pd.read_parquet(news_path)
        news["trading_date"] = pd.to_datetime(news["trading_date"]).dt.normalize()
        news_cols = ["ticker", "trading_date"] + [c for c in news.columns if c in NEWS_FEATURES]
        panel = panel.merge(
            news[news_cols], left_on=["ticker", "date"], right_on=["ticker", "trading_date"], how="left"
        ).drop(columns=["trading_date"], errors="ignore")

    # Advanced news features (Story 8-2): event-weighted, sentiment strength, topic flags
    adv_path = EDA_OUTPUT_DIR / "modeling" / "advanced_news_features.parquet"
    if adv_path.exists():
        adv = pd.read_parquet(adv_path)
        adv["date"] = pd.to_datetime(adv["date"]).dt.normalize()
        panel = panel.merge(adv, on=["ticker", "date"], how="left")
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Modeling feature columns = price + HAR + news (excluding targets/id/date)."""
    exclude = {"ticker", "date"} | set(TARGETS)
    return [c for c in panel.columns if c not in exclude]


def run() -> list:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / MODELING_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    if panel.empty:
        return []
    # drop rows missing any target (can't train/eval without y)
    panel = panel.dropna(subset=TARGETS).reset_index(drop=True)
    panel.to_parquet(outdir / "panel.parquet", index=False)

    train, test = time_split(panel)
    summary = {
        "split_date": SPLIT_DATE,
        "n_total": int(len(panel)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "tickers": list(panel["ticker"].unique()),
        "feature_columns": feature_columns(panel),
        "targets": TARGETS,
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())] if not train.empty else None,
        "test_date_range": [str(test["date"].min().date()), str(test["date"].max().date())] if not test.empty else None,
    }
    (outdir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return [outdir / "panel.parquet", outdir / "split_summary.json"]


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
