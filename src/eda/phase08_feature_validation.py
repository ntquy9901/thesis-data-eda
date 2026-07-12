"""Phase 8 — Feature Engineering Validation (per EDA Guide).

Audit the engineered feature matrix before modeling: missingness, (near-)zero
variance, redundant/duplicate columns, highly-collinear groups, and train/test
distribution drift. Recommend removing constant / duplicate / collinear features.

Outputs (under ``eda_output/feature_engineering/``):
- ``feature_report.csv`` — per-feature stats
- ``collinearity.json``, ``drop_recommendations.json``

The feature matrix joins price metrics + sparse-news features on (ticker, date).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import ensure_output_dirs, phase_output_dir

FEATURE_COLS = [
    "returns", "log_returns", "atr_14", "realized_vol_5d", "realized_vol_20d",
    "parkinson_vol", "news_count_1d", "news_count_3d", "news_count_5d",
    "coverage_ratio_5d", "days_since_last_news", "sentiment_mean",
]
TARGETS = ["pk_t+1", "pk_t+5", "pk_t+10", "rv_t+1", "rv_t+5", "rv_t+10"]
NZV_THRESHOLD = 1e-8
COLLINEAR_THRESHOLD = 0.9
TRAIN_TEST_SPLIT = "2025-01-01"  # ≤2024 train, ≥2025 test (time-based, per Phase 9 policy)


# ---------- pure helpers (unit-tested) ----------
def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing %."""
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["column", "pct_missing"])
    return pd.DataFrame(
        [{"column": c, "pct_missing": round(float(df[c].isna().mean() * 100), 2)} for c in df.columns]
    )


def near_zero_variance(df: pd.DataFrame, threshold: float = NZV_THRESHOLD) -> list[str]:
    """Numeric columns whose variance ≈ 0 (constant / near-constant)."""
    out = []
    for c in df.select_dtypes(include=[np.number]).columns:
        s = df[c].dropna()
        if len(s) > 0 and float(s.var()) <= threshold:
            out.append(c)
    return out


def duplicate_columns(df: pd.DataFrame) -> list[list[str]]:
    """Groups of columns that are exact duplicates (same values)."""
    groups, seen = [], set()
    cols = [c for c in df.columns if df[c].notna().any()]
    for i, c1 in enumerate(cols):
        if c1 in seen:
            continue
        dup = [c2 for c2 in cols[i + 1 :] if df[c1].equals(df[c2])]
        if dup:
            groups.append([c1] + dup)
            seen.update(dup)
    return groups


def collinear_groups(df: pd.DataFrame, threshold: float = COLLINEAR_THRESHOLD) -> list[list[str]]:
    """Clusters of numeric columns with pairwise |corr| > threshold."""
    num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return []
    corr = num.corr().abs()
    visited, groups = set(), []
    for c in corr.columns:
        if c in visited:
            continue
        peers = [o for o in corr.columns if o != c and corr.loc[c, o] > threshold]
        if peers:
            groups.append([c] + peers)
            visited.update(peers)
    return groups


def train_test_drift(df: pd.DataFrame, date_col: str, split: str) -> pd.DataFrame:
    """KS drift (stat + p) per numeric column between train (<split) and test (>=split)."""
    from scipy.stats import ks_2samp

    if date_col not in df.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(df[date_col], errors="coerce")
    train_mask, test_mask = dates < pd.Timestamp(split), dates >= pd.Timestamp(split)
    rows = []
    for c in df.select_dtypes(include=[np.number]).columns:
        a, b = df.loc[train_mask, c].dropna(), df.loc[test_mask, c].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        stat, p = ks_2samp(a, b)
        rows.append({"column": c, "ks_stat": round(float(stat), 4), "ks_p": round(float(p), 4),
                     "drift": bool(p < 0.05)})
    return pd.DataFrame(rows)


# ---------- phase runner ----------
def _build_feature_matrix() -> pd.DataFrame:
    from src.eda.common import EDA_OUTPUT_DIR

    frames = []
    for ticker in EDA_TICKERS:
        pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
        if not pq.exists():
            continue
        price = pd.read_parquet(pq)
        price["date"] = pd.to_datetime(price["date"]).dt.normalize()
        price["ticker"] = ticker
        keep = ["ticker", "date"] + [c for c in (FEATURE_COLS + TARGETS) if c in price.columns]
        frames.append(price[keep])
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)

    news_path = EDA_OUTPUT_DIR / "news" / "sparse_news_features.parquet"
    if news_path.exists():
        news = pd.read_parquet(news_path)
        news["trading_date"] = pd.to_datetime(news["trading_date"]).dt.normalize()
        news_cols = ["ticker", "trading_date"] + [c for c in news.columns if c in FEATURE_COLS]
        panel = panel.merge(
            news[news_cols], left_on=["ticker", "date"], right_on=["ticker", "trading_date"], how="left"
        ).drop(columns=["trading_date"], errors="ignore")
    return panel


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("feature_engineering")
    fm = _build_feature_matrix()
    if fm.empty:
        return []
    written = []

    feat_only = fm[[c for c in FEATURE_COLS if c in fm.columns]]
    miss = missingness(feat_only)
    rep = miss.copy()

    nzv = near_zero_variance(feat_only)
    dups = duplicate_columns(feat_only)
    coll = collinear_groups(feat_only)

    drift = train_test_drift(fm, "date", TRAIN_TEST_SPLIT)
    if not drift.empty:
        drift_map = drift.set_index("column")["drift"].to_dict()
        rep["drift"] = rep["column"].map(lambda c: drift_map.get(c, None))
    rep["near_zero_variance"] = rep["column"].isin(nzv)

    rep_path = outdir / "feature_report.csv"
    rep.to_csv(rep_path, index=False, encoding="utf-8")
    written.append(rep_path)

    (outdir / "collinearity.json").write_text(json.dumps(coll, indent=2), encoding="utf-8")
    written.append(outdir / "collinearity.json")

    # drop recommendations: drop all-but-one per collinear group + dup groups + nzv
    to_drop = set(nzv)
    for grp in dups:
        to_drop.update(grp[1:])  # keep first
    for grp in coll:
        to_drop.update(grp[1:])  # keep first of each collinear group
    rec = {
        "drop_constant_or_near_zero": nzv,
        "drop_duplicate_features": [grp[1:] for grp in dups],
        "drop_collinear_keep_first": [grp[1:] for grp in coll],
        "total_drop_recommendation": sorted(to_drop),
    }
    (outdir / "drop_recommendations.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    written.append(outdir / "drop_recommendations.json")
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
