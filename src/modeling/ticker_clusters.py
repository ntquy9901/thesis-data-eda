"""Story 16-3 — Ticker clustering by news-sensitivity + entity embeddings.

Clusters tickers into sensitive/neutral/insensitive groups based on per-ticker ΔR²
from news features (from significance.json). Also provides entity embedding features
for per-ticker modeling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.modeling.dataset import SPLIT_DATE, TARGETS, build_panel, time_split
from src.modeling.features import ADV_FEATURES_DUAL

SENSITIVITY_TARGET = "pk_t+10"  # longest horizon with signal (from Epic 14 findings)
CLUSTER_NAMES = {1: "sensitive", 0: "neutral", -1: "insensitive"}


def compute_news_sensitivity(panel: pd.DataFrame | None = None) -> dict[str, float]:
    """Per-ticker ΔR² from Ridge price+news_adv_dual vs price at SENSITIVITY_TARGET.

    Returns dict: {ticker: delta_r2}.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if panel is None:
        panel = build_panel()
    if panel.empty:
        return {}

    price_feats = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
    news_feats = ADV_FEATURES_DUAL
    basic_news = ["news_count_1d", "news_count_3d", "news_count_5d", "days_since_last_news", "sentiment_mean"]
    all_feats = price_feats + basic_news + news_feats

    df = panel.dropna(subset=[SENSITIVITY_TARGET]).copy()
    train, test = time_split(df, SPLIT_DATE)
    target = SENSITIVITY_TARGET
    delta_r2: dict[str, float] = {}

    for ticker in EDA_TICKERS:
        sub_train = train[train["ticker"] == ticker]
        sub_test = test[test["ticker"] == ticker]
        if len(sub_train) < 20 or len(sub_test) < 5:
            delta_r2[ticker] = np.nan
            continue

        pipe_price = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe_news = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

        feats_price = [c for c in price_feats if c in sub_train.columns]
        feats_news = [c for c in all_feats if c in sub_train.columns]

        pipe_price.fit(sub_train[feats_price], sub_train[target])
        pipe_news.fit(sub_train[feats_news], sub_train[target])

        r2_price = pipe_price.score(sub_test[feats_price], sub_test[target])
        r2_news = pipe_news.score(sub_test[feats_news], sub_test[target])
        delta_r2[ticker] = round(float(r2_news - r2_price), 6)

    return delta_r2


def cluster_tickers(delta_r2: dict[str, float]) -> dict:
    """Cluster tickers into sensitive (ΔR² > 0.001), neutral, insensitive (ΔR² < -0.001)."""
    clusters: dict[str, list[str]] = {"sensitive": [], "neutral": [], "insensitive": []}
    per_ticker: dict[str, dict] = {}
    for ticker, d in sorted(delta_r2.items()):
        if np.isnan(d):
            per_ticker[ticker] = {"delta_r2": None, "cluster": "unknown", "cluster_code": None}
            continue
        if d > 0.001:
            clusters["sensitive"].append(ticker)
            label = "sensitive"
        elif d < -0.001:
            clusters["insensitive"].append(ticker)
            label = "insensitive"
        else:
            clusters["neutral"].append(ticker)
            label = "neutral"
        per_ticker[ticker] = {"delta_r2": d, "cluster": label, "cluster_code": 1 if label == "sensitive" else (-1 if label == "insensitive" else 0)}
    return {"clusters": clusters, "per_ticker": per_ticker, "n_sensitive": len(clusters["sensitive"]),
            "n_neutral": len(clusters["neutral"]), "n_insensitive": len(clusters["insensitive"])}


def add_ticker_entity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker one-hot + cluster membership as features to the panel."""
    if panel.empty:
        return panel
    df = panel.copy()
    ticker_dummies = pd.get_dummies(df["ticker"], prefix="entity")
    df = pd.concat([df, ticker_dummies], axis=1)
    return df


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    delta = compute_news_sensitivity()
    if not delta:
        return written

    result = cluster_tickers(delta)
    outpath = outdir / "ticker_clusters.json"
    outpath.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(outpath)

    lines = ["# Ticker Clustering by News Sensitivity\n"]
    lines.append(f"\n## Summary\n")
    lines.append(f"- Sensitive (ΔR² > 0.001): {result['n_sensitive']} tickers")
    lines.append(f"- Neutral: {result['n_neutral']} tickers")
    lines.append(f"- Insensitive (ΔR² < -0.001): {result['n_insensitive']} tickers")
    lines.append(f"\n## Sensitive tickers\n")
    for t in result["clusters"]["sensitive"]:
        lines.append(f"- {t}: ΔR²={delta[t]:+.6f}")
    lines.append(f"\n## Per-ticker ΔR²\n")
    for ticker, info in sorted(result["per_ticker"].items()):
        lines.append(f"- {ticker}: {info['delta_r2']:+.6f} → {info['cluster']}")

    rep = outdir / "ticker_clusters_report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    written.append(rep)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
