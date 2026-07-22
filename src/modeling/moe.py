"""Story 16-4 — Mixture-of-Experts (gated model).

2 Ridge experts (price-only, price+news_adv_dual) with per-ticker
deterministic gating from ticker_clusters.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.modeling.dataset import SPLIT_DATE, TARGETS, build_panel, time_split
from src.modeling.features import ADV_FEATURES_DUAL

CLUSTER_WEIGHTS: dict[str, float] = {"sensitive": 1.0, "neutral": 0.5, "insensitive": 0.0}


class SimpleMoE:
    def __init__(self, clusters_path: str | Path | None = None, alpha: float = 1.0):
        self.alpha = alpha
        self.price_only_model: Pipeline | None = None
        self.price_news_model: Pipeline | None = None
        self.per_ticker_weights: dict[str, np.ndarray] = {}
        self.price_feats: list[str] = []
        self.news_feats: list[str] = []
        self._load_clusters(clusters_path)

    def _load_clusters(self, path: str | Path | None) -> None:
        if path is None:
            path = EDA_OUTPUT_DIR / "modeling" / "ticker_clusters.json"
        path = Path(path)
        if not path.exists():
            self.per_ticker_weights = {t: np.array([1.0, 0.0]) for t in EDA_TICKERS}
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        per_ticker = data["per_ticker"]
        self.per_ticker_weights = {}
        for ticker in EDA_TICKERS:
            info = per_ticker.get(ticker, {})
            cluster = info.get("cluster", "insensitive")
            w_news = CLUSTER_WEIGHTS.get(cluster, 0.0)
            self.per_ticker_weights[ticker] = np.array([1.0, w_news])

    def _build_feature_lists(self, panel: pd.DataFrame) -> None:
        price_feats_base = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
        basic_news = ["news_count_1d", "news_count_3d", "news_count_5d", "days_since_last_news", "sentiment_mean"]
        self.price_feats = [c for c in price_feats_base if c in panel.columns]
        self.news_feats = list(dict.fromkeys(c for c in self.price_feats + basic_news + ADV_FEATURES_DUAL if c in panel.columns))

    def fit(self, panel: pd.DataFrame, target: str) -> None:
        df = panel.dropna(subset=[target]).copy()
        self._build_feature_lists(df)
        train, _ = time_split(df, SPLIT_DATE)

        price_only = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=self.alpha)),
        ])
        price_news = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=self.alpha)),
        ])

        price_only.fit(train[self.price_feats], train[target])
        price_news.fit(train[self.news_feats], train[target])

        self.price_only_model = price_only
        self.price_news_model = price_news

    def predict_ticker(self, ticker: str, X_price: pd.DataFrame, X_news: pd.DataFrame) -> np.ndarray:
        if self.price_only_model is None or self.price_news_model is None:
            raise RuntimeError("fit() must be called before predict")
        w = self.per_ticker_weights.get(ticker, np.array([1.0, 0.0]))
        p1 = self.price_only_model.predict(X_price)
        p2 = self.price_news_model.predict(X_news)
        denom = w[0] + w[1]
        if denom == 0:
            return p1
        return (w[0] * p1 + w[1] * p2) / denom

    def evaluate(self, panel: pd.DataFrame, target: str) -> dict:
        df = panel.dropna(subset=[target]).copy()
        if self.price_only_model is None:
            self.fit(df, target)
        train, test = time_split(df, SPLIT_DATE)
        results: dict = {}

        for ticker in sorted(df["ticker"].unique()):
            sub = test[test["ticker"] == ticker]
            if len(sub) < 5:
                continue
            y_true = sub[target].values
            y_pred = self.predict_ticker(ticker, sub[self.price_feats], sub[self.news_feats])
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2 = float(r2_score(y_true, y_pred))

            p_only = self.price_only_model.predict(sub[self.price_feats])
            p_r2 = float(r2_score(y_true, p_only))
            delta = r2 - p_r2
            w = self.per_ticker_weights.get(ticker, np.array([1.0, 0.0]))
            cluster_map = {1.0: "sensitive", 0.5: "neutral", 0.0: "insensitive"}
            cluster = cluster_map.get(w[1], "insensitive")

            results[ticker] = {
                "cluster": cluster,
                "w_price": float(w[0]),
                "w_news": float(w[1]),
                "MoE_R2": round(r2, 6),
                "baseline_price_R2": round(p_r2, 6),
                "delta_R2_vs_baseline": round(delta, 6),
                "MoE_RMSE": round(rmse, 6),
            }

        if not results:
            return {"per_ticker": {}, "aggregate_MoE_R2": None, "aggregate_baseline_R2": None, "n_tickers": 0}
        agg_r2 = np.mean([v["MoE_R2"] for v in results.values()])
        agg_baseline = np.mean([v["baseline_price_R2"] for v in results.values()])
        return {"per_ticker": results, "aggregate_MoE_R2": round(float(agg_r2), 6),
                "aggregate_baseline_R2": round(float(agg_baseline), 6),
                "n_tickers": len(results)}


def write_comparison(moe_results: dict, outpath: str | Path) -> None:
    per_ticker = moe_results["per_ticker"]
    lines = [
        "# MoE Comparison: News-Sensitive Gating vs Price-Only Baseline\n",
        f"\n## Aggregate\n",
        f"- MoE R²: {moe_results['aggregate_MoE_R2']:.6f}",
        f"- Baseline Price-Only R²: {moe_results['aggregate_baseline_R2']:.6f}",
        f"- ΔR²: {moe_results['aggregate_MoE_R2'] - moe_results['aggregate_baseline_R2']:+.6f}",
        f"- Tickers evaluated: {moe_results['n_tickers']}",
        f"\n## Per-Ticker\n",
    ]
    lines.append("| Ticker | Cluster | w_price | w_news | MoE_R² | baseline_price_R² | ΔR² |")
    lines.append("|--------|---------|---------|--------|--------|-------------------|-----|")
    for ticker in sorted(per_ticker.keys()):
        v = per_ticker[ticker]
        lines.append(
            f"| {ticker} | {v['cluster']} | {v['w_price']:.1f} | {v['w_news']:.1f} "
            f"| {v['MoE_R2']:.6f} | {v['baseline_price_R2']:.6f} | {v['delta_R2_vs_baseline']:+.6f} |"
        )
    Path(outpath).write_text("\n".join(lines), encoding="utf-8")


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    panel = build_panel()
    moe = SimpleMoE()

    comparisons = {}
    for target in TARGETS:
        moe.fit(panel, target)
        results = moe.evaluate(panel, target)
        comparisons[target] = results

        out = outdir / f"moe_comparison_{target}.md"
        write_comparison(results, out)
        written.append(out)

    aggregate_summary_path = outdir / "moe_comparison_summary.md"
    lines = ["# MoE Summary Across Targets\n", "\n| Target | MoE_R² | Baseline_R² | ΔR² | n_tickers |\n|--------|--------|-------------|-----|-----------|\n"]
    for target in TARGETS:
        r = comparisons[target]
        d = r["aggregate_MoE_R2"] - r["aggregate_baseline_R2"]
        lines.append(f"| {target} | {r['aggregate_MoE_R2']:.6f} | {r['aggregate_baseline_R2']:.6f} | {d:+.6f} | {r['n_tickers']} |\n")
    aggregate_summary_path.write_text("".join(lines), encoding="utf-8")
    written.append(aggregate_summary_path)
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
