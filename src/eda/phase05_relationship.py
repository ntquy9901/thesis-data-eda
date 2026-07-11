"""Phase 5 — Relationship Analysis (per EDA Guide).

Quantifies news ↔ future-volatility/return relationships: Pearson, Spearman,
Mutual Information, Granger causality, cross-correlation at multiple lags.
Multiple-testing correction (FDR) is applied to p-values.

Pairs analyzed (per EDA Guide): news count vs future vol; sentiment vs future
vol; topic vs vol; negative news vs return. **Parkinson vol (pk_t+h) is the
primary target** because the sibling baselines predict it; realized vol
(rv_t+h) is reported alongside.

Outputs (under ``eda_output/relationship/``):
- ``corr_matrix.csv``, ``granger_results.json``, ``cross_corr.json``, ``mi_results.json``

Reads: ``eda_output/news/sparse_news_features.parquet`` + ``eda_output/price/price_metrics_<ticker>.parquet``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import ensure_output_dirs, phase_output_dir

TARGETS = ["pk_t+1", "pk_t+5", "pk_t+10", "rv_t+1", "rv_t+5", "rv_t+10", "log_returns"]
NEWS_FEATURES = ["news_count_1d", "news_count_3d", "news_count_5d", "sentiment_mean"]
MAX_LAG = 5


# ---------- pure helpers (unit-tested) ----------
def pearson_spearman(x: pd.Series, y: pd.Series) -> dict:
    """Pearson + Spearman r and p on aligned, NaN-dropped pairs.

    Returns None r/p when input has <3 points OR zero variance (constant input
    would otherwise yield NaN p-values that poison FDR correction downstream).
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    none = {"n": int(len(df)), "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None}
    if len(df) < 3 or df["x"].nunique() <= 1 or df["y"].nunique() <= 1:
        return none
    pr, pp = _pearson(df["x"], df["y"])
    sr, sp = _spearman(df["x"], df["y"])
    return {
        "n": int(len(df)),
        "pearson_r": _round(pr), "pearson_p": _round(pp),
        "spearman_r": _round(sr), "spearman_p": _round(sp),
    }


def _round(v) -> float | None:
    """Round a scalar to 4 dp; NaN (float) → None so it can't poison FDR."""
    import math

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4)


def _pearson(x, y):
    from scipy.stats import pearsonr

    return pearsonr(x, y)


def _spearman(x, y):
    from scipy.stats import spearmanr

    return spearmanr(x, y)


def mutual_information(x: pd.Series, y: pd.Series) -> float | None:
    """Mutual information between feature x and target y (NaN-dropped)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 10:
        return None
    from sklearn.feature_selection import mutual_info_regression

    mi = mutual_info_regression(df[["x"]], df["y"], random_state=0)
    return round(float(mi[0]), 4)


def granger_causality(cause: pd.Series, effect: pd.Series, maxlag: int = MAX_LAG) -> dict:
    """Granger causality: does `cause` predict `effect`? Returns min-p over lags."""
    df = pd.DataFrame({"cause": cause, "effect": effect}).dropna()
    if len(df) < 30:
        return {"min_p": None, "best_lag": None, "significant": False}
    from statsmodels.tsa.stattools import grangercausalitytests

    try:
        data = df[["effect", "cause"]].to_numpy()  # [predicted, predictor]
        res = grangercausalitytests(data, maxlag=maxlag)
    except Exception:
        return {"min_p": None, "best_lag": None, "significant": False}
    pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)}
    best_lag = min(pvals, key=pvals.get)
    min_p = pvals[best_lag]
    return {"min_p": round(float(min_p), 4), "best_lag": int(best_lag), "significant": bool(min_p < 0.05)}


def cross_correlation(a: pd.Series, b: pd.Series, max_lag: int = MAX_LAG) -> dict:
    """Correlation between `a` and `b` at lags -max_lag..+max_lag (NaN-dropped, aligned)."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < max_lag * 2:
        return {}
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        out[lag] = round(float(df["a"].corr(df["b"].shift(lag))), 4)
    return out


def fdr_correct(pvalues: list[float]) -> list[bool]:
    """Benjamini-Hochberg FDR correction → boolean reject flags (alpha=0.05)."""
    from statsmodels.stats.multitest import multipletests

    arr = np.array([p if pd.notna(p) else 1.0 for p in pvalues], dtype=float)
    reject, _, _, _ = multipletests(arr, alpha=0.05, method="fdr_bh")  # [0]=reject mask
    return [bool(r) for r in reject]


# ---------- phase runner ----------
def _load_joined_panel() -> pd.DataFrame:
    """Join sparse news panel + per-ticker price metrics on (ticker, date)."""
    from src.eda.common import EDA_OUTPUT_DIR

    news_path = EDA_OUTPUT_DIR / "news" / "sparse_news_features.parquet"
    if not news_path.exists():
        return pd.DataFrame()
    news = pd.read_parquet(news_path)
    news["trading_date"] = pd.to_datetime(news["trading_date"]).dt.normalize()

    frames = []
    for ticker in EDA_TICKERS:
        pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
        if not pq.exists():
            continue
        price = pd.read_parquet(pq)
        price["date"] = pd.to_datetime(price["date"]).dt.normalize()
        price["ticker"] = ticker
        # TARGETS already includes log_returns — select ticker+date+targets (no dup)
        cols = ["ticker", "date"] + [c for c in TARGETS if c in price.columns]
        frames.append(price[cols])
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True).rename(columns={"date": "trading_date"})
    return news.merge(prices, on=["ticker", "trading_date"], how="inner")


def run_phase() -> list:
    from src.eda.common import configure_plots

    ensure_output_dirs()
    configure_plots()
    outdir = phase_output_dir("relationship")
    panel = _load_joined_panel()
    if panel.empty:
        return []
    written = []

    # Derive "negative news" feature (sentiment_mean < 0 on a news day)
    if "sentiment_mean" in panel.columns:
        panel["neg_news"] = (panel["sentiment_mean"] < 0).astype(int)
    features = NEWS_FEATURES + (["neg_news"] if "neg_news" in panel.columns else [])

    # Pearson/Spearman + MI for each feature × target; FDR on BOTH p-value sets
    rows, p_pearson, p_spearman, mi_rows = [], [], [], []
    for feat in features:
        for tgt in TARGETS:
            if tgt not in panel.columns:
                continue
            ps = pearson_spearman(panel[feat], panel[tgt])
            mi = mutual_information(panel[feat], panel[tgt])
            rows.append({"feature": feat, "target": tgt, **ps, "mi": mi})
            p_pearson.append(ps["pearson_p"])
            p_spearman.append(ps["spearman_p"])
            mi_rows.append({"feature": feat, "target": tgt, "mutual_info": mi})
    corr = pd.DataFrame(rows)
    if p_pearson:
        corr["fdr_pearson"] = fdr_correct(p_pearson)
        corr["fdr_spearman"] = fdr_correct(p_spearman)
    corr.to_csv(outdir / "corr_matrix.csv", index=False, encoding="utf-8")
    written.append(outdir / "corr_matrix.csv")
    (outdir / "mi_results.json").write_text(json.dumps(mi_rows, indent=2), encoding="utf-8")
    written.append(outdir / "mi_results.json")

    # Granger (per ticker, news_count_1d → pk_t+1)
    granger = {}
    for ticker, sub in panel.groupby("ticker"):
        if "news_count_1d" in sub and "pk_t+1" in sub:
            g = granger_causality(sub["news_count_1d"], sub["pk_t+1"])
            if g["min_p"] is not None:
                granger[ticker] = g
    (outdir / "granger_results.json").write_text(json.dumps(granger, indent=2), encoding="utf-8")
    written.append(outdir / "granger_results.json")

    # Cross-correlation PER TICKER (pooled shift would cross ticker boundaries),
    # then average across tickers → JSON + PNG
    if "news_count_1d" in panel and "pk_t+1" in panel:
        per_ticker = {}
        for ticker, sub in panel.groupby("ticker"):
            per_ticker[ticker] = cross_correlation(sub["news_count_1d"], sub["pk_t+1"])
        # average across tickers per lag
        avg = {}
        for lag in range(-MAX_LAG, MAX_LAG + 1):
            vals = [d[lag] for d in per_ticker.values() if d and lag in d]
            avg[lag] = round(float(np.mean(vals)), 4) if vals else None
        (outdir / "cross_corr.json").write_text(
            json.dumps({"per_ticker": per_ticker, "mean_across_tickers": avg}, indent=2),
            encoding="utf-8",
        )
        written.append(outdir / "cross_corr.json")
        _plot_cross_corr(avg, outdir / "cross_corr.png")
        written.append(outdir / "cross_corr.png")

    return written


def _plot_cross_corr(avg: dict, path) -> None:
    import matplotlib.pyplot as plt

    lags = sorted(k for k in avg if avg[k] is not None)
    vals = [avg[k] for k in lags]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, vals, color="teal")
    ax.set_title("Cross-correlation: news_count_1d vs pk_t+1 (mean across tickers)")
    ax.set_xlabel("lag (trading days)")
    ax.set_ylabel("correlation")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
