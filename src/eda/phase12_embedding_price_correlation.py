"""Phase 12 — News embedding x price correlation (Story 11-3).

Is the news embedding signal related to price/volatility, and is that relationship
linear (Pearson) or only non-linear/monotonic (Spearman, mutual information)?
Reuses the Pearson/Spearman/MI/FDR machinery from ``phase05_relationship`` — no
statistics reimplemented here.

Outputs -> eda_output/news_embedding/:
- ``embedding_price_corr.csv`` — long format, one row per (emb_i or emb_norm, target)
- ``embedding_price_corr_summary.json`` — linear vs non-linear-only dim counts + top dims
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase05_relationship import TARGETS, fdr_correct, mutual_information, pearson_spearman


def _load_joined_panel() -> pd.DataFrame:
    """Join advanced_news_features.parquet (ticker, date, emb_*) with per-ticker price targets."""
    adv_path = EDA_OUTPUT_DIR / "modeling" / "advanced_news_features.parquet"
    if not adv_path.exists():
        return pd.DataFrame()
    adv = pd.read_parquet(adv_path)
    if adv.empty:
        return pd.DataFrame()
    adv["date"] = pd.to_datetime(adv["date"]).dt.normalize()

    emb_cols = [c for c in adv.columns if c.startswith("emb_")]
    if emb_cols:
        adv["emb_norm"] = np.linalg.norm(adv[emb_cols].to_numpy(dtype=float), axis=1)

    frames = []
    for ticker in EDA_TICKERS:
        pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
        if not pq.exists():
            continue
        price = pd.read_parquet(pq)
        price["date"] = pd.to_datetime(price["date"]).dt.normalize()
        price["ticker"] = ticker
        cols = ["ticker", "date"] + [c for c in TARGETS if c in price.columns]
        frames.append(price[cols])
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True)
    return adv.merge(prices, on=["ticker", "date"], how="inner")


def compute_correlations(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Pearson/Spearman/MI for each feature x target, with FDR correction (Pearson, Spearman separately)."""
    rows, p_pearson, p_spearman = [], [], []
    for feat in features:
        if feat not in panel.columns:
            continue
        for tgt in TARGETS:
            if tgt not in panel.columns:
                continue
            ps = pearson_spearman(panel[feat], panel[tgt])
            mi = mutual_information(panel[feat], panel[tgt])
            rows.append({"feature": feat, "target": tgt, **ps, "mi": mi})
            p_pearson.append(ps["pearson_p"])
            p_spearman.append(ps["spearman_p"])
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["fdr_pearson"] = fdr_correct(p_pearson)
        corr["fdr_spearman"] = fdr_correct(p_spearman)
    return corr


def summarize(corr: pd.DataFrame) -> dict:
    """Linear (Pearson-significant) vs non-linear-only (Spearman/MI-significant, Pearson not) counts."""
    if corr.empty:
        return {"note": "no correlation results (missing advanced_news_features or price_metrics)"}
    emb_rows = corr[corr["feature"].str.startswith("emb_") & (corr["feature"] != "emb_norm")]
    linear = emb_rows[emb_rows["fdr_pearson"]]
    nonlinear_only = emb_rows[emb_rows["fdr_spearman"] & ~emb_rows["fdr_pearson"]]

    top_by_target = {}
    for tgt, sub in emb_rows.dropna(subset=["pearson_r"]).groupby("target"):
        top = sub.reindex(sub["pearson_r"].abs().sort_values(ascending=False).index).head(5)
        top_by_target[tgt] = top[["feature", "pearson_r", "pearson_p", "fdr_pearson"]].to_dict("records")

    return {
        "n_emb_dim_target_pairs": int(len(emb_rows)),
        "linear_significant_count": int(len(linear)),
        "nonlinear_only_significant_count": int(len(nonlinear_only)),
        "top5_abs_pearson_r_per_target": top_by_target,
        "interpretation": (
            "linear_significant_count = (emb_i, target) pairs where Pearson r survives FDR "
            "(a straight-line relationship). nonlinear_only_significant_count = pairs where "
            "Spearman is FDR-significant but Pearson is not (monotonic-but-not-linear signal)."
        ),
    }


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("news_embedding")
    panel = _load_joined_panel()
    if panel.empty:
        return []

    emb_cols = [c for c in panel.columns if c.startswith("emb_")]
    features = emb_cols + (["emb_norm"] if "emb_norm" in panel.columns else [])
    corr = compute_correlations(panel, features)
    if corr.empty:
        return []

    written = []
    out_csv = outdir / "embedding_price_corr.csv"
    corr.to_csv(out_csv, index=False, encoding="utf-8")
    written.append(out_csv)

    summary = summarize(corr)
    out_json = outdir / "embedding_price_corr_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.append(out_json)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
