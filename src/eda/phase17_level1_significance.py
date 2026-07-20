"""Phase 17 — Level 1 Statistical Significance (Story 14-1).

Per ``docs/gpt-guide/news_feature_evaluation_guideline.md`` Level 1: for every candidate
feature (Positive/Negative/Fear/Optimism/Uncertainty score, Event type), evaluate Pearson,
Spearman, Kendall Tau, Mutual Information, AND Distance Correlation against each forward
Parkinson-vol target — MI/dcor catch nonlinear relationships Pearson would score as zero.

Candidate features (12): 5 sentiment scores (``src.features.sentiment_scores``) + 7 event-type
counts (the existing ``TOPIC_CATEGORIES`` taxonomy, counted market-wide here).

Outputs -> eda_output/level1_significance/:
- ``level1_corr.csv`` — long format, one row per (feature, target): all 5 statistics + FDR flags
- ``level1_summary.json`` — per-method significant-pair counts + MI-but-not-Pearson highlights
  (guideline's headline interpretation: "MI ≈ 0 -> feature likely useless"; "Pearson ≈ 0 but
  MI > 0 -> possible nonlinear predictive relationship").
"""

from __future__ import annotations

import json

import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase05_relationship import (
    TARGETS,
    distance_correlation,
    fdr_correct,
    kendall_tau,
    mutual_information,
    pearson_spearman,
)
from src.features.sentiment_scores import EVENT_TYPE_COLS, SENTIMENT_SCORE_COLS

CANDIDATE_FEATURES = SENTIMENT_SCORE_COLS + EVENT_TYPE_COLS


def _load_joined_panel() -> pd.DataFrame:
    """Join sentiment_features.parquet (ticker, date, 5 scores + 7 event counts) with per-ticker
    price targets."""
    feat_path = EDA_OUTPUT_DIR / "modeling" / "sentiment_features.parquet"
    if not feat_path.exists():
        return pd.DataFrame()
    feats = pd.read_parquet(feat_path)
    if feats.empty:
        return pd.DataFrame()
    feats["date"] = pd.to_datetime(feats["date"]).dt.normalize()

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
    return feats.merge(prices, on=["ticker", "date"], how="inner")


def compute_level1_correlations(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """All 5 statistics for each feature x target, FDR-corrected on Pearson/Spearman/Kendall."""
    rows, p_pearson, p_spearman, p_kendall = [], [], [], []
    for feat in features:
        if feat not in panel.columns:
            continue
        for tgt in TARGETS:
            if tgt not in panel.columns:
                continue
            ps = pearson_spearman(panel[feat], panel[tgt])
            kt = kendall_tau(panel[feat], panel[tgt])
            mi = mutual_information(panel[feat], panel[tgt])
            dc = distance_correlation(panel[feat], panel[tgt])
            rows.append({
                "feature": feat, "target": tgt, **ps,
                "kendall_tau": kt["kendall_tau"], "kendall_p": kt["kendall_p"],
                "mi": mi, "dcor": dc,
            })
            p_pearson.append(ps["pearson_p"])
            p_spearman.append(ps["spearman_p"])
            p_kendall.append(kt["kendall_p"])
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["fdr_pearson"] = fdr_correct(p_pearson)
        corr["fdr_spearman"] = fdr_correct(p_spearman)
        corr["fdr_kendall"] = fdr_correct(p_kendall)
    return corr


def summarize(corr: pd.DataFrame) -> dict:
    """Per guideline interpretation: MI≈0 -> useless; Pearson≈0 but MI>0 -> nonlinear signal."""
    if corr.empty:
        return {"note": "no correlation results (missing sentiment_features or price_metrics)"}
    linear = corr[corr["fdr_pearson"] == True]  # noqa: E712
    nonlinear_only = corr[(corr["fdr_spearman"] == True) & (corr["fdr_pearson"] == False)]  # noqa: E712
    likely_useless = corr[corr["mi"].notna() & (corr["mi"] < 0.01)]
    nonlinear_candidates = corr[
        corr["mi"].notna() & (corr["mi"] > 0.01) & corr["pearson_r"].notna() & (corr["pearson_r"].abs() < 0.02)
    ]
    return {
        "n_feature_target_pairs": int(len(corr)),
        "linear_significant_count": int(len(linear)),
        "nonlinear_only_significant_count": int(len(nonlinear_only)),
        "likely_useless_mi_near_zero_count": int(len(likely_useless)),
        "nonlinear_candidate_pearson_near_zero_mi_positive_count": int(len(nonlinear_candidates)),
        "nonlinear_candidates": nonlinear_candidates[["feature", "target", "pearson_r", "mi", "dcor"]].to_dict("records"),
        "interpretation": (
            "MI ~ 0 -> feature likely useless for this target. Pearson ~ 0 but MI > 0 -> possible "
            "nonlinear predictive relationship (per docs/gpt-guide guideline Level 1)."
        ),
    }


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("level1_significance")
    panel = _load_joined_panel()
    if panel.empty:
        return []

    corr = compute_level1_correlations(panel, CANDIDATE_FEATURES)
    if corr.empty:
        return []

    written = []
    out_csv = outdir / "level1_corr.csv"
    corr.to_csv(out_csv, index=False, encoding="utf-8")
    written.append(out_csv)

    summary = summarize(corr)
    out_json = outdir / "level1_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.append(out_json)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
