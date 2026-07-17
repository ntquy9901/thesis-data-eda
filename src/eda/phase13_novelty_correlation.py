"""Phase 13 — Embedding novelty/staleness x price correlation (Story 12-1).

Is "fresh" news (low similarity to recent same-ticker articles) more informative than "stale"
(rehashed) news? Reuses the Pearson/Spearman/MI/FDR machinery from ``phase05_relationship`` —
no statistics reimplemented here. Validation-only (no baseline-model retraining, per story scope).

CAVEAT: ``novelty_mean``'s rolling window (default 5 days) overlaps across consecutive trading
days, so this feature is autocorrelated over time — naive Pearson/Spearman/FDR p-values assume
i.i.d. samples and will look more "significant" than a rigorous (block-bootstrap-corrected) test
would support. Read the correlations below as exploratory, not confirmatory (same caveat as
``phase15_temporal_decay_correlation.py``'s decayed-embedding feature).

Output -> eda_output/news_embedding/: novelty_price_corr.csv
"""

from __future__ import annotations

import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase05_relationship import TARGETS, fdr_correct, mutual_information, pearson_spearman
from src.features.news_embeddings import novelty_daily


def _load_joined_panel(group: str = "tong_hop", window_days: int = 5) -> pd.DataFrame:
    """Join novelty_daily (ticker, date, novelty_mean) with per-ticker price targets."""
    novelty = novelty_daily(group, window_days)
    if novelty.empty:
        return pd.DataFrame()
    novelty["date"] = pd.to_datetime(novelty["date"]).dt.normalize()

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
    return novelty.merge(prices, on=["ticker", "date"], how="inner")


def compute_novelty_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/MI for novelty_mean x each target, FDR-corrected."""
    if "novelty_mean" not in panel.columns:
        return pd.DataFrame()
    rows, p_pearson, p_spearman = [], [], []
    for tgt in TARGETS:
        if tgt not in panel.columns:
            continue
        ps = pearson_spearman(panel["novelty_mean"], panel[tgt])
        mi = mutual_information(panel["novelty_mean"], panel[tgt])
        rows.append({"feature": "novelty_mean", "target": tgt, **ps, "mi": mi})
        p_pearson.append(ps["pearson_p"])
        p_spearman.append(ps["spearman_p"])
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["fdr_pearson"] = fdr_correct(p_pearson)
        corr["fdr_spearman"] = fdr_correct(p_spearman)
    return corr


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("news_embedding")
    panel = _load_joined_panel()
    if panel.empty:
        return []
    corr = compute_novelty_correlations(panel)
    if corr.empty:
        return []
    out = outdir / "novelty_price_corr.csv"
    corr.to_csv(out, index=False, encoding="utf-8")
    return [out]


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
