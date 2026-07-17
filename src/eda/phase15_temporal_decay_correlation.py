"""Phase 15 — Temporal-decay-weighted embedding x price correlation (Story 12-3).

Does a decay-weighted, multi-day news signal show a different horizon pattern (T+1/T+5/T+10)
than the flat same-day embedding average (Story 11-3)? Reuses phase05_relationship statistics.
Single fixed halflife (default 5 trading days) — per-horizon grid search is out of scope.

**CAVEAT (read before citing these numbers):** the 20-day lookback windows OVERLAP across
consecutive trading days (day T and day T+1 share ~19 days of contributing articles), so
consecutive rows are highly autocorrelated. Naive Pearson/Spearman p-values assume i.i.d. samples
and are inflated by this overlap — on real data this feature shows far more "FDR-significant"
pairs than the non-overlapping flat embedding (Story 11-3), which is a plausible sign of this
artifact rather than a genuinely stronger signal. Treat the correlation counts here as
directional/exploratory only; a rigorous read would need a block-bootstrap or Newey-West-style
correction for the overlapping-window autocorrelation (not implemented — out of scope for this
story's validation-only depth).

Output -> eda_output/news_embedding/: decay_price_corr.csv
"""

from __future__ import annotations

import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase05_relationship import TARGETS, fdr_correct, mutual_information, pearson_spearman
from src.features.news_embeddings import decayed_embedding_features


def _load_joined_panel(group: str = "tong_hop", halflife_days: int = 5) -> pd.DataFrame:
    """Join decayed embedding features (ticker, date, emb_decay_*) with per-ticker price targets."""
    decayed = decayed_embedding_features(group, halflife_days=halflife_days)
    if decayed.empty:
        return pd.DataFrame()
    decayed["date"] = pd.to_datetime(decayed["date"]).dt.normalize()

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
    return decayed.merge(prices, on=["ticker", "date"], how="inner")


def compute_decay_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/MI for each emb_decay_i x target, FDR-corrected."""
    emb_cols = [c for c in panel.columns if c.startswith("emb_decay_")]
    rows, p_pearson, p_spearman = [], [], []
    for feat in emb_cols:
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


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("news_embedding")
    panel = _load_joined_panel()
    if panel.empty:
        return []
    corr = compute_decay_correlations(panel)
    if corr.empty:
        return []
    out = outdir / "decay_price_corr.csv"
    corr.to_csv(out, index=False, encoding="utf-8")
    return [out]


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
