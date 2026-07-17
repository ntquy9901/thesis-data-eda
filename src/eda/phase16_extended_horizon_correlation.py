"""Phase 16 — Extended-horizon (T+15, T+20) embedding x price correlation.

Extends Story 11-3's T+1/T+5/T+10 check to longer horizons (ad-hoc user question: does news
matter further out?). Reuses ``phase03_price_eda.parkinson_targets`` (parameterized by horizon)
and ``phase05_relationship`` statistics — no new target/statistics logic.

Output -> eda_output/news_embedding/: extended_horizon_corr.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EDA_TICKERS, PRICE_DATA_DIR
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase03_price_eda import add_returns, parkinson_targets, parkinson_volatility
from src.eda.phase05_relationship import fdr_correct, mutual_information, pearson_spearman

HORIZONS = (15, 20)
TARGETS = [f"pk_t+{h}" for h in HORIZONS]


def _build_extended_targets() -> pd.DataFrame:
    """Per-ticker pk_t+15/pk_t+20 (leakage-safe forward Parkinson vol)."""
    frames = []
    for ticker in EDA_TICKERS:
        p = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = add_returns(df)
        pk = parkinson_volatility(df["high"], df["low"])
        targets = parkinson_targets(pk, horizons=HORIZONS)
        out = pd.concat([df[["date"]], targets], axis=1)
        out["ticker"] = ticker
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_joined_panel() -> pd.DataFrame:
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

    price_extra = _build_extended_targets()
    if price_extra.empty:
        return pd.DataFrame()
    return adv.merge(price_extra, on=["ticker", "date"], how="inner")


def compute_extended_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/MI for each emb_i/emb_norm x pk_t+15/pk_t+20, FDR-corrected."""
    features = [c for c in panel.columns if c.startswith("emb_")]
    rows, p_pearson, p_spearman = [], [], []
    for feat in features:
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
    corr = compute_extended_correlations(panel)
    if corr.empty:
        return []
    out = outdir / "extended_horizon_corr.csv"
    corr.to_csv(out, index=False, encoding="utf-8")
    return [out]


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
