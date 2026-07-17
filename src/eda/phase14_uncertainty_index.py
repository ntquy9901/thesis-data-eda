"""Phase 14 — Vietnamese Baker-Bloom-Davis (BBD) style uncertainty index (Story 12-2).

BBD's EPU index: an article counts as "uncertain" iff it contains >=1 term from EACH of three
keyword categories (Economy AND Policy AND Uncertainty). Daily (market-wide, ALL DISCOVERED
sources — both "khách quan" and "tổng hợp" groups combined, since this is a market-wide signal
not a per-group one) aggregate ratio, correlated against a MARKET-WIDE AVERAGE of price targets
(mean across all tickers per date) via the existing phase05_relationship helpers — this is a
single daily index, not a per-ticker feature, so it is correlated 1:1 per date (correlating it
against each ticker's target separately would duplicate the same X value ~30x per date and
understate the true p-values).

METHODOLOGICAL CAVEAT (inherited from BBD's own design, not a bug): like the original
Baker-Bloom-Davis index, this is plain keyword-presence counting with no negation handling
(e.g. "không có rủi ro" / "no risk" still matches "rủi ro") and no proximity/co-occurrence
requirement between the three categories. This is the accepted simplicity/cost tradeoff of the
BBD method — a more precise classifier is out of scope for this story.

Output -> eda_output/uncertainty/: uncertainty_index.csv, uncertainty_price_corr.csv,
uncertainty_price_corr_summary.json
"""

from __future__ import annotations

import json
import unicodedata

import pandas as pd

from config import EDA_TICKERS
from src.data.discover_news import discover_source_files, load_source
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase04_news_eda import _trading_calendar, effective_trading_date
from src.eda.phase05_relationship import TARGETS, fdr_correct, mutual_information, pearson_spearman

# Economy: macro/firm-performance terms an EPU-style article would mention alongside a policy +
# uncertainty term (BBD's "E" category, translated/localized, not an exhaustive econ vocabulary).
ECON_KW = ["kinh tế", "gdp", "tăng trưởng", "doanh nghiệp", "xuất khẩu", "lạm phát", "sản xuất"]
# Policy: government/regulatory-action terms (BBD's "P" category).
POLICY_KW = ["chính sách", "quy định", "ngân hàng nhà nước", "chính phủ", "thông tư", "nghị định", "quốc hội"]
# Uncertainty: explicit hedging/risk-language terms (BBD's "U" category). "biến động" and
# "thị trường" were deliberately EXCLUDED from ECON/UNCERTAINTY — both are near-ubiquitous in
# routine market reporting and would make the Economy/Uncertainty categories almost always true,
# defeating BBD's requirement that each category be independently discriminating.
UNCERTAINTY_KW = ["bất định", "không chắc chắn", "rủi ro", "lo ngại", "bất ổn", "khó lường"]


def _normalize(text: str) -> str:
    """NFC-normalize before substring matching — source CSVs may mix NFC/NFD Vietnamese text
    (per CLAUDE.md's documented cross-source encoding variance), and a keyword literal in one
    normalization form silently fails to match text in the other."""
    return unicodedata.normalize("NFC", str(text)).lower()


def is_uncertain(text: str) -> bool:
    """An article is "uncertain" iff it matches >=1 term from EACH of Economy, Policy, Uncertainty."""
    t = _normalize(text)
    return (
        any(kw in t for kw in ECON_KW)
        and any(kw in t for kw in POLICY_KW)
        and any(kw in t for kw in UNCERTAINTY_KW)
    )


def build_uncertainty_index() -> pd.DataFrame:
    """Daily (date, n_articles, n_uncertain, uncertainty_ratio) across ALL discovered sources
    (market-wide — not scoped to one khách_quan/tổng_hợp group)."""
    frames = []
    for source, path in discover_source_files().items():
        try:
            frames.append(load_source(source, path))
        except Exception:
            continue
    news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if news.empty:
        return pd.DataFrame()
    title = news.get("title", pd.Series(index=news.index)).fillna("")
    lead = news.get("lead", pd.Series(index=news.index)).fillna("")
    text = (title.astype(str) + " " + lead.astype(str)).str.strip()

    trading = _trading_calendar()
    eff_date = effective_trading_date(news["pub_date"], trading)
    df = pd.DataFrame({"date": eff_date.dt.normalize(), "uncertain": text.apply(is_uncertain)})
    df = df.dropna(subset=["date"])
    daily = df.groupby("date").agg(n_articles=("uncertain", "size"), n_uncertain=("uncertain", "sum")).reset_index()
    daily["uncertainty_ratio"] = daily["n_uncertain"] / daily["n_articles"]
    return daily


def _load_joined_panel() -> pd.DataFrame:
    """Join the market-wide uncertainty index with a MARKET-WIDE AVERAGE of price targets
    (mean across all tickers per date) — one row per date, not one row per (ticker, date).
    Joining the single daily uncertainty value against every ticker's target separately would
    duplicate that value ~30x per date (pseudo-replication), inflating the effective sample size
    and biasing the correlation p-values downward."""
    idx = build_uncertainty_index()
    if idx.empty:
        return pd.DataFrame()

    frames = []
    for ticker in EDA_TICKERS:
        pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
        if not pq.exists():
            continue
        price = pd.read_parquet(pq)
        price["date"] = pd.to_datetime(price["date"]).dt.normalize()
        cols = ["date"] + [c for c in TARGETS if c in price.columns]
        frames.append(price[cols])
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True)
    target_cols = [c for c in TARGETS if c in prices.columns]
    market = prices.groupby("date")[target_cols].mean().reset_index()
    return idx.merge(market, on="date", how="inner")


def compute_uncertainty_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman/MI for uncertainty_ratio x each target, FDR-corrected."""
    if "uncertainty_ratio" not in panel.columns:
        return pd.DataFrame()
    rows, p_pearson, p_spearman = [], [], []
    for tgt in TARGETS:
        if tgt not in panel.columns:
            continue
        ps = pearson_spearman(panel["uncertainty_ratio"], panel[tgt])
        mi = mutual_information(panel["uncertainty_ratio"], panel[tgt])
        rows.append({"feature": "uncertainty_ratio", "target": tgt, **ps, "mi": mi})
        p_pearson.append(ps["pearson_p"])
        p_spearman.append(ps["spearman_p"])
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["fdr_pearson"] = fdr_correct(p_pearson)
        corr["fdr_spearman"] = fdr_correct(p_spearman)
    return corr


def summarize(corr: pd.DataFrame) -> dict:
    """Pearson- vs Spearman-only significance summary, same shape as phase12/phase13."""
    if corr.empty:
        return {"note": "no correlation results (missing uncertainty index or price_metrics)"}
    linear = corr[corr["fdr_pearson"]]
    nonlinear_only = corr[corr["fdr_spearman"] & ~corr["fdr_pearson"]]
    return {
        "n_targets_tested": int(len(corr)),
        "linear_significant_count": int(len(linear)),
        "nonlinear_only_significant_count": int(len(nonlinear_only)),
        "interpretation": (
            "linear_significant_count = targets where Pearson r survives FDR (straight-line "
            "relationship with the market-wide daily uncertainty ratio). "
            "nonlinear_only_significant_count = targets where Spearman is FDR-significant but "
            "Pearson is not (monotonic-but-not-linear signal)."
        ),
    }


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("uncertainty")
    written = []

    idx = build_uncertainty_index()
    if idx.empty:
        return []
    idx_path = outdir / "uncertainty_index.csv"
    idx.to_csv(idx_path, index=False, encoding="utf-8")
    written.append(idx_path)

    panel = _load_joined_panel()
    if panel.empty:
        return written
    corr = compute_uncertainty_correlations(panel)
    if corr.empty:
        return written
    corr_path = outdir / "uncertainty_price_corr.csv"
    corr.to_csv(corr_path, index=False, encoding="utf-8")
    written.append(corr_path)

    summary_path = outdir / "uncertainty_price_corr_summary.json"
    summary_path.write_text(json.dumps(summarize(corr), indent=2, default=str), encoding="utf-8")
    written.append(summary_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
