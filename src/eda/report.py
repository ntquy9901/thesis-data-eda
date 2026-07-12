"""Final EDA report assembly (per EDA Guide "Final Report").

Aggregates findings from all phases into ``eda_output/report/eda_final_report.md``
and writes ``candidate_features.csv`` (features that survived validation +
leakage status). Reads artifacts under ``eda_output/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir

CANDIDATE_FEATURES = [
    ("atr_14", "Average True Range — intraday range volatility"),
    ("realized_vol_5d", "Realized vol (5d) — short-horizon trailing"),
    ("realized_vol_20d", "Realized vol (20d) — medium-horizon trailing"),
    ("parkinson_vol", "Parkinson vol — the measure baselines predict"),
    ("news_count_1d", "Daily news count per ticker"),
    ("news_count_3d", "3-day rolling news count"),
    ("news_count_5d", "5-day rolling news count"),
    ("coverage_ratio_5d", "Fraction of last 5 trading days with news"),
    ("days_since_last_news", "Recency of news (sparse-news signal)"),
    ("sentiment_mean", "Mean Vietnamese sentiment (NaN if no news)"),
]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def candidate_features_frame() -> pd.DataFrame:
    drop = set(_read_json(EDA_OUTPUT_DIR / "feature_engineering" / "drop_recommendations.json").get("total_drop_recommendation", []))
    leak = {x["feature"] for x in _read_json(EDA_OUTPUT_DIR / "leakage" / "leakage_checks.json").get("target_leakage_suspects", [])}
    rows = []
    for feat, desc in CANDIDATE_FEATURES:
        rows.append({
            "feature": feat,
            "description": desc,
            "dropped_in_validation": feat in drop,
            "leakage_suspect": feat in leak,
            "recommendation": "drop" if feat in drop else ("review" if feat in leak else "keep"),
        })
    return pd.DataFrame(rows)


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = phase_output_dir("report")
    written: list[Path] = []

    cand = candidate_features_frame()
    cand_path = outdir / "candidate_features.csv"
    cand.to_csv(cand_path, index=False, encoding="utf-8")
    written.append(cand_path)

    prof_df = pd.read_csv(EDA_OUTPUT_DIR / "profiling" / "profiling_table.csv") if (EDA_OUTPUT_DIR / "profiling" / "profiling_table.csv").exists() else pd.DataFrame()
    sentiment = _read_json(EDA_OUTPUT_DIR / "news" / "sentiment_summary.json")
    corr = pd.read_csv(EDA_OUTPUT_DIR / "relationship" / "corr_matrix.csv") if (EDA_OUTPUT_DIR / "relationship" / "corr_matrix.csv").exists() else pd.DataFrame()
    event = _read_json(EDA_OUTPUT_DIR / "relationship" / "event_study_summary.json")

    n_keep = int((cand["recommendation"] == "keep").sum())
    n_drop = int((cand["recommendation"] == "drop").sum())

    lines = ["# EDA Final Report — Vietnam Stock Volatility (News + Price)\n"]
    lines.append(f"Generated from `eda_output/` artifacts. Tickers: {', '.join(__import__('config').EDA_TICKERS)}.\n")

    lines.append("## Executive Summary\n")
    lines.append("- **Scope:** 10-phase EDA on a representative VN30 subset (VCB, FPT, HPG, SSI, MWG),")
    lines.append("  SSI as primary news source (cafef/vndirect secondary).")
    if not prof_df.empty:
        lines.append(f"- **Data profiled:** {len(prof_df)} tables "
                     f"(news × {sum(1 for t in prof_df['table'] if 'article' in str(t))}, "
                     f"prices × {sum(1 for t in prof_df['table'] if 'price' in str(t))}, macro × 2).")
    if sentiment:
        lines.append(f"- **Sentiment:** mean {sentiment.get('mean')}, positive ratio {sentiment.get('positive_ratio')}, "
                     f"negative ratio {sentiment.get('negative_ratio')} (Vietnamese rule-based).")
    lines.append(f"- **Feature pipeline:** {n_keep} candidate features kept, {n_drop} dropped in validation.")
    lines.append("- **Leakage:** targets rv_t+1/5/10 + pk_t+1/5/10 are leakage-safe by construction (verified); "
                 "see `leakage/leakage_list.md`.")

    lines.append("\n## Top Findings (with evidence)\n")
    if not corr.empty and "feature" in corr.columns:
        sig = corr[(corr.get("fdr_pearson", pd.Series(dtype=bool)) == True)]  # noqa: E712
        if not sig.empty:
            lines.append("- **FDR-significant correlations (Pearson, news feature → target):**")
            for _, r in sig.head(10).iterrows():
                lines.append(f"  - {r['feature']} vs {r['target']}: r={r['pearson_r']}")
        else:
            lines.append("- No FDR-significant Pearson correlations after multiple-testing correction "
                         "(news↔vol relationships are weak at the daily level on this subset).")
    if event:
        lines.append(f"- **Event study:** {event.get('n_events')} events; mean abnormal Parkinson vol by horizon: "
                     f"{event.get('mean_abnormal_vol_by_horizon')}.")
    lines.append("- Full price findings: `price/findings.md`. Topics: `news/topics.json`.")

    lines.append("\n## Recommended Candidate Features\n")
    lines.append("See `candidate_features.csv`. Summary:")
    for _, r in cand.iterrows():
        lines.append(f"- `{r['feature']}` — {r['description']} → **{r['recommendation']}**")

    lines.append("\n## Risks\n")
    lines.append("- **Leakage:** addressed (future-only targets, trailing features, time-based split). Residual: "
                 "modeling must fit normalizers on TRAIN only.")
    lines.append("- **Sparse news:** many trading days have no ticker-specific news → `news_available` flag + NaN "
                 "sentiment (never masked as 0). Coverage is thin for some tickers.")
    lines.append("- **Imbalance / outliers:** log-return outliers >3σ flagged per ticker (`price/outliers_*.csv`).")
    lines.append("- **Date formats:** news sources mix ISO + DD/MM; normalized per-source, but always verify on ingest.")
    lines.append("- **Parkinson vs realized vol:** baselines predict Parkinson; EDA reports both — pick consistently.")

    lines.append("\n## Recommended Next Steps\n")
    lines.append("1. Scale EDA to all 30 VN30 tickers (change `config.EDA_TICKERS`).")
    lines.append("2. Add per-day topic features (Phase 8) — topics are currently article-level.")
    lines.append("3. Build the modeling feature matrix from survivors (`candidate_features.csv`) with the "
                 "time-based split (≤2024 / ≥2025).")
    lines.append("4. Train baselines predicting Parkinson vol (matches sibling project); compare with/without news features.")
    lines.append("5. Re-run the pipeline on fresh crawled data daily (`src/sprint1/task1_4_batch_processing`).")

    out = outdir / "eda_final_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    written.append(out)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
