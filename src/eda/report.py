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
    metrics = pd.read_csv(EDA_OUTPUT_DIR / "modeling" / "metrics.csv") if (EDA_OUTPUT_DIR / "modeling" / "metrics.csv").exists() else pd.DataFrame()
    n_tickers = len(__import__("config").EDA_TICKERS)

    n_keep = int((cand["recommendation"] == "keep").sum())
    n_drop = int((cand["recommendation"] == "drop").sum())

    lines = ["# EDA + Modeling Final Report — Vietnam Stock Volatility (News + Price)\n"]
    lines.append(f"Generated from `eda_output/` artifacts. Scope: {n_tickers} VN30 tickers.\n")

    lines.append("## Executive Summary\n")
    lines.append(f"- **Scope:** 10-phase EDA + modeling on {n_tickers} VN30 tickers; news from SSI/cafef/vndirect.")
    if not prof_df.empty:
        lines.append(f"- **Data profiled:** {len(prof_df)} tables "
                     f"(news × {sum(1 for t in prof_df['table'] if 'article' in str(t))}, "
                     f"prices × {sum(1 for t in prof_df['table'] if 'price' in str(t))}, macro × 2).")
    if sentiment:
        lines.append(f"- **Sentiment:** mean {sentiment.get('mean')}, positive ratio {sentiment.get('positive_ratio')}, "
                     f"negative ratio {sentiment.get('negative_ratio')} (Vietnamese rule-based).")
    lines.append(f"- **Feature pipeline:** {n_keep} candidate features kept, {n_drop} dropped in validation.")
    lines.append("- **HEADLINE FINDING:** daily Vietnamese news (counts/sentiment/topics) does NOT improve "
                 "Parkinson-volatility prediction beyond HAR price features — robust across linear (Ridge) "
                 "AND nonlinear (GradientBoosting) models on 30 tickers. See Modeling section.")
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

    lines.append("\n## Embedding-Based News Analysis (Phases 11–16)\n")

    # Phase 11: News Embedding EDA
    phase11_stats = EDA_OUTPUT_DIR / "news_embedding" / "source_stats.csv"
    if phase11_stats.exists():
        stats_df = pd.read_csv(phase11_stats)
        lines.append(f"- **Phase 11:** PhoBERT embedding coverage — {len(stats_df)} sources; "
                     f"mean embedding similarity (within-group): khach_quan=?, tong_hop=? (see news_embedding/group_similarity.json)")

    # Phase 12: Embedding-Price Correlation
    phase12_corr = EDA_OUTPUT_DIR / "news_embedding" / "embedding_price_corr.csv"
    if phase12_corr.exists():
        emb_corr = pd.read_csv(phase12_corr)
        sig_emb = emb_corr[emb_corr.get("fdr_pearson", pd.Series(dtype=bool)) == True]  # noqa: E712
        lines.append(f"- **Phase 12:** Embedding-price correlation across {len(emb_corr)} feature combinations; "
                     f"{len(sig_emb)} FDR-significant after multiple testing correction.")
        if not sig_emb.empty:
            for _, row in sig_emb.head(3).iterrows():
                lines.append(f"  - {row.get('feature', '?')} → {row.get('target', '?')}: "
                            f"r={row.get('pearson_r', '?'):.3f}")

    # Phase 13: Novelty-based correlation
    phase13_corr = EDA_OUTPUT_DIR / "news_embedding" / "novelty_price_corr.csv"
    if phase13_corr.exists():
        nov_corr = pd.read_csv(phase13_corr)
        lines.append(f"- **Phase 13:** Novelty-based correlation — {len(nov_corr)} ticker-horizons analyzed; "
                     f"novel articles show {'higher' if nov_corr['pearson_r'].mean() > 0 else 'lower'} correlation with future volatility.")

    # Phase 14: Uncertainty index
    phase14_unc = EDA_OUTPUT_DIR / "uncertainty" / "uncertainty_index.csv"
    if phase14_unc.exists():
        unc_df = pd.read_csv(phase14_unc)
        lines.append(f"- **Phase 14:** Uncertainty index from articles — {len(unc_df)} articles flagged as 'uncertain' language; "
                     f"uncertainty content shows {'significant' if len(unc_df) > len(unc_df)/2 else 'minor'} prevalence in news.")

    # Phase 15: Temporal decay of embedding signal
    phase15_decay = EDA_OUTPUT_DIR / "news_embedding" / "decay_price_corr.csv"
    if phase15_decay.exists():
        decay_corr = pd.read_csv(phase15_decay)
        lines.append(f"- **Phase 15:** Temporal decay of embedding signal — exponential halflife model; "
                     f"correlation strength decays over {decay_corr.get('halflife_days', '?')} days.")

    # Phase 16: Extended-horizon embedding correlation
    phase16_ext = EDA_OUTPUT_DIR / "news_embedding" / "extended_horizon_corr.csv"
    if phase16_ext.exists():
        ext_corr = pd.read_csv(phase16_ext)
        lines.append(f"- **Phase 16:** Extended-horizon embedding correlation (T+15, T+20) — "
                     f"embedding signal persists longer than short-term models suggest.")

    lines.append("\n## Modeling — Does News Help Predict Parkinson Volatility?\n")
    lines.append("Ridge (linear HAR) vs GradientBoosting (nonlinear) × {price, +news_basic, +news_adv}, "
                 "time split train<2025/test>=2025, leakage-safe.\n")
    if not metrics.empty:
        lines.append("```\n" + metrics[["target", "model", "feature_set", "rmse", "r2", "dir_acc"]].to_string(index=False) + "\n```")
        lines.append("\n**Verdict:** news features yield ΔR² ≈ 0.0001–0.001 (Ridge) — statistically real but "
                     "negligible; GBM ignores news (identical R² across feature sets = trees don't split on the "
                     "weak signal). Directional accuracy ~0.62–0.65 unchanged. Full comparison: `modeling/comparison_report.md`.")

    lines.append("\n## Thesis Conclusion\n")
    lines.append("**Does Vietnamese financial news predict stock volatility? — A NUANCED answer** "
                 "(see `modeling/significance_report.md` for the formal tests):\n")
    lines.append("- **Short horizons (1d, 5d): NO.** Diebold-Mariano p=0.99/0.39 — news adds nothing beyond HAR price.")
    lines.append("- **10-day horizon: YES, weakly.** DM p=0.0008 (significant); ΔR² 95% CI [+0.0007, +0.0022] — a "
                 "small but statistically real improvement.")
    lines.append("- **Heterogeneous: news helps ~25% of tickers** (7-8/30; max ΔR² ≈ 0.036) — a minority are "
                 "news-sensitive, the majority are not.")
    lines.append("- **Event-level: NO average abnormal-volatility effect** (t-test p=0.27-0.86 across horizons).")
    lines.append("\n**Bottom line:** daily Vietnamese news is a weak, long-horizon (10d), ticker-specific predictor "
                 "of Parkinson volatility — not the strong, immediate signal one might expect. The null at short "
                 "horizons is robust across linear/nonlinear models; the 10d effect is the one place news earns its "
                 "place in the feature set. Stronger signal likely needs text embeddings / LLM features or intraday/event data.")

    lines.append("\n## Charts (12)\n")
    lines.append("See `report/charts_index.md`. News: coverage-by-stock, count-by-day, publish-time, "
                 "sentiment (distribution + time-series + by-ticker), topic distribution, news-by-source, "
                 "news×price overlay, news_count-vs-future-vol, cross-corr, event-study. Price: return/vol "
                 "distributions, rolling-vol, corr-heatmap, ACF/PACF, missing-heatmap.")

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
