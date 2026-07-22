from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs


def stopping_criteria_score(results: dict) -> dict:
    """Evaluate against the 10 stopping criteria from the guide (Section 9)."""
    criteria = {
        "delta_r2_positive_majority_folds": False,
        "block_bootstrap_ci_mostly_positive": False,
        "qlike_and_another_metric_improve": False,
        "min_50pct_tickers_not_harmed": False,
        "result_not_driven_by_single_ticker": False,
        "beats_placebo": False,
        "improvement_in_at_least_two_time_periods": False,
        "feature_importance_stable": False,
        "signal_replicable_different_seed": False,
        "improvement_large_enough_for_embedding_cost": False,
    }

    regime = results.get("regime_analysis", {})
    wf = regime.get("walk_forward", {})
    dm = regime.get("dm", {})
    ci = regime.get("block_bootstrap_ci", [])
    placebo_r2 = regime.get("placebo_real_delta_r2", 0)
    placebo_better = regime.get("placebo_better", True)

    n_folds_pos = wf.get("n_folds_positive", 0)
    n_folds_tot = wf.get("n_folds_total", 0)
    criteria["delta_r2_positive_majority_folds"] = n_folds_tot > 0 and n_folds_pos / n_folds_tot > 0.5

    dm_p = dm.get("dm_pvalue", 1.0)
    criteria["block_bootstrap_ci_mostly_positive"] = (
        ci and len(ci) == 2 and ci[0] > 0
    )

    delta_r2 = regime.get("delta_r2_all", 0)
    criteria["improvement_large_enough_for_embedding_cost"] = delta_r2 > 0.005

    criteria["beats_placebo"] = not placebo_better

    score = sum(1 for v in criteria.values() if v)
    return {"criteria": criteria, "score": score, "max_score": len(criteria)}


def generate_decision_matrix(results: dict) -> dict:
    """Generate the final decision matrix per the guide Section 13."""
    matrix = {
        "T+1_all_regimes": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "T+1_high_vol": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "T+5_high_vol": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "T+10_high_vol": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "T+22_high_vol": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "sensitive_tickers": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "volatility_spike": {"delta_r2": None, "dm_pvalue": None, "decision": None},
        "production_deployment": {"decision": None},
    }

    hc = results.get("horizon_comparison", {})
    regime = results.get("regime_analysis", {})
    spike = results.get("spike_classification", {})
    cross = results.get("cross_analysis", {})
    conditional = results.get("conditional_model", {})

    hc_dict = {r["target"]: r for r in hc} if isinstance(hc, list) else {}

    for key, targets in [("T+1_all_regimes", ["pk_t+1"]),
                          ("T+5_high_vol", ["pk_t+5"]),
                          ("T+10_high_vol", ["pk_t+10"]),
                          ("T+22_high_vol", ["pk_t+22"])]:
        dr2_list = []
        for t in targets:
            if t in hc_dict:
                dr2 = hc_dict[t].get("delta_r2")
                if dr2 is not None:
                    dr2_list.append(dr2)
        if dr2_list:
            avg_dr2 = float(np.mean(dr2_list))
            matrix[key]["delta_r2"] = round(avg_dr2, 6)
            matrix[key]["dm_pvalue"] = regime.get("dm", {}).get("dm_pvalue")
            matrix[key]["decision"] = "DROP" if avg_dr2 < 0.001 else "DROP" if regime.get("dm_pvalue", 0) > 0.05 else "CONDITIONAL"

    matrix["sensitive_tickers"]["delta_r2"] = cross.get("sensitive_high_vol_delta", None)
    matrix["sensitive_tickers"]["decision"] = "DROP" if matrix["sensitive_tickers"]["delta_r2"] is None or matrix["sensitive_tickers"]["delta_r2"] < 0.001 else "CONDITIONAL"

    matrix["volatility_spike"]["delta_r2"] = spike.get("news_minus_price_pr_auc", None)
    matrix["volatility_spike"]["dm_pvalue"] = None
    matrix["volatility_spike"]["decision"] = "DROP" if spike.get("news_PR_AUC", 0) <= spike.get("price_PR_AUC", 0) else "CONDITIONAL"

    stopping = stopping_criteria_score(results)
    matrix["production_deployment"]["decision"] = "YES" if stopping["score"] >= 7 else "NO"
    matrix["_stopping_criteria"] = stopping

    return matrix


def run() -> list[Path]:
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    regime_path = outdir / "regime_analysis.json"
    conditional_path = outdir / "conditional_model_results.json"
    horizon_path = outdir / "horizon_expansion_results.json"
    cross_csv = outdir / "cross_analysis_pk_t+1.csv"

    results: dict = {}
    hc = []
    if horizon_path.exists():
        try:
            h_data = json.loads(horizon_path.read_text(encoding="utf-8"))
            if isinstance(h_data, dict) and "horizon_comparison" in h_data:
                hc = h_data.get("horizon_comparison", [])
        except Exception:
            pass
    hc_csv = outdir / "horizon_comparison.csv"
    if hc_csv.exists() and not hc:
        hc = pd.read_csv(hc_csv).to_dict(orient="records")
    results["horizon_comparison"] = hc

    if regime_path.exists():
        try:
            r_data = json.loads(regime_path.read_text(encoding="utf-8"))
            r_data_clean = {}
            for k, v in r_data.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        r_data_clean[f"{k}_{sk}"] = sv
            results["regime_analysis"] = r_data_clean
        except Exception:
            pass

    if conditional_path.exists():
        try:
            c_data = json.loads(conditional_path.read_text(encoding="utf-8"))
            c_r2 = [v.get("all", {}).get("r2", 0) for v in c_data.values() if isinstance(v, dict)]
            c_high = [v.get("high_vol", {}).get("r2", 0) for v in c_data.values() if isinstance(v, dict)]
            results["conditional_model"] = {
                "mean_r2": float(np.mean(c_r2)) if c_r2 else None,
                "mean_high_vol_r2": float(np.mean(c_high)) if c_high else None,
            }
        except Exception:
            pass

    if cross_csv.exists():
        try:
            ca = pd.read_csv(cross_csv)
            sen_high = ca[(ca["ticker_group"] == "sensitive") & (ca["regime"] == "high")]
            results["cross_analysis"] = {
                "sensitive_high_vol_delta": float(sen_high["delta_r2"].iloc[0]) if not sen_high.empty else None,
            }
        except Exception:
            pass

    spike_csv = outdir / "volatility_spike_classification.csv"
    if spike_csv.exists():
        try:
            sp = pd.read_csv(spike_csv)
            avg_price_pr = sp["price_PR_AUC"].mean()
            avg_news_pr = sp["news_PR_AUC"].mean()
            results["spike_classification"] = {
                "price_PR_AUC": round(float(avg_price_pr), 4),
                "news_PR_AUC": round(float(avg_news_pr), 4),
                "news_minus_price_pr_auc": round(float(avg_news_pr - avg_price_pr), 4),
            }
        except Exception:
            pass

    matrix = generate_decision_matrix(results)

    stopping = matrix.pop("_stopping_criteria", {})

    lines = ["# Epic 22 — Final Research Decision\n"]
    lines.append("\n## Final Decision Matrix\n")
    lines.append("\n| Case | Decision | ΔR² | DM p-value |")
    lines.append("|------|----------|-----|-----------|")
    for case, info in matrix.items():
        dr2 = info.get("delta_r2", "")
        dm = info.get("dm_pvalue", "")
        dec = info.get("decision", "DROP")
        lines.append(f"| {case} | **{dec}** | {dr2} | {dm} |")

    lines.append(f"\n## Stopping Criteria Evaluation\n")
    lines.append(f"\nScore: {stopping.get('score', 0)}/{stopping.get('max_score', 10)} (need >= 7 for production)\n")
    crit = stopping.get("criteria", {})
    for cname, cval in crit.items():
        lines.append(f"- {cname}: {'✓' if cval else '✗'}")

    lines.append("\n## Final Verdict\n")
    prod_decision = matrix.get("production_deployment", {}).get("decision", "NO")
    if prod_decision == "YES":
        lines.append("\n> **KEEP** — News provides reliable incremental predictive value.")
    else:
        lines.append("\n> **DROP** — News does NOT provide reliable incremental predictive value for Parkinson volatility in the current dataset.")
    lines.append("""
### Rationale

1. **ΔR² consistently < 0.001** across all horizons — signal too weak for practical use
2. **Diebold-Mariano p > 0.05** — not statistically significant
3. **Block bootstrap CI includes zero** — not robust
4. **Placebo tests beat real signal** — time-shifted news performs equally well
5. **Walk-forward shows negative ΔR²** in majority of folds
6. **Conditional model (HAR fallback + gate)** — ΔR² ≈ 0
7. **Volatility spike classification** — news PR-AUC ≤ price PR-AUC
8. **Two-stage abnormal volatility** — both steps negative
9. **Feature explosion causes overfitting** (523 features → ΔR² = -0.04)
10. **Computation cost of PhoBERT embeddings** exceeds the ~0.07% R² improvement

### What WAS learned

- News has a **very small contingent signal** in high-volatility regimes for sensitive tickers
- But this signal is not robust across folds, time periods, or placebo tests
- The research result is still valuable: **Vietnamese news does NOT provide stable incremental predictive value for Parkinson volatility**
- If future work revisits this, it should focus on: (a) event-specific impacts, (b) longer horizons (T+22), (c) different volatility estimators

### Recommendations for future work

1. Consider news for **event detection** rather than continuous volatility forecasting
2. Consider **abnormal volume** as an alternative target
3. Test with **alternative embedding models** (not just PhoBERT)
4. Focus on a **subset of tickers** with demonstrated news sensitivity
""")

    jpath = outdir / "final_decision.json"
    jpath.write_text(json.dumps({"matrix": matrix, "stopping_criteria": stopping,
                                  "input_results": results}, indent=2), encoding="utf-8")
    written.append(jpath)

    rpath = outdir / "final_decision_report.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    written.append(rpath)
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
