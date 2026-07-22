# Night Run 2 — Epics 19-22 Final Research Decision

**Date:** 2026-07-22 06:00  
**Objective:** Validate high-vol regime signal, build conditional model, expand horizons, make final keep/drop decision

## Summary

### Epic 19 — Regime-Conditional News Validation

| Analysis | Key Finding |
|---|---|
| Ex-ante regimes (rolling percentile, no look-ahead) | Implemented for 60/70/80/90% thresholds |
| Threshold sensitivity | All thresholds show tiny ΔR² in high-vol (+0.00005 to +0.0065) |
| High-vol-only model | **NEGATIVE** ΔR² (−0.05 to −0.12) — catastrophically worse |
| Sensitive × High-vol cross | Sensitive tickers in high-vol: +0.001 to +0.003 (n=184 only) |
| Walk-forward | **0/3 folds positive** for T+5, T+10, T+22; 1/3 for T+1 |
| DM test | All p > 0.05 — NOT significant |
| Block bootstrap CI | Includes zero for all horizons |
| Placebo tests | Time-shifted news **often beats** real-aligned news |
| **Decision** | **DROP — all horizons** (score 0/6) |

### Epic 20 — Conditional Lightweight Model

| Component | Result |
|---|---|
| Feature set | Reduced to 13 features (from 523) |
| HAR fallback architecture | Implemented (price-only for low-vol, +news adjustment for high-vol) |
| Hard gate | ΔR² ≈ 0 (0.000001) |
| Soft gate (sigmoid) | ΔR² ≈ 0 (0.000002) |
| Clipping + shrinkage | Applied (λ=0.3, clip at 1st percentile) |
| **Verdict** | Conditional model does NOT recover signal |

### Epic 21 — Horizon and Target Expansion

| Horizon | ΔR² (all regimes) | ΔR² (high-vol) |
|---|---|---|
| pk_t+1 | +0.000736 | +0.000972 |
| pk_t+5 | +0.000807 | +0.001827 |
| pk_t+10 | −0.000707 | +0.001108 |
| pk_t+22 | +0.000928 | +0.001171 |

- Spike classification: News PR-AUC ≤ Price PR-AUC — **news doesn't help**
- Two-stage abnormal vol: Both steps negative
- **Conclusion:** No horizon or target transformation recovers signal

### Epic 22 — Final Decision

| Case | Decision |
|---|---|
| T+1 all regimes | **DROP** |
| T+1 high-vol | **DROP** |
| T+5 high-vol | **DROP** |
| T+10 high-vol | **DROP** |
| T+22 high-vol | **DROP** |
| Sensitive tickers | **CONDITIONAL** |
| Volatility spike | **DROP** |
| **Production deployment** | **NO** |

**Stopping criteria score:** 0/10 (need ≥7 for production)

## Files changed/created

| File | Purpose |
|---|---|
| `src/modeling/regime.py` | Epic 19 — regime validation, threshold sensitivity, DM+block bootstrap+placebo |
| `src/modeling/conditional_model.py` | Epic 20 — HAR fallback + hard/soft gate + shrinkage |
| `src/modeling/horizon_analysis.py` | Epic 21 — horizon comparison + spike classification + two-stage |
| `src/modeling/final_decision.py` | Epic 22 — decision matrix + stopping criteria |
| `sprint-status.yaml` | Updated to close all epics |

## Output files generated

| File | Description |
|---|---|
| `eda_output/modeling/regime_validation_report.md` | Full Epic 19 report |
| `eda_output/modeling/regime_analysis.json` | JSON results |
| `eda_output/modeling/threshold_sensitivity_*.csv` | Per-target threshold analysis (4 files) |
| `eda_output/modeling/cross_analysis_*.csv` | Sensitive × regime cross (4 files) |
| `eda_output/modeling/walk_forward_*.csv` | Walk-forward per fold (4 files) |
| `eda_output/modeling/conditional_model_report.md` | Epic 20 report |
| `eda_output/modeling/horizon_expansion_report.md` | Epic 21 report |
| `eda_output/modeling/horizon_comparison.csv` | Multi-horizon comparison |
| `eda_output/modeling/volatility_spike_classification.csv` | Spike PR-AUC/ROC-AUC |
| `eda_output/modeling/abnormal_volatility_results.json` | Two-stage results |
| `eda_output/modeling/final_decision_report.md` | Final decision |
| `eda_output/modeling/final_decision.json` | Decision matrix JSON |

## Code review findings fixed

| Issue | File | Fix |
|---|---|---|
| CRITICAL: None crash (C3, C4) | `significance.py` | Guard `dm_pvalue is None`, guard `delta_r2_ci` |
| MAJOR: permuted_r2 label (M3) | `significance.py` | Renamed to `permuted_r2_mean`, separate from `drop` |
| CRITICAL: TimeSeriesSplit leak (C1) | `residual_model.py` | Sort by date only (not ticker) |
| CRITICAL: None guard (C2) | `residual_model.py` | `_safe_delta` handles None |
| MAJOR: first-obs novelty=1.0 (M1) | `features.py` | Initialize ewma_hist with first embedding |
| MAJOR: NaN decay (M2) | `features.py` | Apply (1-alpha) decay during NaN gaps |
| MINOR: dead code (m1) | `features.py` | Removed `_emb_norm` |
| MAJOR: dedup (M4) | `moe.py` | `dict.fromkeys()` for news_feats |
| MINOR: empty guard (m2) | `moe.py` | Guard empty results in aggregate |
| MINOR: silent neutral (m4) | `ticker_clusters.py` | NaN for insufficient data |
| MINOR: cache quantiles (m5) | `per_ticker_eval.py` | Cache q33/q66 |
| MINOR: idxmax edge case | `per_ticker_eval.py` | Guard `notna().any()` |
| MINOR: HURTS label (m6) | `baseline.py` | Changed "no effect" to "HURTS" |
| MINOR: tz-aware (tz) | `dataset.py` | `.dt.tz_localize(None)` |

## Pre-existing issues (not fixed)

- Test `test_real_run_models_smoke` expects only 3 feature sets (pre-dates Epics 14/16/17 additions)
- `dcor` module not installed (3 tests skip)
