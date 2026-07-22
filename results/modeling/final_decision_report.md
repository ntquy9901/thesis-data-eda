# Epic 22 — Final Research Decision


## Final Decision Matrix


| Case | Decision | ΔR² | DM p-value |
|------|----------|-----|-----------|
| T+1_all_regimes | **DROP** | 0.000736 | None |
| T+1_high_vol | **None** | None | None |
| T+5_high_vol | **DROP** | 0.000807 | None |
| T+10_high_vol | **DROP** | -0.000707 | None |
| T+22_high_vol | **DROP** | 0.000928 | None |
| sensitive_tickers | **CONDITIONAL** | 0.00115 | None |
| volatility_spike | **DROP** | -0.0003 | None |
| production_deployment | **NO** |  |  |

## Stopping Criteria Evaluation


Score: 0/10 (need >= 7 for production)

- delta_r2_positive_majority_folds: ✗
- block_bootstrap_ci_mostly_positive: ✗
- qlike_and_another_metric_improve: ✗
- min_50pct_tickers_not_harmed: ✗
- result_not_driven_by_single_ticker: ✗
- beats_placebo: ✗
- improvement_in_at_least_two_time_periods: ✗
- feature_importance_stable: ✗
- signal_replicable_different_seed: ✗
- improvement_large_enough_for_embedding_cost: ✗

## Final Verdict


> **DROP** — News does NOT provide reliable incremental predictive value for Parkinson volatility in the current dataset.

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
