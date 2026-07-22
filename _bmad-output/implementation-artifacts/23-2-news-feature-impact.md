# Story 23.2: News Feature Impact on Per-ticker RF

Status: ready-for-dev

## Story

As a data scientist,
I want to evaluate per-ticker RF with incremental news feature groups on 1-year train / Jan 2026 test,
so that I can measure which news features contribute (or hurt) per-ticker Parkinson volatility prediction.

## Acceptance Criteria

1. RF (200 trees, max_depth=10, random_state=0) trained per ticker on 1-year (2025) train, tested on Jan 2026
2. **4 feature sets** tested, each building on price-only baseline:
   - `price+news_basic`: price + [news_count_1d, news_count_3d, news_count_5d, days_since_last_news, sentiment_mean]
   - `price+news_adv_dual`: price + basic + kq_emb_0..31 + th_emb_0..31 + kq/th_emb_norm + kq/th_topic_*_count
   - `price+news_adv_dual_ewma30`: above + ewma_kq_emb_0..31 + ewma_th_emb_0..31 + ewma_norm
   - `price+news_adv_full`: all ~493 features including multi-ewma, novelty, dispersion, shock
3. Full metrics per ticker per feature set: R², RMSE, MAE, QLIKE, DirAcc, MAPE, Theil's U, Pearson r, Spearman r
4. Output: `results/modeling/per_ticker_rf_impact_{target}.csv` — columns: [ticker, feature_set, r2, rmse, mae, qlike, dir_acc, mape, theils_u, pearson_r, spearman_r, delta_r2]
5. **Delta R²** computed per ticker: `delta_r2 = r2(feature_set) - r2(price_only)`
6. Summary: mean delta R² per feature set across 30 tickers, plus count of tickers improved/hurt
7. Targets: pk_t+1 (primary), pk_t+5, pk_t+10, pk_t+22

## Tasks / Subtasks

- [ ] Extend `per_ticker_rf.py` with `evaluate_feature_sets()` function
  - [ ] Define feature set columns using existing constants from features.py
  - [ ] Loop per ticker x feature_set x target
  - [ ] Compute delta_r2 vs price-only baseline
- [ ] Generate summary table: mean(delta_r2) per feature_set per target
- [ ] Verify: 30 tickers x 4 feature_sets x 4 targets = 480 rows

## Dev Notes

- **Extend Story 23-1's module**: add to `src/modeling/per_ticker_rf.py`
- **Feature set column lists** should reference constants from `src/modeling/features.py` and `src/modeling/dataset.py`
- **QLIKE careful**: with many embedding features, QLIKE can explode (observed in previous Ridge experiments). Log any QLIKE > 1e6 as warnings.
- **RF handles NaN natively** — no imputer needed; but verify no hidden leakage (e.g. future data in EWMA features)
- **Train/test split**: same as Story 23-1 (1-year train 2025-01-02 to 2025-12-31, test 2026-01-02 to 2026-01-31)

### References

- Feature constants: `src/modeling/features.py` — ADV_FEATURES_DUAL, EWMA_FEATURES, ADV_FEATURES_DUAL_FULL
- Basic news features: `src/modeling/dataset.py::NEWS_FEATURES`
- Panel: `eda_output/modeling/panel.parquet`
- Story 23-1 module: `src/modeling/per_ticker_rf.py`

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List

- `src/modeling/per_ticker_rf.py` (modify — add evaluate_feature_sets)
- `results/modeling/per_ticker_rf_impact_pk_t+1.csv`
- `results/modeling/per_ticker_rf_impact_pk_t+5.csv`
- `results/modeling/per_ticker_rf_impact_pk_t+10.csv`
- `results/modeling/per_ticker_rf_impact_pk_t+22.csv`
