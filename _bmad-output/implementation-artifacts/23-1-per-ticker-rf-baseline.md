# Story 23.1: Per-ticker RF Baseline

Status: done
baseline_revision: 0472e800e044b0ba48ecf70035c9ee3a9db437f7

## Story

As a data scientist,
I want to evaluate Random Forest per ticker using only price features on a 1-year train / Jan 2026 test window,
so that I have a per-ticker nonlinear baseline to compare against price+news models.

## Acceptance Criteria

1. RF (200 trees, max_depth=10, random_state=0) trained per ticker on 1 year train window (2025-01-02 to 2025-12-31), tested on Jan 2026 (2026-01-02 to 2026-01-31)
2. Full metrics computed per ticker: R², RMSE, MAE, QLIKE, DirAcc, MAPE, Theil's U, Pearson r, Spearman r
3. Output saved to `results/modeling/per_ticker_rf_baseline_{target}.csv` with columns: [ticker, r2, rmse, mae, qlike, dir_acc, mape, theils_u, pearson_r, spearman_r, n_train, n_test]
4. Targets: pk_t+1 (primary), pk_t+5, pk_t+10, pk_t+22
5. Feature set: PRICE_FEATURES (har_daily, har_weekly, har_monthly, atr_14, realized_vol_5d, realized_vol_20d) — **no news features**
6. Summary statistics reported: mean, std, min, max, median across 30 tickers for each metric

## Tasks / Subtasks

- [x] Add per-ticker RF evaluation function in `src/modeling/per_ticker_rf.py` (new module)
  - [x] Wire train/test split by ticker with configurable train_window_days and test_period
  - [x] Implement full metrics computation (extend `compute_metrics` from baseline.py)
  - [x] Save results CSV + summary statistics
- [x] Verify: run and confirm 30 tickers x 4 targets = 120 rows in output, no NaN metrics

## Dev Notes

- **New module**: create `src/modeling/per_ticker_rf.py` (don't modify baseline.py)
- **RF config**: `RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)`
- **Train window**: exactly 1 calendar year before test start. For Jan 2026 test: train = 2025-01-02 to 2025-12-31
- **Data source**: use `panel.parquet` from `eda_output/modeling/panel.parquet`
- **Metrics to add** beyond baseline's rmse/mae/r2/qlike/dir_acc:
  - MAPE: `mean(abs((y_true - y_pred) / y_true)) * 100`
  - Theil's U: `sqrt(mean((y_pred - y_true)^2)) / (sqrt(mean(y_pred^2)) + sqrt(mean(y_true^2)))`
  - Pearson r: `pearsonr(y_true, y_pred).statistic`
  - Spearman r: `spearmanr(y_true, y_pred).statistic`
- **Import pattern**: follow `per_ticker_eval.py` (Story 18-3) for per-ticker looping structure
- **Targets**: `TARGETS = ["pk_t+1", "pk_t+5", "pk_t+10", "pk_t+22"]` from dataset.py
- **Tickers**: VN30 list in `src/modeling/dataset.py:TICKERS`

### References

- Panel data: `eda_output/modeling/panel.parquet`
- Existing per-ticker pattern: `src/modeling/per_ticker_eval.py` (uses Ridge, but per-ticker loop pattern is reusable)
- Metrics function: `src/modeling/baseline.py::compute_metrics()`
- Feature constants: `src/modeling/dataset.py::PRICE_FEATURES`
- Existing output: `results/modeling/per_ticker_rf_baseline_pk_t+1.csv`

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List

- `src/modeling/per_ticker_rf.py` (new)
- `results/modeling/per_ticker_rf_baseline_pk_t+1.csv`
- `results/modeling/per_ticker_rf_baseline_pk_t+5.csv`
- `results/modeling/per_ticker_rf_baseline_pk_t+10.csv`
- `results/modeling/per_ticker_rf_baseline_pk_t+22.csv`
