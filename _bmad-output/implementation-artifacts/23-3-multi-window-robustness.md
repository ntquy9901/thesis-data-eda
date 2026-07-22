# Story 23.3: Multi-window Robustness for Per-ticker RF

Status: ready-for-dev

## Story

As a data scientist,
I want to evaluate per-ticker RF on 3-year and 5-year train windows with all feature sets,
so that I can verify that per-ticker news contribution conclusions are robust to training window length.

## Acceptance Criteria

1. RF (200 trees, max_depth=10, random_state=0) trained per ticker on:
   - **3-year window**: 2023-01-02 to 2025-12-31, test: Jan 2026 (2026-01-02 to 2026-01-31)
   - **5-year window**: 2021-01-04 to 2025-12-31, test: Jan 2026
2. Same 4 feature sets as Story 23-2 (price_only, price+news_basic, price+news_adv_dual, price+news_adv_full)
3. Full metrics per ticker x feature_set x window x target: R², RMSE, MAE, QLIKE, DirAcc, MAPE, Theil's U, Pearson r, Spearman r
4. Output: `results/modeling/per_ticker_rf_window_{window}yr_{target}.csv` — columns: [ticker, feature_set, r2, rmse, mae, qlike, dir_acc, mape, theils_u, pearson_r, spearman_r, delta_r2]
5. **Cross-window consistency report**:
   - For each ticker: does delta_r2 sign (positive/negative) agree across all 3 windows (1yr, 3yr, 5yr)?
   - Correlation of delta_r2 across windows (Pearson between 1yr vs 3yr, 1yr vs 5yr, 3yr vs 5yr)
   - Tickers where delta_r2 is consistently positive vs consistently negative vs inconsistent
6. Targets: pk_t+1 (primary), pk_t+5, pk_t+10, pk_t+22

## Tasks / Subtasks

- [ ] Add `evaluate_window()` to `per_ticker_rf.py` accepting `train_start` and `train_end` params
  - [ ] Loop over windows: 1yr, 3yr, 5yr
  - [ ] Reuse feature set evaluation from 23-2
- [ ] Generate cross-window consistency report as markdown
- [ ] Verify: 30 tickers x 4 feature_sets x 3 windows x 4 targets = 1440 rows across all CSVs

## Dev Notes

- **Extend** `src/modeling/per_ticker_rf.py` from Stories 23-1/23-2
- **Window definitions** (exact dates):
  - 1yr: train_start=2025-01-02, train_end=2025-12-31
  - 3yr: train_start=2023-01-02, train_end=2025-12-31
  - 5yr: train_start=2021-01-04, train_end=2025-12-31
- **Test window fixed**: Jan 2026 for all (to isolate train window as the variable)
- **Consistency report**: saved as `results/modeling/per_ticker_rf_window_consistency.md`
- **Potential issue**: 5-year window for tickers with less history (e.g. some VN30 tickers listed after 2021). Verify n_train >= 100 trading days per ticker; flag otherwise.
- **Performance**: 30 tickers x 3 windows x 4 feature sets x 4 targets = up to 1,440 RF fits. Each fit on ~250-1250 rows. Expect < 30 min total.

### References

- Story 23-2: `src/modeling/per_ticker_rf.py::evaluate_feature_sets()`
- Ticker listing dates: available in panel.parquet per ticker date range
- Panel: `eda_output/modeling/panel.parquet`

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List

- `src/modeling/per_ticker_rf.py` (modify — add window support)
- `results/modeling/per_ticker_rf_window_1yr_pk_t+1.csv`
- `results/modeling/per_ticker_rf_window_3yr_pk_t+1.csv`
- `results/modeling/per_ticker_rf_window_5yr_pk_t+1.csv`
- `results/modeling/per_ticker_rf_window_consistency.md`
