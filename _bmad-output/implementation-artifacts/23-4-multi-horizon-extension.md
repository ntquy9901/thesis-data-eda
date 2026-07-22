# Story 23.4: Multi-horizon Extension for Per-ticker RF

Status: ready-for-dev

## Story

As a data scientist,
I want to extend per-ticker RF evaluation to longer horizons (pk_t+5, pk_t+10, pk_t+22) across all feature sets and windows,
so that I can determine whether per-ticker news contribution varies by forecast horizon.

## Acceptance Criteria

1. Reuse all combinations from Stories 23-1/23-2/23-3 but evaluate ALL targets: pk_t+1, pk_t+5, pk_t+10, pk_t+22
2. Output: single consolidated CSV `results/modeling/per_ticker_rf_all_horizons.csv` — columns: [ticker, target, feature_set, window, r2, rmse, mae, qlike, dir_acc, mape, theils_u, pearson_r, spearman_r, delta_r2]
3. Horizon comparison report:
   - Mean delta_r2 per (target, feature_set, window) across all tickers
   - Heatmap-ready summary: pivot table with target as rows, feature_set as columns, cell = mean(delta_r2)
   - Per-ticker: correlation of delta_r2 across horizons (does a ticker responsive at t+1 stay responsive at t+22?)
4. **Key question answered**: "Does news help more at shorter or longer horizons?" — report the pattern

## Tasks / Subtasks

- [ ] Consolidate all per-ticker evaluation into a single run function
  - [ ] Loop: window x feature_set x target x ticker
  - [ ] Save consolidated CSV
- [ ] Generate horizon comparison markdown report
- [ ] Verify: 30 tickers x 3 windows x 4 feature_sets x 4 targets = 1,440 rows in consolidated CSV

## Dev Notes

- **Don't re-run everything from scratch** — consolidate existing code from 23-1/23-2/23-3 into one orchestrator function
- **Expected pattern from earlier pool results**: delta_r2 decreases at longer horizons (t+22 < t+1). Verify per-ticker.
- **pk_t+5 already evaluated at pool level in Epic 21** — this story adds the per-ticker dimension
- **Panel data already contains all target columns** — no need to regenerate
- **Output size**: ~1,440 rows, fine for CSV

### References

- Targets: `src/modeling/dataset.py::TARGETS`
- Previous horizon analysis (pooled): `src/modeling/horizon_analysis.py`
- Panel: `eda_output/modeling/panel.parquet`
- Stories 23-1/23-2/23-3

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List

- `src/modeling/per_ticker_rf.py` (modify — add consolidated runner)
- `results/modeling/per_ticker_rf_all_horizons.csv`
- `results/modeling/per_ticker_rf_horizon_comparison.md`
