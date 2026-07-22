# Story 23.5: Final Synthesis Report

Status: ready-for-dev

## Story

As a data scientist,
I want to synthesize all per-ticker nonlinear analysis into a final report summarizing which tickers benefit from news, which windows/horizons show signal, and the final recommendation,
so that stakeholders can make an informed keep/drop decision on news features per ticker.

## Acceptance Criteria

1. **Ticker ranking table**: all 30 tickers ranked by price-only R² (1yr window, pk_t+1), grouped as:
   - High predictability (R² > 0.6): list tickers
   - Medium (0.3 < R² <= 0.6): list tickers
   - Low (R² <= 0.3): list tickers
2. **News contribution table**: per ticker, delta_r2 for each feature_set (1yr, pk_t+1). Highlight:
   - Consistently positive delta_r2 across all feature_sets
   - Positive only on some feature_sets
   - Consistently negative
3. **Window consistency**: per ticker, does delta_r2 sign agree across 1yr/3yr/5yr? Flag tickers with inconsistent sign.
4. **Horizon effect**: per target, mean delta_r2 across tickers. Does news help more at shorter horizons?
5. **Top-N tickers for news**: the 3-5 tickers with the strongest positive news contribution. Include detailed profile of each.
6. **Final recommendation per ticker**: KEEP news or DROP news (based on consistent positive delta_r2 AND economic significance > 0.01 R² improvement)
7. **Output**: `results/modeling/per_ticker_rf_synthesis_report.md`

## Tasks / Subtasks

- [ ] Generate synthesis report by reading all CSVs from Stories 23-1 through 23-4
  - [ ] Or integrate directly if run in sequence
- [ ] Ticker ranking by price-only R²
- [ ] News contribution analysis per ticker
- [ ] Cross-window consistency analysis
- [ ] Horizon effect analysis
- [ ] Top-N ticker profiles
- [ ] Final per-ticker keep/drop matrix
- [ ] Verify: report references actual data (no hallucinated numbers)

## Dev Notes

- **This story uses data from Stories 23-1 through 23-4** — it does NOT re-run models
- **Expected key finding** (from prior pooled analysis): PDR >> SAB > VHM > MBB > TCB most predictable on price-only; news delta_r2 ~ 0 or negative across the board; PDR may be the only exception where news adds marginal value
- **Keep threshold**: delta_r2 >= +0.01 AND consistent positive sign across windows AND economic significance (DirAcc improvement or QLIKE not exploding)
- **Per-ticker Keep/Drop matrix** should be in the same format as `final_decision.py` output for consistency
- **Report file** placed at `results/modeling/per_ticker_rf_synthesis_report.md`

### References

- Story 23-1 CSVs: `results/modeling/per_ticker_rf_baseline_{target}.csv`
- Story 23-2 CSVs: `results/modeling/per_ticker_rf_impact_{target}.csv`
- Story 23-3 CSVs: `results/modeling/per_ticker_rf_window_{window}yr_{target}.csv`
- Story 23-4 CSV: `results/modeling/per_ticker_rf_all_horizons.csv`
- Previous final decision: `results/modeling/final_decision_report.md`

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List

- `results/modeling/per_ticker_rf_synthesis_report.md` (new — report only)
