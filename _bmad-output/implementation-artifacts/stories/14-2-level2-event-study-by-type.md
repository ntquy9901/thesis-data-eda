# Story 14.2 — Level-2 Event Study segmented by event type (+ CAR)

**Epic:** 14 (Level 1/2 Feature Evaluation per docs/gpt-guide guideline)
**Story key:** `14-2-level2-event-study-by-type`
**Status:** done

## Context
`docs/gpt-guide/news_feature_evaluation_guideline.md` Level 2 requires "treat each event
independently" (earnings, dividend, M&A, ... — not pooled by generic |sentiment| magnitude as
Phase 6 does) over a T-5..T0..T+10 window, with average volatility, average return, abnormal
return, and Cumulative Abnormal Return (CAR). Phase 6 (`phase06_event_study.py`) only had
abnormal volatility — no abnormal return, no CAR, no per-type segmentation.

## Requirements (Acceptance Criteria)
- [x] `src/eda/phase18_event_study_by_type.py`: reuses Phase 6's `window_mean`/`window_sum`/
  `HORIZONS` (no reimplementation).
- [x] `market_benchmark_returns()`: equal-weighted mean log_return across all EDA_TICKERS per
  date — the abnormal-return benchmark (no separate VN-Index file confirmed available;
  documented assumption).
- [x] `event_type_window_metrics()`: adds `pre_car`/`post_car` (cumulative abnormal return, sum
  of `log_return - market_return` over the window) alongside the existing abnormal-vol/return
  metrics.
- [x] `event_days_by_type()`: per (ticker, event_type) distinct event dates from the Story-14-1
  per-article event-type flags (exploded by mentioned ticker) — NOT top-|sentiment| selection;
  every day with >=1 article of that type counts as an event.
- [x] Per (event_type, horizon): one-sample t-test on abnormal_vol AND post_car (mean ≠ 0?).
- [x] Dashboard: new "Event Study by Type" page (event-type + horizon selectors, t-test
  significance badges, abnormal-vol histogram).
- [x] Unit tests for all pure helpers + real-data smoke test.

## Code-review fix (found during self-verification, not by /bmad-code-review)
- **[High]** `_one_sample_ttest` crashed with `AttributeError` on real data: `abnormal_vol`
  column becomes `dtype=object` when `window_mean`/`window_sum` return `None` (not `NaN`) at
  series edges, and scipy's `ttest_1samp` cannot handle object dtype. Fixed:
  `pd.to_numeric(values, errors="coerce").dropna()` before the test.

## Out of Scope
- CAR against a true market-cap-weighted VN-Index (no such file confirmed in
  `stock_vol_prediction01/data/raw`) — equal-weighted VN30 average used instead, documented.
- Statistical correction for event-window overlap across tickers/dates (guideline flags this as
  exploratory-only for similar autocorrelated features elsewhere in the codebase).

## Verify
```bash
uv run pytest tests/unit/test_phase18_event_study_by_type.py -q
uv run python -m src.eda.phase18_event_study_by_type
```
Definition of Done: acceptance boxes checked, diff-coverage gate, `/bmad-code-review` addressed.
