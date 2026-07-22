# Summary: Extended Horizon pk_t+22

**Date:** 2026-07-21 23:30
**Epic/Story:** 15 / 15.1 — Extended Horizon pk_t+22
**Status:** ✅ Done

## Changes Made

| File | Change |
|------|--------|
| `src/eda/phase03_price_eda.py:27` | `TARGET_HORIZONS = (1, 5, 10, 22)` — auto-generates `pk_t+22` + `rv_t+22` |
| `src/modeling/dataset.py:28` | Added `"pk_t+22"` to TARGETS |
| `src/eda/phase05_relationship.py:28` | Added `"pk_t+22", "rv_t+22"` to TARGETS |
| `src/eda/phase08_feature_validation.py:29` | Added `"pk_t+22", "rv_t+22"` to TARGETS |
| `src/eda/phase09_leakage.py:23` | Added `"pk_t+22", "rv_t+22"` to TARGETS |
| `src/dashboard/app.py:110,113` | Added `pk_t+22`, `rv_t+22` to display; updated description |
| `tests/unit/test_phase03_price_eda.py:109,118` | Updated assertions for `pk_t+22`, `rv_t+22` |

## Test Results

- **21/21 passed** — test_phase03_price_eda.py (including new assertions)
- **66/69 passed** — related test suite (3 pre-existing failures: missing `dcor` module)
- **0 regressions** from this change

## BMAD Artifacts

- Sprint status updated: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Story file: `_bmad-output/implementation-artifacts/stories/15-1-extended-horizon-pk22.md`
- Epic 15 added to: `_bmad-output/planning-artifacts/epics.md`

## Next Steps

1. Re-run `src.eda.phase03_price_eda` to regenerate parquet files with `pk_t+22` column
2. Re-run downstream phases (phase05, phase08, phase09) for complete data
3. Re-run modeling pipeline to include pk_t+22 in model evaluation
