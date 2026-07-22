# Story 15.1 — Extended Horizon: Add pk_t+22

**Epic:** 15 (Extended Horizon)
**Story key:** `15-1-extended-horizon-pk22`
**Status:** in-progress

## Context

Current pipeline only computes forward Parkinson volatility targets at 1, 5, 10 trading days (pk_t+1, pk_t+5, pk_t+10). Add pk_t+22 (~1 calendar month) to all stages: feature engineering, modeling, EDA phases, and dashboard. This allows evaluating whether news signals have predictive power at longer horizons.

## Requirements (Acceptance Criteria)

- [ ] `src/eda/phase03_price_eda.py`: Add `22` to `TARGET_HORIZONS` tuple → auto-generates `pk_t+22` and `rv_t+22` columns
- [ ] `src/modeling/dataset.py`: Add `"pk_t+22"` to `TARGETS` list
- [ ] `src/eda/phase05_relationship.py`: Add `"pk_t+22"` and `"rv_t+22"` to `TARGETS` list
- [ ] `src/eda/phase08_feature_validation.py`: Add `"pk_t+22"` and `"rv_t+22"` to `TARGETS` list
- [ ] `src/eda/phase09_leakage.py`: Add `"pk_t+22"` and `"rv_t+22"` to `TARGETS` list
- [ ] `src/dashboard/app.py`: Add `"pk_t+22"` and `"rv_t+22"` to display column list and update description text
- [ ] `tests/unit/test_phase03_price_eda.py`: Update assertions to include `pk_t+22` and `rv_t+22`
- [ ] Re-run pipeline: `python -m src.eda.phase03_price_eda` to regenerate parquet files with new columns
- [ ] Verify: dashboard shows pk_t+22 data; no regressions

## Files to modify

| File | Change |
|------|--------|
| `src/eda/phase03_price_eda.py:27` | `TARGET_HORIZONS = (1, 5, 10, 22)` |
| `src/modeling/dataset.py:28` | Add `"pk_t+22"` to TARGETS |
| `src/eda/phase05_relationship.py:28` | Add `"pk_t+22", "rv_t+22"` to TARGETS |
| `src/eda/phase08_feature_validation.py:29` | Add `"pk_t+22", "rv_t+22"` to TARGETS |
| `src/eda/phase09_leakage.py:23` | Add `"pk_t+22", "rv_t+22"` to TARGETS |
| `src/dashboard/app.py:110` | Add `"pk_t+22", "rv_t+22"` to column list |
| `src/dashboard/app.py:113` | Update description to mention pk_t+22 |
| `tests/unit/test_phase03_price_eda.py:109,118` | Add pk_t+22, rv_t+22 assertions |

## Verification

1. `uv run pytest tests/unit/test_phase03_price_eda.py -v` — all pass
2. `uv run streamlit run src/dashboard/app.py` — pk_t+22 visible on Price EDA page
3. No other regressions (`uv run pytest tests/ -q --tb=short`)
