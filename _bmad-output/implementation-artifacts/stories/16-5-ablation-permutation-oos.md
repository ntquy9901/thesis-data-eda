# Story 16.5 — Ablation + Permutation Importance + OOS Evaluation

**Epic:** 16 (Advanced News Signal with Embedding + Ticker-Specific Models)
**Story key:** `16-5-ablation-permutation-oos`
**Status:** backlog

## Context

Currently all news features are kept in the model regardless of their contribution. There is no systematic feature importance measurement and no out-of-sample (OOS) holdout. This makes it impossible to know which news features actually matter, and risks overfitting to the 2023-2025 training period. We need a rigorous feature evaluation methodology: permutation importance to measure each feature's marginal contribution, and a held-out 2026H1 OOS period as the final evaluation set that is never used in train/test splits.

## Requirements (Acceptance Criteria)

- [ ] `src/modeling/significance.py`: Add `permutation_importance(model, X_test, y_test, features, n_repeat=10, random_state=42)`
  - For each feature: shuffle values in test set, measure ΔR² drop vs unshuffled baseline
  - Return `pd.DataFrame` with columns: `feature | baseline_r2 | permuted_r2 | drop | drop_std`
- [ ] OOS evaluation: hold out `2026-01-01` to end of available data as final OOS period
  - OOS data is NOT used in any train/test split or hyperparameter tuning
  - Only used once at the end to evaluate the final selected model
- [ ] Ablation: for each news feature group (emb_*, ewma_*, topic counts, cluster features), drop the entire group and measure ΔR²
  - Groups to test: `kq_emb` (32 columns), `th_emb` (32), `ewma_kq_emb` (32), `ewma_th_emb` (32), `topic_counts`, `ticker_cluster`, `emb_norm` (2)
  - Report grouped ablation alongside per-feature permutation importance
- [ ] Summary table output: `eda_output/modeling/news_feature_ranking.csv`
  - Columns: `feature_group | feature_name | ΔR² (ablation) | permutation_drop | perm_drop_std | keep/drop`
  - Keep decision: if both ablation ΔR² < 0.001 AND permutation_drop < 0.001 → "drop", else "keep"
- [ ] Command-line entry point: `python -m src.modeling.significance --full` runs all steps
- [ ] Unit tests:
  - Permutation importance: known feature with zero importance should produce drop ≈ 0
  - OOS split: no OOS dates leak into train/test
  - Ablation: dropping all features should reduce R² to ~0 (or noise floor)
  - Summary table format and keep/drop logic

## Files to modify

| File | Change |
|------|--------|
| `src/modeling/significance.py:200-280` | Add `permutation_importance()` function |
| `src/modeling/significance.py:280-350` | Add `ablation_analysis()` function (drop feature groups) |
| `src/modeling/significance.py:350-420` | Add `oos_evaluate()` function (holdout split, final eval) |
| `src/modeling/significance.py:420-480` | Add `run_full_analysis()` orchestration + `--full` CLI |
| `src/modeling/significance.py:1-50` | Add argument parser for `--full`, `--n-repeat`, `--oos-start` |
| `tests/unit/test_significance.py` | Add test classes `TestPermutationImportance`, `TestOOS`, `TestAblation` |

## Dependencies

- Story 16-1 (Dual-Group Embedding Features) — full `ADV_FEATURES_DUAL` feature set
- Story 16-2 (EWMA Embedding) — EWMA feature group
- Story 16-3 (Ticker Clustering) — ticker cluster feature group
- `src/modeling/baseline.py` — `run_baseline()` for trained models
- `src/modeling/dataset.py` — `load_modeling_data()` with date filtering

## Technical Notes

- OOS period must be strictly future relative to all training data. Verify no data leak: `max(train/val dates) < min(oos dates)`.
- Permutation importance is computed on the test set (not OOS) to avoid overinterpreting OOS results.
- Ablation drops groups independently (not cumulatively) to measure each group's marginal contribution.
- The keep/drop threshold (0.001 ΔR²) corresponds to ~0.1% improvement, below which the feature is practically useless.
- All random operations use `random_state=42` for reproducibility.

## Out of Scope

- SHAP values — permutation importance is sufficient for linear Ridge models
- Feature selection based on ranking (recommendations only, no automatic removal)
- Retraining without dropped features — keep/drop recommendations are advisory
- OOS evaluation of MoE (Story 16-4) — price+news Ridge only
- Dashboard integration

## Verification

1. `uv run python -m src.modeling.significance --full` — generates `eda_output/modeling/news_feature_ranking.csv`
2. Inspect ranking CSV: verify all feature groups present, drop values non-NaN
3. Verify OOS split: `python -c "import pandas as pd; df=pd.read_csv('eda_output/modeling/news_feature_ranking.csv'); assert not df.empty"`
4. `uv run pytest tests/unit/test_significance.py -v -k "TestPermutationImportance or TestOOS or TestAblation"` — all pass
5. `uv run pytest tests/unit --cov=src --cov-report=xml -q` — coverage gate pass
