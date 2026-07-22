---
project_name: 'data_eda'
date: '2026-07-22'
---

# Project Context for AI Agents — Vietnamese Stock Volatility & News Analysis

## Research Conclusion (as of 2026-07-22)

**Vietnamese news does NOT provide stable incremental predictive value for Parkinson volatility.** All Epics 16-22 complete. Final decision: DROP news branch. Production deployment: NO.

Key evidence:
- ΔR² < 0.001 across all horizons (T+1, T+5, T+10, T+22)
- DM p > 0.05 — not statistically significant
- Block bootstrap CI includes zero — not robust
- Placebo time-shift beats real signal
- 0/3 walk-forward folds positive
- Conditional model (HAR fallback + gate): ΔR² ≈ 0
- Spike classification: News PR-AUC ≤ Price PR-AUC
- Stopping criteria score: 0/10 (need ≥7)

## Technology Stack
- Python 3.10+, uv package manager
- Core: pandas, numpy, scikit-learn, scipy
- Modeling: Ridge regression, SimpleImputer, Pipeline, StandardScaler
- NLP: underthesea, sentence-transformers (PhoBERT), PCA for embeddings
- No deep learning (LSTM/Transformer) — complexity doesn't help weak signal
- Visualization: matplotlib, seaborn (dashboard)
- Testing: pytest, pytest-cov

## Critical Implementation Rules

### Data Sources (MANDATORY — never deviate)
1. **News**: `D:\bmad-projects\crawl_data\data` ONLY
2. **Price**: `D:\bmad-projects\stock_vol_prediction01\data\raw` ONLY
3. Never write to source directories
4. All output goes to `data/processed/` or `eda_output/`

### Modeling Rules
1. **NO look-ahead**: Expanding window percentile for regime, never full-dataset
2. **Time-based split only**: train < 2025-01-01, test >= 2025-01-01
3. **Preprocessing fit on TRAIN only**: impute + scale inside each fold
4. **HAR baseline always**: har_daily (1d), har_weekly (5d), har_monthly (22d)
5. **Parkinson vol targets**: pk_t+1, pk_t+5, pk_t+10, pk_t+22
6. **Use Ridge(alpha=1.0) as default**, HistGradientBoosting as nonlinear check
7. **Max 30 features** for news — 523 features causes overfitting (ΔR² = −0.04)
8. **Shared PCA basis** for dual-group embeddings (not per-group PCA)

### Code Quality
1. Type hints + pathlib for all new code
2. No bare `except`, no mutable default args
3. Clean Code rules in CLAUDE.md (22 rules) — enforce on all changes
4. BMAD workflow: sprint → story → task → dev/review → done
5. Code review REQUIRED: Blind Hunter + Edge Case Hunter + Acceptance Auditor
6. Coverage gate: diff-cover --fail-under=100 on changed lines
7. Every change needs a test; data pipeline tests need real-data-sample smoke

### Key File Locations
- `src/modeling/` — all modeling modules (baseline, regime, conditional, horizon, final_decision)
- `src/modeling/regime.py` — Epic 19: regime validation, DM test, block bootstrap, placebo
- `src/modeling/conditional_model.py` — Epic 20: HAR fallback + gate
- `src/modeling/horizon_analysis.py` — Epic 21: multi-horizon, spike, two-stage
- `src/modeling/final_decision.py` — Epic 22: decision matrix
- `eda_output/modeling/` — all generated results (JSON, CSV, MD reports)
- `_bmad-output/` — sprint status, story files
- `docs/gpt-guide/` — research guides (huong_cai_thien, lo_trinh)

### Know Known Issues
- `test_real_run_models_smoke` expects only 3 feature sets (pre-dates Epics 14/16/17)
- `dcor` module not installed (3 tests skip)
- LF/CRLF warnings on Windows — cosmetic only
