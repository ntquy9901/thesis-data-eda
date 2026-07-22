# Story 16.4 — Mixture-of-Experts (Gated Model)

**Epic:** 16 (Advanced News Signal with Embedding + Ticker-Specific Models)
**Story key:** `16-4-mixture-of-experts`
**Status:** backlog

## Context

News helps only a subset of tickers (the ~25% "sensitive" cluster from Story 16.3). Applying news features to insensitive tickers adds noise and degrades predictions. A naive approach of separate models per cluster is fragile because clusters are small. Instead, use a simple gating approach: train a price-only expert for all tickers, and a price+news expert that only activates for sensitive tickers via a per-ticker gate.

This is a lightweight "mixture of experts" simplified to the extreme: only 2 experts, soft gating by cluster assignment, no backpropagation of gates. The gate weight is deterministic from the cluster label, not learned jointly.

## Requirements (Acceptance Criteria)

- [ ] New file: `src/modeling/moe.py`
- [ ] `SimpleMoE` class:
  - 2 Ridge experts: `price_only_model` (trained on all tickers), `price_news_model` (trained on all tickers with dual-group embedding features)
  - `per_ticker_weights`: mapping from ticker to `[w_price, w_news]` where `w_news` = cluster soft assignment (1.0 for sensitive, 0.0 for insensitive, 0.5 for neutral)
  - `predict(ticker, X_price, X_news) -> y_pred`: weighted average of expert predictions
- [ ] Gate weight = soft cluster assignment from `ticker_clusters.json` (Story 16-3):
  - **sensitive**: `w_news = 1.0` (full news branch)
  - **neutral**: `w_news = 0.5` (blend)
  - **insensitive**: `w_news = 0.0` (price-only)
- [ ] `evaluate()`: train/test split (same as baseline), compute RMSE, R², ΔR² vs price-only baseline
- [ ] Comparison table output: `eda_output/modeling/moe_comparison.md` with per-ticker and aggregate metrics
- [ ] Integration: `python -m src.modeling.moe` runs full evaluation end-to-end
- [ ] Unit tests:
  - Gate weight assignment for each cluster
  - Prediction shape and range
  - Edge cases: all sensitive, all insensitive, empty cluster, single-ticker cluster
  - Deterministic reproducibility (same seed → same results)

## Files to modify

| File | Change |
|------|--------|
| `src/modeling/moe.py` | **New file** — `SimpleMoE` class, `evaluate()`, comparison table writer |
| `src/modeling/__init__.py` | Export `SimpleMoE` from package |
| `tests/unit/test_moe.py` | **New file** — test gating, predictions, comparison output |

## Dependencies

- Story 16-1 (Dual-Group Embedding Features) — `ADV_FEATURES_DUAL` features
- Story 16-3 (Ticker Clustering) — `ticker_clusters.json` input
- `src/modeling/baseline.py` — `run_baseline()` for price-only and price+news reference models
- `sklearn.linear_model.Ridge` — expert regressor

## Technical Notes

- The MoE is intentionally simple: no joint training, no gating network, no gradient flow. The gate is an interpretable deterministic function of cluster membership. This avoids overfitting on 30 tickers and keeps the baseline comparison clean.
- Experts are `Ridge(alpha=1.0)` matching the baseline pipeline. No hyperparameter search for experts (use baseline defaults).
- The comparison table includes: ticker | cluster | w_price | w_news | MoE_R² | baseline_price_R² | ΔR²_vs_baseline

## Out of Scope

- Learned gating network / softmax router — deterministic per spec
- More than 2 experts
- Neural network backbones
- Online / streaming training
- Dashboard integration

## Verification

1. `uv run python -m src.modeling.moe` — runs end-to-end, generates `eda_output/modeling/moe_comparison.md`
2. Inspect comparison table: verify sensitive tickers have `w_news=1.0`, insensitive have `w_news=0.0`
3. Verify MoE aggregate R² >= price-only baseline (or document regression if it occurs)
4. `uv run pytest tests/unit/test_moe.py -v` — all pass
5. `uv run pytest tests/unit --cov=src --cov-report=xml -q` — coverage gate pass
