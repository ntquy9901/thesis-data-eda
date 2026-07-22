# Story 16.2 — EWMA Embedding Aggregation

**Epic:** 16 (Advanced News Signal with Embedding + Ticker-Specific Models)
**Story key:** `16-2-ewma-embedding-aggregation`
**Status:** backlog

## Context

Market-level EDA showed EWMA(30d) centroid norm correlates at r=+0.27 with Parkinson volatility — a stronger signal than raw daily embeddings. The intuition: news topics shift gradually (e.g. a bank sector story persists across multiple days), and EWMA captures the evolving "news state" rather than treating each day independently. Apply the same logic per-ticker in the modeling feature pipeline.

Current daily mean-pooled embeddings per (ticker, date) are independent cross-sectionally. EWMA aggregation smooths them and creates a persistent news-state signal that can capture gradual sentiment/topic drift.

## Requirements (Acceptance Criteria)

- [ ] `src/modeling/features.py`: Add `ewma_embedding_features()` function
- [ ] For each ticker, compute EWMA (exponentially weighted moving average, half-life=30 trading days) on the 32-dim mean-pooled embedding vector
- [ ] Apply to both groups: `tong_hop` and `khach_quan`
- [ ] Output columns: `ewma_kq_emb_0..31`, `ewma_th_emb_0..31`
- [ ] EWMA formula: `ewma_t = α * x_t + (1-α) * ewma_{t-1}`, where `α = 1 - exp(-ln(2) / half_life)`
- [ ] Initial value (`ewma_0`) = first available embedding for that ticker (not zero-padded to avoid cold-start bias)
- [ ] Missing trading days: carry forward last EWMA value (no interpolation)
- [ ] Add columns to `ADV_FEATURES_DUAL` constant
- [ ] Update baseline.py `FEATURE_SETS` with `"price+news_adv_dual_ewma"` key
- [ ] Unit tests:
  - EWMA decay factor `α` matches half-life=30
  - Carry-forward behavior on missing days
  - Output shape matches input (N ticker-dates × 64 dims)
  - Works for both groups independently

## Files to modify

| File | Change |
|------|--------|
| `src/modeling/features.py:520-580` | Add `ewma_embedding_features(emb_df, half_life=30)` function |
| `src/modeling/features.py:280-320` | Add `ewma_kq_emb_*`, `ewma_th_emb_*` to `ADV_FEATURES_DUAL` |
| `src/modeling/features.py:400-450` | Call `ewma_embedding_features()` after mean-pooling, before column concatenation |
| `src/modeling/baseline.py:85-95` | Add `"price+news_adv_dual_ewma"` feature set |
| `tests/unit/test_modeling_features.py` | Add test class `TestEWMAEmbedding` |

## Dependencies

- Story 16-1 (Dual-Group Embedding Features) — `ADV_FEATURES_DUAL` constant must exist
- `pandas.DataFrame.ewm` (uses `halflife` parameter, no external deps)

## Out of Scope

- Multi-half-life EWMA sweep (e.g. 15d, 60d) — use 30d only unless evidence shows otherwise
- Ticker-cluster-aware EWMA (separate story 16-3)
- Windowed rolling mean alternative — EWMA only per spec
- Changes to EDA phases or dashboard

## Verification

1. `uv run python -m src.modeling.features` — `ewma_embedding_features()` callable, `ADV_FEATURES_DUAL` includes EWMA columns
2. `uv run python -m src.modeling.baseline` — `price+news_adv_dual_ewma` listed in feature sets
3. `uv run pytest tests/unit/test_modeling_features.py -v -k TestEWMAEmbedding` — all pass
4. Verify EWMA decay: for a ticker with constant embedding `x`, ewma should converge to `x`; for a step change, verify `(ewma_t - x_new) / (ewma_{t-1} - x_new) = 1-α`
5. `uv run pytest tests/unit --cov=src --cov-report=xml -q` — coverage gate pass
