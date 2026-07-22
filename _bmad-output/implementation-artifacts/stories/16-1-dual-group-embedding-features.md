# Story 16.1 — Dual-Group Embedding Features

**Epic:** 16 (Advanced News Signal with Embedding + Ticker-Specific Models)
**Story key:** `16-1-dual-group-embedding-features`
**Status:** in-progress

## Context

Current pipeline only uses `tong_hop` (securities firms) group for embedding features (`emb_0..31`). The market-level EDA found `khach_quan` (mainstream press) has stronger volatility signal. Need to process both groups and add L2-normalized embedding vectors as modeling features.

The `tong_hop` group aggregates all news sources into a single `news_articles.csv`, while `khach_quan` contains per-source raw articles (`cafef`, `ssi`, `vndirect`, `vnstock`, `hsc`). Both groups already have pre-computed PhoBERT embedding caches from Epic 11 (`data/features/news_emb_articles_tong_hop.parquet`, `data/features/news_emb_articles_khach_quan.parquet`). The gap is in the modeling feature pipeline: `ADV_FEATURES` only pulls `tong_hop` embeddings.

## Requirements (Acceptance Criteria)

- [ ] `src/modeling/features.py`: Modify `build_advanced_features()` to process both `"khach_quan"` and `"tong_hop"` groups
- [ ] Column naming: `kq_emb_0..31` (khach_quan), `th_emb_0..31` (tong_hop), `kq_emb_norm`, `th_emb_norm`
- [ ] `emb_norm` = L2 norm of the 32-dim mean-pooled embedding vector per (ticker, date): `sqrt(sum(emb_i^2))`
- [ ] New `ADV_FEATURES_DUAL` constant containing all dual-group features (64 embedding dims + 2 norms + existing topic counts)
- [ ] `src/modeling/baseline.py`: Update `FEATURE_SETS` with `"price+news_adv_dual"` key using `ADV_FEATURES_DUAL`
- [ ] Keep backward compat: existing `"price+news_adv"` unchanged (only `tong_hop`)
- [ ] Unit tests verifying:
  - Both groups are processed independently
  - Column naming convention is correct
  - `emb_norm` equals L2 norm of the 32-dim vector
  - Backward compat — old `"price+news_adv"` still works and uses only `tong_hop`
  - `ADV_FEATURES_DUAL` shape matches expected column count (64 + 2 + topic counts)

## Files to modify

| File | Change |
|------|--------|
| `src/modeling/features.py:280-320` | Add `ADV_FEATURES_DUAL` constant with `kq_emb_*`, `th_emb_*`, `kq_emb_norm`, `th_emb_norm` |
| `src/modeling/features.py:400-450` | Modify `build_advanced_features()` to load both groups, prefix columns, compute L2 norm |
| `src/modeling/features.py:500-520` | Add `_compute_emb_norm(group_emb_df)` helper for L2 norm |
| `src/modeling/baseline.py:85-95` | Add `"price+news_adv_dual"` to `FEATURE_SETS` using `ADV_FEATURES_DUAL` |
| `tests/unit/test_modeling_features.py` | Add test class `TestDualGroupEmbedding` |
| `tests/unit/test_modeling_baseline.py` | Add test for `price+news_adv_dual` feature set |

## Dependencies

- `ADV_FEATURES` (existing, unchanged) — single-group `tong_hop` features
- `build_group_embeddings(group)` from `src/features/news_embeddings.py` — already supports both groups

## Out of Scope

- EWMA aggregation (separate story 16-2)
- Ticker clustering (separate story 16-3)
- Changing the EDA phases — they keep their existing embedding references
- Dashboard updates — tracked separately if needed

## Verification

1. `uv run python -m src.modeling.features` — imports without error, `ADV_FEATURES_DUAL` defined
2. `uv run python -m src.modeling.baseline` — `price+news_adv_dual` listed in available feature sets
3. `uv run pytest tests/unit/test_modeling_features.py -v -k TestDualGroupEmbedding` — all pass
4. `uv run pytest tests/unit/test_modeling_baseline.py -v` — all pass
5. `uv run pytest tests/unit --cov=src --cov-report=xml -q` — coverage gate pass
