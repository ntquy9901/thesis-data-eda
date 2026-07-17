# Story 12.3 — Temporal decay / half-life weighting of embedding features

**Epic:** 12 (Advanced News Signal Extensions)
**Story key:** `12-3-temporal-decay-features`
**Status:** done

## Context
From `reports/2026-07-15_news_datamining_research.md` §7 (top-3 recommendation #3): the project's
strongest finding is that news matters at T+10 but not T+1/T+5 — consistent with slow information
diffusion. Current features only aggregate SAME-DAY news (flat window). Decay weighting
(`w = λ^d`, d = days old) tests whether a multi-day, decayed-weighted news signal explains the
T+10-only pattern better than flat same-day aggregation.

## Requirements (Acceptance Criteria)
- [x] `src/features/news_embeddings.py`: `decayed_embedding_features(group="tong_hop", halflife_days=5, lookback_days=20) -> pd.DataFrame`
  — per (ticker, TRADING DAY, not just article-publish days): exponentially-decayed weighted
  mean of `raw_*` embeddings over the preceding `lookback_days`, weight
  `w_i = 0.5 ** (age_i / halflife_days)`, normalized to sum to 1 across contributing articles
  (trading day with zero contributing articles in the lookback → not emitted → NaN on reindex,
  consistent with the existing no-news-NaN rule). `halflife_days <= 0` raises `ValueError`.
- [x] PCA-reduce the decayed embeddings the same way as the flat features (own basis; the
  fallback for <2 train rows keeps the honest `emb_*`-named raw embedding, per Story 11-1's
  round-2 fix — same mechanism reused here).
- [x] Add as NEW columns (`emb_decay_0..emb_decay_{dim-1}`) — additive alongside existing
  `ADV_FEATURES`, not replacing them (fixed single `halflife_days=5` default; per-horizon λ
  grid-search is explicitly out of scope).
- [x] Validate: `src/eda/phase15_temporal_decay_correlation.py` correlates `emb_decay_*` against
  all `phase05_relationship.TARGETS` the same way Story 11-3 did for the flat embedding — see
  Code-review fixes below for the honest caveat on how to read the comparison.
- [x] Unit tests: decay-weight computation (verified pre-PCA by bypassing `_reduce`: weights are
  exactly `0.5**(age/halflife)` normalized), zero-contribution days absent, `halflife_days<=0`
  raises, real-data-slice smoke test for the correlation comparison.

## Code-review fixes (bmad-code-review 3-layer)
- **[Critical — the core bug]** The original implementation only emitted a decayed value on
  days the ticker had its OWN same-day article (`unique_dates = np.unique(dates)` derived from
  article dates, not the trading calendar). This silently collapsed the "multi-day decayed
  signal" back to same-day-only coverage for sparse-news tickers — defeating the story's actual
  research question. Fixed: the date grid is now every trading day in the ticker's active
  coverage range (`_trading_calendar()`), so a trading day with no same-day article but a recent
  one within the lookback window now gets a real (non-NaN) decayed value. New regression test:
  `test_decayed_embedding_features_covers_non_publication_trading_days`.
- **[High]** `halflife_days<=0` caused silent `nan`/`inf` propagation (division by zero) into the
  weights, embedding, PCA, and downstream correlation — now raises `ValueError` up front.
- **[Medium]** `w.sum()` could theoretically be zero (all weights underflowed) — guarded.
- **[Low]** The `"source"` field in the output was always the ticker's EARLIEST article's source,
  not any article actually contributing to that day's window — a wrong, misleading value with no
  consumer depending on it. Removed rather than fixed (simplification — not needed downstream).
- **[Disclosure]** Added a prominent module-docstring + function-docstring caveat: `lookback_days`
  windows overlap across consecutive trading days, so `emb_decay_*` is autocorrelated over time —
  naive Pearson/Spearman/FDR p-values assume i.i.d. samples and are inflated by this overlap. On
  real data this feature showed 129/224 "FDR-significant" pairs vs. only 4/224 for the flat
  (non-overlapping) embedding (Story 11-3) — this dramatic gap is itself evidence supporting the
  autocorrelation-artifact explanation rather than a genuinely 30x-stronger signal. Treat these
  correlations as exploratory only; a rigorous read needs a block-bootstrap or Newey-West-style
  correction (not implemented — out of scope, matches this story's validation-only depth).

## Out of Scope
- Per-horizon λ grid search (report explicitly calls this future work) — one fixed default halflife.
- Full baseline-model (Ridge/GBM) retraining with the new feature set — correlation-level
  validation only, matching this epic's established depth (11-3, 12-1, 12-2).
- Block-bootstrap/autocorrelation-corrected significance testing (flagged as a caveat, not fixed).

## Verify
```bash
uv run pytest tests/unit/test_temporal_decay.py -q
```
Definition of Done: acceptance boxes checked, C0=100%/C1≥80% diff-coverage, `/bmad-code-review` addressed.
