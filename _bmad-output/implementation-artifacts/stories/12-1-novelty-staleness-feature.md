# Story 12.1 — Embedding-based novelty/staleness feature

**Epic:** 12 (Advanced News Signal Extensions)
**Story key:** `12-1-novelty-staleness-feature`
**Status:** done

## Context
From `reports/2026-07-15_news_datamining_research.md` §5 (top-3 recommendation #1): stale
(republished/rehashed) news gets a weaker/slower price response than fresh news (Tetlock). We
already cache raw PhoBERT embeddings per article keyed by `url`
(`src/features/news_embeddings.py`) — novelty = 1 − max cosine similarity to the same ticker's
articles in a trailing window. Zero new encoding cost.

## Requirements (Acceptance Criteria)
- [x] `src/features/news_embeddings.py`: `compute_novelty_scores(group="tong_hop", window_days=5) -> pd.DataFrame`
  — per (article, ticker) row (from the exploded raw table): cosine similarity (on `raw_*`, the
  768-dim space — NOT the lossy 32-dim PCA) to that ticker's articles published in the preceding
  `window_days` trading days (including same-day articles other than itself); `novelty = 1 - max_similarity`
  (no prior/same-day articles in window → novelty = 1.0, maximally novel by convention).
- [x] Aggregate per (ticker, date): `novelty_mean` (day's articles' mean novelty) as a new feature.
- [~] **DEVIATION (documented, not silently unmet):** `novelty_mean` is NOT wired into
  `src/modeling/features.py::ADV_FEATURES`. Reason: this story's own "Out of Scope" section
  explicitly excludes full baseline-model retraining — wiring a new column into `ADV_FEATURES`
  without re-running/re-validating the Ridge/GBM/DM-test pipeline (Epic 9) would silently change
  the modeling feature set without the validation depth that change deserves. `novelty_mean` is
  validated at the correlation level only (`phase13_novelty_correlation.py`), matching Story
  11-3's established depth for new signals. Full modeling integration is a follow-up if the
  correlation results (see `eda_output/news_embedding/novelty_price_corr.csv`) warrant it.
- [x] `src/eda/phase13_novelty_correlation.py`: correlate `novelty_mean` against price targets
  (reuse `phase05_relationship` Pearson/Spearman/MI/FDR helpers, same pattern as Story 11-3's
  `phase12_embedding_price_correlation.py`) → `eda_output/news_embedding/novelty_price_corr.csv`.
- [x] Unit tests: pure-function test for the cosine-similarity/window logic (synthetic small
  ticker history with known repeated same-day/different-day vs. novel articles), real-data-slice
  smoke test for `run_phase()`.

## Code-review fixes (bmad-code-review 3-layer)
- **[High]** Same-day duplicate articles were never compared against each other (`<` instead of
  `<=` window boundary) — a same-day rehash was scored as maximally novel. Fixed.
- **[Medium]** Cosine-similarity float overshoot outside `[-1,1]` was not clipped before computing
  novelty — fixed (`np.clip(..., -1.0, 1.0)`).
- **[Low]** `compute_novelty_correlations` had no guard for a missing `novelty_mean` column
  (KeyError risk) — fixed, mirrors the same fix in Story 12-2's `compute_uncertainty_correlations`.
- **[Disclosure]** Added the same overlapping-rolling-window autocorrelation caveat already
  documented in Story 12-3's `phase15_temporal_decay_correlation.py` — `novelty_mean`'s 5-day
  window shares the same statistical-independence caveat.

## Out of Scope
- Full baseline-model retraining/DM-test with the new feature (Epic 9-style) — this story validates
  via correlation only, consistent with Story 11-3's validation depth.
- Per-source ("khách quan") novelty — "tổng hợp" only (matches the modeling pipeline's existing
  choice of group).

## Verify
```bash
uv run pytest tests/unit/test_novelty_staleness.py tests/unit/test_phase13_novelty_correlation.py -q
uv run python -m src.eda.phase13_novelty_correlation
```
Definition of Done: acceptance boxes checked, C0=100%/C1≥80% diff-coverage, `/bmad-code-review` addressed.
