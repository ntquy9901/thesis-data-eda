# Story 12.2 — Vietnamese Baker-Bloom-Davis (BBD) style uncertainty index

**Epic:** 12 (Advanced News Signal Extensions)
**Story key:** `12-2-uncertainty-index`
**Status:** done

## Context
From `reports/2026-07-15_news_datamining_research.md` §4 (top-3 recommendation #2): BBD's EPU
index is a simple, market-wide daily frequency count of articles matching an
(Economy ∧ Policy ∧ Uncertainty) keyword triple, historically correlated with realized volatility
(BBD's own r=0.73 with VIX). Cheapest item on the list — keyword counting only, no ML, reuses the
`TOPIC_CATEGORIES` keyword-matching pattern already in `phase04_news_eda.py`.

## Requirements (Acceptance Criteria)
- [x] `src/eda/phase14_uncertainty_index.py` (phase13 was claimed by Story 12-1's novelty
  correlation): Vietnamese keyword triple — `ECON_KW`, `POLICY_KW`, `UNCERTAINTY_KW`
  (module-level, each with an inline rationale comment). An article is "uncertain" iff it
  contains ≥1 term from ALL THREE categories.
- [x] `build_uncertainty_index() -> pd.DataFrame`: daily (market-wide, using the "tổng hợp"
  consolidated corpus) aggregate — `date, n_articles, n_uncertain, uncertainty_ratio`.
- [x] Correlate `uncertainty_ratio` against a MARKET-WIDE AVERAGE of forward Parkinson & realized
  vol (mean across all tickers per date — NOT per-ticker, see code-review fix below) → CSV +
  summary JSON in `eda_output/uncertainty/` (registered in `EDA_SUBDIRS`).
- [x] Unit tests: keyword-triple classification (positive/negative cases per category, plus
  NFC/NFD Unicode-normalization robustness), daily aggregation logic, real-data-slice smoke test
  for the correlation run.

## Code-review fixes (bmad-code-review 3-layer)
- **[High]** Original `_load_joined_panel` joined the single daily `uncertainty_ratio` value
  against EVERY ticker's target separately (pseudo-replication: ~30x duplication of the same X
  value per date), inflating the effective sample size and biasing p-values downward. Fixed:
  price targets are now averaged ACROSS tickers into one market-wide value per date before the
  join — one row per date, matching the index's own "market-wide daily" semantics.
- **[Medium]** Missing JSON summary artifact (AC required "CSV + summary JSON") — added
  `summarize()` + `uncertainty_price_corr_summary.json` (same linear/non-linear-only shape as
  Story 11-3/12-1).
- **[Medium]** No Unicode NFC/NFD normalization before keyword matching — a keyword literal
  (NFC) could silently fail to match text stored in NFD form. Fixed via `unicodedata.normalize`.
- **[Low]** `compute_uncertainty_correlations` had no guard for a missing `uncertainty_ratio`
  column (KeyError risk) — fixed.
- **[Low]** Keyword lists lacked inline rationale — added; also removed "thị trường"/"biến động"
  (too generic/ubiquitous in routine market reporting, would make Economy/Uncertainty almost
  always true and defeat BBD's independent-discrimination requirement).
- **Dismissed (inherent to the BBD method, not a bug):** no negation handling (e.g. "không có rủi
  ro" still matches "rủi ro") and no cross-category proximity requirement — this is BBD's own
  accepted simplicity/cost tradeoff, documented in the module docstring rather than "fixed."

## Out of Scope
- Monthly BBD-style index (original BBD is monthly; this project's price/vol series is daily —
  daily index is the appropriate granularity here, not a deviation).
- Full baseline-model integration — correlation-level validation only (matches 11-3/12-1 depth).

## Verify
```bash
uv run pytest tests/unit/test_uncertainty_index.py -q
uv run python -m src.eda.phase14_uncertainty_index
```
Definition of Done: acceptance boxes checked, C0=100%/C1≥80% diff-coverage, `/bmad-code-review` addressed.
