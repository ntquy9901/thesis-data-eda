# Story 11.3 — News embedding × price correlation EDA (linear vs non-linear)

**Epic:** 11 (News Embedding EDA)
**Story key:** `11-3-embedding-price-correlation`
**Status:** done

## Context
User question: "tin tức tích cực thì giá có lên không, có mối quan hệ thế nào qua các vector" —
does the news embedding (now the only news signal, Story 11-1) relate to price/volatility, and
is that relationship linear or not? Reuse the existing Pearson/Spearman/MI/FDR machinery in
`src/eda/phase05_relationship.py` (`pearson_spearman`, `mutual_information`, `fdr_correct`) —
do not reimplement statistics.

## Requirements (Acceptance Criteria)
- [x] `src/eda/phase12_embedding_price_correlation.py`:
  - Join `eda_output/modeling/advanced_news_features.parquet` (ticker, date, `emb_0..emb_31`)
    with each ticker's `price_metrics_<ticker>.parquet`. Targets: reuses `phase05_relationship.TARGETS`
    as-is (`log_returns`, `pk_t+1/5/10`, `rv_t+1/5/10`) rather than a separate hardcoded subset —
    simplicity-first (no duplicate constant); AC originally only named the `pk_*`/`log_returns`
    subset, `rv_*` are an accepted incidental inclusion from reusing the shared constant.
  - For each `emb_i` × target: Pearson r/p (linear) + Spearman r/p (monotonic/non-linear-tolerant)
    + mutual information (general non-linear dependence), via `phase05_relationship` helpers.
  - FDR-correct (BH, alpha=0.05) across all (emb_i, target) p-values, separately for Pearson and Spearman.
  - Also compute `emb_norm` (L2 norm of the mean-pooled embedding — "news intensity" proxy) vs
    each target, same 3 statistics.
  - Summarize: count of dims where Pearson is FDR-significant ("linear signal") vs dims where
    only Spearman/MI is significant but not Pearson ("non-linear-only signal").
  - Outputs → `eda_output/news_embedding/`: `embedding_price_corr.csv` (long format, all
    emb_i x target rows) + `embedding_price_corr_summary.json` (linear vs non-linear-only counts,
    top-5 |Pearson r| dims per target).
- [x] Unit tests `tests/unit/test_phase12_embedding_corr.py`: synthetic panel for the linear-signal
  path; `summarize()`'s linear-vs-nonlinear-only bucketing is unit-tested directly against a
  hand-built `corr` table (a data-generating process that's FDR-significant on Spearman but NOT
  on Pearson at n=200 turned out to be numerically hard to construct reliably — direct logic
  testing is the more robust choice here); real-data-slice smoke test for `run_phase()`.

## Code-review notes (round 2)
- Original test only asserted `linear_significant_count`, never `nonlinear_only_significant_count`
  despite the synthetic fixture being built to exercise it — added `test_summarize_nonlinear_only_bucket`
  testing the bucketing logic directly (see AC note above for why it's a hand-built table, not a
  synthetic statistical panel).

## Out of Scope
- Causal claims — correlation/MI only, consistent with the rest of the EDA (Phase 5 disclaimer).
- Per-ticker breakdown (aggregate across all tickers, like Phase 5's `corr_matrix.csv`).

## Verify
```bash
uv run pytest tests/unit/test_phase12_embedding_corr.py -q
uv run python -m src.eda.phase12_embedding_price_correlation
```
Definition of Done: acceptance boxes checked, C0=100%/C1≥80% diff-coverage, `/bmad-code-review` addressed.
