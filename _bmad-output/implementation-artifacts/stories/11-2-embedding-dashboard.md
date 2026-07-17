# Story 11.2 — Dashboard page for News Embedding EDA

**Epic:** 11 (News Embedding EDA)
**Story key:** `11-2-embedding-dashboard`
**Status:** done

## Context
Story 11-1 produces `eda_output/news_embedding/` artifacts (source stats, embedding coverage,
group scatter PNG, group similarity JSON). The Streamlit dashboard (`src/dashboard/app.py`,
`src/dashboard/data.py`) has a page per phase (Overview/Price/News/Modeling/Significance) but
none for Phase 11 yet.

## Requirements (Acceptance Criteria)
- [x] `src/dashboard/data.py`: added `load_news_embedding_source_stats()`, `load_news_embedding_coverage()`,
  `load_embedding_price_corr()` (thin wrappers over existing generic csv-loader pattern).
- [x] `src/dashboard/app.py`: new `page_news_embedding()` — shows source stats table, embedding
  coverage table, the `group_scatter.png` image (khách quan vs tổng hợp), and the
  within/across-group cosine similarity numbers from `group_similarity.json`.
- [x] Registered as `"News Embedding"` in `PAGES`.
- [x] Unit tests in `tests/unit/test_dashboard.py`: new loaders return empty/default on missing
  artifacts, no exception.

## Out of Scope
- Story 11-3 (embedding × price correlation) — separate page/story: `page_embedding_correlation()`,
  registered as `"Embedding Correlation"`. (Code-review note: an earlier draft merged the
  correlation UI into this page, violating this exact boundary — split into its own page/function
  during round-2 review.)

## Verify
```bash
uv run pytest tests/unit/test_dashboard.py -q
uv run streamlit run src/dashboard/app.py   # manual: "News Embedding" page loads without error
```
Definition of Done: acceptance boxes checked, C0=100%/C1≥80% diff-coverage, `/bmad-code-review` addressed.
