# Summary — Epic 11: News Embedding EDA (Stories 11-1, 11-2, 11-3)

## What changed
Replaced the rule-based Vietnamese sentiment dictionary (`calculate_sentiment_vietnamese`) with
PhoBERT `[CLS]` embedding vectors as the sole news signal in the modeling pipeline, split news into
two comparable groups ("khách quan" = per-source raw articles; "tổng hợp" = consolidated
`news_articles.csv`), added an EDA phase comparing the two groups, and a correlation study testing
whether news embeddings relate to price/volatility linearly or only non-linearly. Added a Streamlit
dashboard page for each. Also converted the review-round-2 code-review fixes into an incremental,
article-level embedding cache (keyed by `url`) so daily crawls only cost O(new articles), not a full
re-encode.

## Files changed (path → purpose)
- `src/nlp/embeddings.py` — `extract_phobert_embeddings()`, frozen PhoBERT `[CLS]` pooling.
- `src/features/news_embeddings.py` — incremental article-level cache, PCA reduction (own-basis +
  shared-basis variants), replaces the old sentiment pipeline.
- `src/modeling/features.py` — `ADV_FEATURES` now `emb_0..emb_31` + topic counts (sentiment removed).
- `src/modeling/baseline.py` — `NEWS_ADVANCED` now imports the embedding feature list (was hardcoded).
- `src/eda/phase11_news_embedding_eda.py` (new) — source stats, embedding coverage, shared-PCA
  scatter, cosine similarity, `eda_output/news_embedding/`.
- `src/eda/phase12_embedding_price_correlation.py` (new) — Pearson/Spearman/MI + FDR, linear vs
  non-linear-only bucketing, reuses `phase05_relationship` statistics helpers.
- `src/dashboard/data.py` / `app.py` — two new pages: "News Embedding", "Embedding Correlation".
- `src/eda/common.py` — added `news_embedding` to `EDA_SUBDIRS`.
- `CLAUDE.md` — coverage gate raised to C0=100%/C1≥80% (explicit user request, unrelated to Epic 11
  otherwise).
- BMAD artifacts: `_bmad-output/implementation-artifacts/stories/11-{1,2,3}-*.md`,
  `sprint-status.yaml` (all three stories → `done`).

## Tests + coverage
- 173 unit tests pass (`uv run pytest tests/unit -q`).
- New/updated test files: `test_news_embeddings.py`, `test_phase11_news_embedding_eda.py`,
  `test_phase12_embedding_corr.py`, `test_modeling_features.py`, `test_modeling_baseline.py`,
  `test_dashboard.py`, `test_eda_common.py`.
- **Diff-coverage (C0): 100%** on all 8 changed source files (414 changed lines, 0 missing) —
  `diff-cover coverage.xml --fail-under=100 --compare-branch origin/main`.
- Real-data-slice smoke tests included per file (not just synthetic fixtures): `_source_stats()`,
  `build_group_embeddings()`, `run_phase()` for both new phases, all run once against the real
  crawl_data/eda_output artifacts.

## Code review (bmad-code-review, 3-layer: Blind Hunter + Edge Case Hunter + Acceptance Auditor)
Run twice (round 1 on the initial Story 11-1 diff, round 2 on Stories 11-1's incremental-cache
refactor + 11-2 + 11-3 together). All CONFIRMED/PLAUSIBLE findings addressed:

**Round 1 (5 fixed):** test run polluting the real embedding cache with mocked data (critical — cache
file deleted, tests isolated via `tmp_path`); cache had no daily-refresh staleness check; small-train
PCA fallback silently mislabeled a raw dimension as PCA; two groups compared via incomparable
independently-fit PCA bases; `TICKER_PATTERN` missing case-insensitivity; missing
`FileNotFoundError` symmetry; `run_phase()` over-reporting a skipped plot; bare `assert` for
finiteness.

**Round 2 (6 fixed):** `_reduce()`'s fallback column naming (`raw_*`) broke every downstream
`emb_*`-only consumer, silently producing all-NaN features — fixed to always emit `emb_*` +
`pca_applied` flag; corrupted cache parquet would crash all future runs — now self-heals; no
dimension-drift guard on cache append; Story 11-2's dashboard page violated its own "separate page"
scope for 11-3 — split into two pages; a test asserted only half of the linear/non-linear bucketing
logic — added a direct unit test for the missing bucket; three stories' docs/status were stale
relative to the shipped code — synced.

**Dismissed (thesis-scale, not production-multi-writer):** cache write race conditions, URL
normalization, `build_comparable_group_embeddings`'s degenerate <2-row fallback, CLAUDE.md
coverage-gate edit (explicit user instruction, not a defect).

## Real result (populated `eda_output/news_embedding/` from actual data)
- Embedding cache: 11.9MB (khách quan) + 28.8MB (tổng hợp), incremental going forward.
- Embedding × price correlation: 224 (emb_i, target) pairs tested — **4 linear-significant, 9
  non-linear-only-significant** (FDR-corrected). The news↔price relationship in this dataset, where
  present, skews toward monotonic-but-not-linear rather than pure linear.

## Commands run
```bash
uv run pytest tests/unit -q                                   # 173 passed
uv run pytest tests/unit --cov=src --cov-report=xml -q        # 173 passed
diff-cover coverage.xml --fail-under=100 --compare-branch origin/main   # 100%, 0 missing
uv run ruff check <changed files>                              # 0 issues (repo-wide pre-existing
                                                                 # issues in untouched files not fixed,
                                                                 # per Surgical Changes rule)
```
mypy: not run this session (pre-existing project note: "numpy stub env issue" per project-context.md).

## Follow-ups (not done this session, flagged for future work)
- Research report `reports/2026-07-15_news_datamining_research.md` (separate agent task) ranks 3
  next methods: embedding-based novelty/staleness feature, Vietnamese BBD-style uncertainty index,
  temporal decay/half-life weighting — none implemented yet, pending user decision.
- Dismissed edge cases (cache write races, URL normalization) — real in theory, low priority at
  current single-user batch-pipeline scale; revisit if the pipeline becomes multi-writer/scheduled.

## Definition of Done
- [x] Code satisfies the request (embedding features + 2 EDA phases + dashboard), no unrelated refactor beyond the explicitly requested CLAUDE.md coverage-gate edit.
- [x] Tests: 173 passed; C0=100% diff-coverage.
- [x] Checks run: pytest + ruff (changed files) actually executed, not claimed.
- [x] Lint scope: only changed files checked/fixed; pre-existing repo debt left alone.
- [x] Code review: `/bmad-code-review` 3-layer, twice, all confirmed findings addressed.
- [x] Summary report: this file.
- [~] Smoke: no dedicated `smoke`-marked test in this repo's config; real-data-slice integration tests serve this role per the existing test suite convention (`test_real_*_smoke` naming, no pytest marker registered).
- [x] Impact analysis: grepped `NEWS_ADVANCED`/`ADV_FEATURES`/sentiment usages across `src/`; `baseline.py`'s hardcoded feature list was the one real blast-radius hit, fixed to import dynamically.
- [x] Similar check: searched for other sentiment/PhoBERT usages in `src/` — historical phases 04-10 intentionally left untouched (Out of Scope, documented in Story 11-1).
