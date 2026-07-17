# Story 11.1 — PhoBERT Embedding Features (khách quan vs tổng hợp), drop sentiment

**Epic:** 11 (News Embedding EDA)
**Story key:** `11-1-news-embedding-features`
**Status:** done

## Context
Sentiment (`calculate_sentiment_vietnamese`, keyword dict) loses information. Replace it with
PhoBERT embedding vectors as the only news signal in the modeling feature pipeline. Reuse the
extraction approach already proven in `D:\bmad-projects\stock_vol_prediction01\baselines\2026-07-07_embedding_baseline\code\extract_embeddings.py`
(PhoBERT `[CLS]`, frozen, train-only-fit PCA) — do not redesign.

Two news groups (per CLAUDE.md documented sources):
- **khách quan**: per-source raw articles (`cafef`, `ssi`, `vndirect`, `vnstock`, `hsc` via `load_news_articles(source=...)`)
- **tổng hợp**: consolidated `news_articles.csv` (`load_news_articles(source=None)`)

## Requirements (Acceptance Criteria)
- [x] `src/nlp/embeddings.py`: `extract_phobert_embeddings(texts, model="vinai/phobert-base", max_len=64, batch_size=32) -> np.ndarray` (frozen PhoBERT, `[CLS]` pooling), ported from the sibling script above.
- [x] `src/features/news_embeddings.py`: PhoBERT embeddings cached INCREMENTALLY at the article level, keyed by `url` (`data/features/news_emb_articles_{group}.parquet`) — only articles not already cached get encoded, so a daily crawl adding N articles costs O(N), not a full re-encode (superseded an earlier mtime-whole-group-invalidation design during code review round 2). PCA (`dim<=32`, train-period fit) applied on demand: `build_group_embeddings(group)` (own basis, used by modeling) and `build_comparable_group_embeddings()` (ONE shared basis across both groups, used by Phase-11 EDA for a valid cross-group comparison).
- [x] `src/modeling/features.py`: `calculate_sentiment_vietnamese` usage removed; `ADV_FEATURES` = per-(ticker,date) mean-pooled embedding components (`emb_0..emb_31`) + topic counts.
- [x] `src/eda/phase11_news_embedding_eda.py`: PCA 2D scatter (shared basis) of both groups, cosine-similarity within/across groups, source stats, embedding coverage → `eda_output/news_embedding/`.
- [x] Unit tests: `tests/unit/test_news_embeddings.py`, `tests/unit/test_modeling_features.py`, `tests/unit/test_modeling_baseline.py` — mocked PhoBERT (no network/GPU), cache dir isolated via `tmp_path` (never touches real `data/features/`).

## Code-review fixes — round 1 (bmad-code-review 3-layer)
- **[Critical]** Test run was writing mocked/random embeddings into the REAL `data/features/` cache — fixed by redirecting `FEATURES_DIR` to `tmp_path` in all test fixtures; the polluted cache file was deleted.
- **[High]** Cache had no staleness check (violated CLAUDE.md daily-refresh rule) — fixed via mtime comparison against source news CSVs. **Superseded in round 2** by the incremental article-level cache (see below), which makes mtime staleness moot (new articles are detected by `url`, not file timestamps).
- **[High]** Small-train-set PCA fallback silently mislabeled a raw (non-PCA) dimension as `emb_0` — fixed: honest fallback kept `raw_*` columns. **Refined in round 2** (see below).
- **[Medium]** Two groups were compared via independently-fit PCA (invalid cross-group comparison) — fixed via `build_comparable_group_embeddings()` (one shared PCA fit on pooled train rows).
- **[Medium]** `TICKER_PATTERN` missing case-insensitivity — fixed (`re.IGNORECASE`).
- **[Medium]** Missing `try/except FileNotFoundError` symmetry for the "tổng hợp" consolidated file (crashed instead of graceful skip) — fixed in both `news_embeddings.py` and `phase11_news_embedding_eda.py`.
- **[Medium]** `run_phase()` reported `group_scatter.png` as written even when the plot was skipped — fixed (conditional append).
- **[Low]** Bare `assert` for embedding finiteness (stripped under `python -O`) — replaced with explicit `raise ValueError`.
- Dismissed as false-positive/out-of-scope: CLAUDE.md coverage-gate rewrite (explicit user instruction), `df["source"]=s` mutation (fresh DataFrame per call, no shared-state risk), removal of `VietnameseNLPProcessor` ticker path (regex word-boundary is actually more precise than the old substring match), 2-color hardcode / `batch_size=0` / docstring path (no real call site).

## Code-review fixes — round 2 (after refactoring the cache to incremental/article-level, run alongside Stories 11-2/11-3)
- **[Critical]** `_reduce()`'s small-train fallback named output columns `raw_*` while every downstream consumer (`EMB_FEATURES` in `features.py`, `startswith("emb_")` filters in `phase11`/`phase12`) only recognizes `emb_*` — silently produced all-NaN/empty results with no error. Fixed: output is ALWAYS `emb_*`; a new `pca_applied: bool` column records whether PCA actually ran, so the fallback stays honest/inspectable without breaking every consumer's naming assumption.
- **[High]** Corrupted/truncated cache parquet (e.g. from a killed process) would crash every future run — fixed: read wrapped in try/except, treated as empty (self-heals via full re-encode).
- **[Medium]** No guard against embedding-dimension drift (e.g. a different model) when appending to the cache — fixed: cached `raw_*` column count is checked against `RAW_DIM`; mismatch discards the stale cache instead of silently NaN-padding.
- Dismissed/deferred (thesis-scale single-user batch pipeline, not a concurrent production system): non-atomic cache writes / concurrent-run race, URL-collision dedup policy (`drop_duplicates` keeps first, no normalization), `build_comparable_group_embeddings`'s degenerate <2-pooled-train-rows fallback (per-group PCA, loses shared-basis guarantee) — all real in theory, none plausible at this project's actual usage pattern (single sequential runs).

## Out of Scope
- Historical EDA phases 04–10 and `report.py` (already-done epics) keep sentiment as-is — no retroactive rewrite.
- Dashboard integration (follow-up story if needed).

## Technical Notes
- `transformers<5` + `sentencepiece` required (env gotcha noted in sibling code).
- No hardcoded absolute paths; use `config/__init__.py` paths.
- `load_news_articles()` reads `crawl_data` CSVs live (no cache) — always picks up the latest daily crawl automatically; verified source files refreshed 2026-07-14 09:xx (news/cafef/ssi/vndirect same-day, vnstock/hsc slightly older per that source's own crawl schedule).

## Verify
```bash
uv run pytest tests/unit/test_news_embeddings.py tests/unit/test_modeling_features.py tests/unit/test_modeling_baseline.py tests/unit/test_eda_common.py -q
uv run pytest tests/unit --cov=src --cov-report=xml -q
diff-cover coverage.xml --fail-under=100 --compare-branch origin/main   # C0 gate (100%); C1 branch coverage >=80%
```
Definition of Done: acceptance boxes checked, C0=100%/C1≥80% diff-coverage on changed lines, code-review findings addressed, dashboard update (follow-up, requested separately) tracked as a new task.
