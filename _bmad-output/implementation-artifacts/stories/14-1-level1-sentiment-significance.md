# Story 14.1 — Level-1 Statistical Significance (sentiment + event-type)

**Epic:** 14 (Level 1/2 Feature Evaluation per docs/gpt-guide guideline)
**Story key:** `14-1-level1-sentiment-significance`
**Status:** done

## Context
`docs/gpt-guide/news_feature_evaluation_guideline.md` Level 1 requires evaluating 8 candidate
features (Positive/Negative/Fear/Optimism/Uncertainty score, Event type/severity/confidence)
with 5 statistical methods (Pearson, Spearman, Kendall Tau, Mutual Information, Distance
Correlation). Sentiment was previously dropped from modeling entirely (Story 11-1, replaced by
PhoBERT embeddings). User explicitly requested reinstating the 5 sentiment scores for BOTH EDA
screening AND the modeling panel (reversing part of 11-1), while explicitly deferring
severity/confidence (no labeled data, needs its own story).

## Requirements (Acceptance Criteria)
- [x] `src/features/sentiment_scores.py`: 5-category Vietnamese keyword lexicon
  (positive/negative/fear/optimism/uncertainty), same scoring methodology as existing BBD
  pattern (fraction of distinct keywords matched, [0,1]) — auditable, no ML.
- [x] Event type operationalized via existing `TOPIC_CATEGORIES` (7 categories), counted
  market-wide per (ticker, date) — `event_type` candidate feature per guideline.
- [x] Vectorized scoring (`_vectorized_lexicon_scores`/`_vectorized_event_flags`) — the 1.45M-row
  post-backfill corpus made the original per-row `.apply()` prohibitively slow.
- [x] `src/eda/phase05_relationship.py`: added `kendall_tau()` + `distance_correlation()` (added
  `dcor` dependency) alongside existing Pearson/Spearman/MI.
- [x] `src/eda/phase17_level1_significance.py`: all 12 features (5 sentiment + 7 event-type) ×
  3 targets × 5 statistics, FDR-corrected, summary flags MI≈0 (useless) vs Pearson≈0-but-MI>0
  (nonlinear candidate) per guideline interpretation.
- [x] Sentiment features merged into the modeling panel (`src/modeling/dataset.py build_panel()`)
  and new ablation feature sets in `src/modeling/baseline.py` (`price+sentiment5`,
  `price+event_type`, `price+sentiment5+event_type`) isolated from the bundled `news_adv` set.
- [x] `src/modeling/significance.py`: generalized `_fit_pair`/`per_ticker_delta_r2` to accept
  an arbitrary comparison feature set; added per-family DM test + bootstrap CI section.
- [x] Dashboard: new "Level-1 Significance" page + per-family ablation table on "Significance"
  page.
- [x] Unit tests for all new pure helpers + real-data smoke tests.

## Findings
- `price+sentiment5` is DM-significant at **T+1** (p=0.033, ΔR² 95% CI [9e-06, 0.000184]) —
  distinct from the embedding-based `news_adv` finding (only T+10 significant). Sentiment and
  embedding features may capture different signal.
- `event_type` counts alone: not significant at any horizon.

## Data-quality fix (discovered mid-story, not originally scoped)
User backfilled `crawl_data/data` mid-session (~1.45M new rows across thanhnien/tuoitre/
vietnamplus). This exposed a source-name collision bug in `src/data/discover_news.py`:
`objective/news_unenriched_thanhnien_records.csv` and the new top-level `thanhnien_articles.csv`
both infer the source name `thanhnien` — the old code silently kept only the alphabetically-last
path, **losing 150/66 rows of the objective/ tier data for thanhnien/tuoitre entirely** (0% URL
overlap — verified NOT duplicates). Fixed: `discover_source_files()` now disambiguates ALL
colliding names by parent directory (`thanhnien_root`/`thanhnien_objective`) instead of silently
dropping one. `KHACH_QUAN_SOURCES` in `news_embeddings.py` updated to include the new
disambiguated names (group classification is exact-string-match based).

## Out of Scope
- Event severity / event confidence (deferred per user decision — no labeled data, needs its
  own design story).
- Rebuilding PhoBERT embeddings over the new 1.45M-row backfill (hours-scale compute; a
  separate, explicitly-approved job — see final report follow-ups).

## Verify
```bash
uv run pytest tests/unit/test_sentiment_scores.py tests/unit/test_phase17_level1_significance.py tests/unit/test_phase05_06_relationship.py -q
uv run python -m src.features.sentiment_scores
uv run python -m src.eda.phase17_level1_significance
```
Definition of Done: acceptance boxes checked, diff-coverage gate, `/bmad-code-review` addressed.
