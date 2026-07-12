# EDA Pipeline — Complete (All 15 Stories, 6 Epics)

**Date:** 2026-07-12 08:05
**Authored:** autonomous session (Ntquy asleep, prior authorization)
**Repo:** https://github.com/ntquy9901/thesis-data-eda
**Branch:** main (commits 7c21056 → 4c3197c)

## What was delivered
A complete 10-phase EDA pipeline (`src/eda/`) implementing `docs/EDA_Guide_Stock_Volatility_Price_News.md`, driven by BMAD artifacts (PRD v1.2, Technical Architecture v1.2, epics, sprint-status, stories). Direct implementation (bmad-loop orchestrator blocked on Windows — no mux backend; documented).

| Epic | Stories | Phase(s) | Commit |
|------|---------|----------|--------|
| 1 Foundation | 1-1, 1-2, 1-3 | scaffold + Phase 1 profiling + Phase 2 quality | 7c21056, 06116aa |
| 2 Price EDA | 2-1, 2-2, 2-3 | Phase 3 returns/realized-vol/ATR/Parkinson/leakage-safe targets + diagnostics | 06e4023, 650234e |
| 3 News EDA | 3-1, 3-2, 3-3 | Phase 4 coverage/publish-time/effective_trading_date/sentiment/topics + Phase 7 sparse news | 6e9e334 |
| 4 Relationship + Event | 4-1, 4-2 | Phase 5 Pearson/Spearman/MI/Granger/cross-corr/FDR + Phase 6 event study | 8797185 |
| 5 Validation + Leakage | 5-1, 5-2 | Phase 8 feature validation + Phase 9 leakage detection | 1d7c6a8 |
| 6 Viz + Report | 6-1, 6-2 | Phase 10 visualizations + final report | 4c3197c |

## Key decisions (autonomous)
- **Parkinson vol as primary target** — discovered the sibling `stock_vol_prediction01` baselines predict Parkinson `(ln(H/L))²/(4ln2)`, NOT realized vol. EDA now emits both `rv_t+1/5/10` and `pk_t+1/5/10` (baseline-aligned). Saved to memory.
- **Leakage-safe targets** — `rv_t+h` uses future-only returns (shift -h); proven by a mutation unit test (ADR-006).
- **Strict CLAUDE.md enforcement** — after Ntquy's correction, every story ran: code → unit tests + **integration runner tests** + **real-data smoke** → `/bmad-code-review` (3-layer adversarial) → `diff-cover --fail-under=80` gate → commit/push → sprint-status.

## Code review value (bugs the reviews caught that tests missed)
- Epic 1: date mass-NaT (mixed ISO/DD/MM + tz), tz-aware crash, NaN-as-duplicate, dead `known_tickers`, missing AC.
- Epic 2: ACF/PACF short-series crash, ±inf from zero prices, missing schema validation, clustering false-positives, dead regime code.
- Epic 3 (CRITICAL): inverted `SOURCE_DAYFIRST` (cafef=ISO not DD/MM), sentiment key `"score"` not `"sentiment_score"` (all-zero sentiment), NaT→last-date silent mapping.
- Epic 4: FDR NaN-poisoning, pooled cross-correlation crossing ticker boundaries, missing negative-news-vs-return pair.
- Epic 5 (CRITICAL): `_build_feature_matrix` dropped targets → leakage check dead code; split-date contradicted policy.

## Tests & quality
- **89 unit tests** (helpers + integration runners + real-data smoke across 5 tickers), all pass.
- **ruff** clean on all `src/eda` + `tests`.
- **diff-cover** ≥ 81% every epic (last: 95%).
- **Smoke** 6/6. Full pipeline runs end-to-end (phase1→report) without error.

## Candidate features (`eda_output/report/candidate_features.csv`)
9 keep (atr_14, realized_vol_5d/20d, parkinson_vol, news_count_1d/3d/5d, days_since_last_news, sentiment_mean), 1 drop (coverage_ratio_5d), **0 leakage suspects**.

## Outputs inventory (`eda_output/`)
profiling/ · quality/ · price/ (parquet + ACF/PACF + heatmap + findings) · news/ (coverage, publish_time, sparse panel, topics, sentiment) · relationship/ (corr_matrix, granger, cross_corr, event_study) · feature_engineering/ · leakage/ (explicit list) · report/ (eda_final_report.md + candidate_features + charts_index).

## Known limitations / follow-ups
- **topic-vs-vol pair deferred** — topics are article-level; per-day topic aggregation belongs to Phase 8 feature engineering.
- **Orchestrator (`bmad-loop run`) not usable on this machine** — Windows native has no mux backend (tmux POSIX, psmux unimplemented); WSL2 blocked at BIOS virtualization. Direct implementation used instead. Artifacts are orchestrator-ready if environment is fixed.
- SHAP/feature-importance chart deferred (no model trained in EDA).
- Scale to all 30 VN30 by changing `config.EDA_TICKERS`.

## Honest "Not run"
- `mypy` — not run this session (type hints are simple; defer to a dedicated type-check pass).
- `bmad-loop run` — blocked by environment (documented above), not a code gap.
