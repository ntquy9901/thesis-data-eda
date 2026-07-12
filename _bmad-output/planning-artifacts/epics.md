# Epics & Stories — EDA for Vietnam Stock Volatility Prediction

**Source:** `docs/PRD.md` (v1.2), `docs/Technical_Architecture.md` (v1.2), `docs/EDA_Guide_Stock_Volatility_Price_News.md`
**Scope:** Full 10 EDA phases on VN30 subset (VCB, FPT, HPG, SSI, MWG); SSI primary news source, cafef/vndirect secondary. Scales to all 30 via config.
**Convention:** Story key = `{epic}-{story}-{slug}`. Status tracked in `_bmad-output/implementation-artifacts/sprint-status.yaml`. Each story produces artifacts under `eda_output/`.
**Hard rule:** Every finding backed by statistics or visualization. Never modify raw data. Leakage must be explicit.

---

## Epic 1: EDA Foundation (Phase 1 — Profiling, Phase 2 — Quality)

### Story 1.1: EDA scaffold + config
**Goal:** Create `src/eda/` package skeleton, config for subset tickers, shared output helpers.
**Acceptance:**
- [ ] `src/eda/__init__.py` + `src/eda/common.py` (output path helpers, ticker list, plot styling)
- [ ] `config.EDA_TICKERS = ["VCB","FPT","HPG","SSI","MWG"]`, `EDA_OUTPUT_DIR = {project-root}/eda_output`
- [ ] `eda_output/` subdirectories created (profiling, quality, price, news, relationship, feature_engineering, leakage, report)
- [ ] Smoke test: `import src.eda.common` succeeds; output dirs exist
**Verify:** `.venv/Scripts/python.exe -m pytest -m smoke`
**FR:** FR-017

### Story 1.2: Phase 1 — Dataset Profiling
**Goal:** Profiling table for every input table (news sources, price files, macro).
**Acceptance:**
- [ ] `src/eda/phase01_profiling.py` → `profile_table(tickers) -> DataFrame`
- [ ] Columns: table, row_count, col_count, dtypes, primary_key, candidate_key, memory_mb, date_min, date_max
- [ ] Output: `eda_output/profiling/profiling_table.csv`
- [ ] Covers: ssi, cafef, vndirect news + 5 tickers + DXY/SBV macro
**Verify:** Run module → CSV exists with ≥9 rows; date ranges sane
**FR:** FR-008 | **Phase:** 1

### Story 1.3: Phase 2 — Data Quality
**Goal:** Missingness, duplicates, invalid values across news + price.
**Acceptance:**
- [ ] `src/eda/phase02_quality.py`
- [ ] Missingness: % by column, by stock, by date; pattern → `missingness_report.csv`
- [ ] Duplicates: duplicated news (by url/title hash), duplicated price rows → `duplicate_report.json`
- [ ] Invalid values: negative volume, impossible prices (high<low), future timestamps, invalid tickers → `invalid_values.json`
- [ ] All outputs in `eda_output/quality/`
**Verify:** Reports generated; no exceptions on real data
**FR:** FR-002, FR-008 | **Phase:** 2

---

## Epic 2: Price EDA (Phase 3)

### Story 2.1: Returns, realized volatility, ATR (leakage-safe targets)
**Goal:** Compute core price metrics + T+1/T+5/T+10 realized vol targets.
**Acceptance:**
- [ ] `src/eda/phase03_price_eda.py` → per ticker: returns, log_returns, atr_14, realized_vol_5d/20d
- [ ] **Targets:** rv_t+1, rv_t+5, rv_t+10 computed with strict future-only returns (ADR-006)
- [ ] Output: `eda_output/price/price_metrics_<ticker>.parquet`
- [ ] Unit test: rv_t+h uses only returns [t+1, t+h] (no look-ahead)
**Verify:** Unit test + parquet exists for 5 tickers
**FR:** FR-003, FR-009 | **Phase:** 3

### Story 2.2: Rolling stats, ACF/PACF, correlation heatmap
**Goal:** Time-series diagnostics + visualizations.
**Acceptance:**
- [ ] Rolling statistics (20/60-day windows) for returns and vol
- [ ] ACF/PACF plots per ticker → `acf_pacf_<ticker>.png`
- [ ] Cross-ticker correlation heatmap → `corr_heatmap.png`
- [ ] Rolling volatility plot → `rolling_vol.png`
- [ ] All in `eda_output/price/`
**Verify:** PNG files exist and are non-empty
**FR:** FR-003, FR-015 | **Phase:** 3

### Story 2.3: Outliers, volatility clustering, regime report
**Goal:** Narrative findings on price behavior.
**Acceptance:**
- [ ] Outlier detection (>3σ on returns) → `outliers_<ticker>.csv`
- [ ] Volatility clustering check (ARCH-LM or Ljung-Box on squared returns)
- [ ] Regime change detection (rolling vol quantile breaks)
- [ ] Markdown summary → `eda_output/price/findings.md`
**Verify:** findings.md references statistics/evidence
**FR:** FR-003 | **Phase:** 3

---

## Epic 3: News EDA (Phase 4 + Phase 7 Sparse News)

### Story 3.1: Coverage, publish-time, effective_trading_date
**Goal:** Temporal structure of news aligned to trading days.
**Acceptance:**
- [ ] `src/eda/phase04_news_eda.py`
- [ ] Coverage: news/day, news/stock, days without news → `coverage_report.csv`
- [ ] Publish-time: before/during/after market, weekend → `publish_time.png` + counts
- [ ] `effective_trading_date` mapping (news before close → same day; after close/weekend → next trading day)
- [ ] Outputs in `eda_output/news/`
**Verify:** effective_trading_date aligns ≥95% of news to a trading day
**FR:** FR-002, FR-010 | **Phase:** 4

### Story 3.2: Sentiment summaries + topics
**Goal:** Sentiment distribution + topic extraction.
**Acceptance:**
- [ ] Reuse rule-based sentiment from `task1_3_vietnamese_nlp.py`; add summaries (mean/std/min/max, pos/neg ratio)
- [ ] Topic modeling (NMF or LDA on title+lead) → top 7 topics; map to earnings/dividend/M&A/management/regulation/macro/sector where possible
- [ ] Source distribution + repost/duplicate rate → `source_report.json`
- [ ] Outputs in `eda_output/news/`
**Verify:** sentiment_summary.json + topics present
**FR:** FR-003 | **Phase:** 4

### Story 3.3: Sparse news features (Phase 7)
**Goal:** News-availability features that respect "no news ≠ neutral news".
**Acceptance:**
- [ ] `src/eda/phase07_sparse_news.py`
- [ ] Features per (ticker, trading_date): coverage_ratio, days_since_last_news, news_count_1d/3d/5d, `news_available` flag (1/0)
- [ ] Sentiment ONLY filled when news_available=1; NaN otherwise (never 0 for "no news")
- [ ] Output: `eda_output/news/sparse_news_features.parquet`
- [ ] Unit test: no-news rows have news_available=0 and NaN sentiment
**Verify:** parquet + unit test pass
**FR:** FR-010 | **Phase:** 7

---

## Epic 4: Relationship & Event Study (Phase 5 + Phase 6)

### Story 4.1: Relationship analysis methods
**Goal:** Quantify news ↔ future volatility/return.
**Acceptance:**
- [ ] `src/eda/phase05_relationship.py`
- [ ] Pairs analyzed: news_count vs future vol (rv_t+1/5/10); sentiment vs future vol; topic vs vol; negative news vs return
- [ ] Methods: Pearson, Spearman, Mutual Information, Granger causality, cross-correlation (lags 0–5)
- [ ] Multiple-testing correction (FDR) on p-values
- [ ] Output: `eda_output/relationship/{corr_matrix.csv, granger_results.json, cross_corr.png}`
**Verify:** All 5 methods produce results; FDR applied
**FR:** FR-004, FR-014 | **Phase:** 5

### Story 4.2: Event study (T-10/T-5/T-1 → T+1/T+5/T+10)
**Goal:** Market reaction around important news events.
**Acceptance:**
- [ ] `src/eda/phase06_event_study.py`
- [ ] Select important events (top sentiment magnitude / topic-tagged)
- [ ] Windows: pre T-10/T-5/T-1, post T+1/T+5/T+10
- [ ] Metrics: realized vol, return, abnormal volatility (vs baseline)
- [ ] Output: `eda_output/relationship/event_study.csv` + `event_study_plot.png`
**Verify:** events table non-empty; abnormal vol computed
**FR:** FR-004.5, FR-011 | **Phase:** 6

---

## Epic 5: Feature Validation & Leakage (Phase 8 + Phase 9)

### Story 5.1: Feature validation (Phase 8)
**Goal:** Audit engineered feature set before modeling.
**Acceptance:**
- [ ] `src/eda/phase08_feature_validation.py`
- [ ] Checks: missingness, variance (near-zero), redundancy (duplicate), correlation (|r|>0.9 collinear), drift (train/test distribution shift)
- [ ] Recommendations: drop constant/duplicate/highly-collinear features
- [ ] Output: `eda_output/feature_engineering/{feature_report.csv, collinearity.json, drop_recommendations.json}`
**Verify:** drop_recommendations non-empty and justified
**FR:** FR-013 | **Phase:** 8

### Story 5.2: Leakage detection + explicit list (Phase 9)
**Goal:** Enumerate every potential leakage; verify none active.
**Acceptance:**
- [ ] `src/eda/phase09_leakage.py`
- [ ] Checks: publish vs trading timestamp; future information in features; rolling-window look-ahead; target leakage (feature correlates with target via construction); normalization leakage (fit on full data)
- [ ] **Explicit leakage list** → `eda_output/leakage/leakage_list.md` (each item: description, status=fixed/accepted, mitigation)
- [ ] Machine-readable → `leakage_checks.json`
- [ ] Time-based split enforced (train ≤ 2024, test ≥ 2025)
**Verify:** leakage_list.md exists; every item has a status
**FR:** FR-012 | **Phase:** 9

---

## Epic 6: Visualization & Final Report (Phase 10 + Final Report)

### Story 6.1: EDA visualization pack (11 charts)
**Goal:** All required charts from Phase 10.
**Acceptance:**
- [ ] `src/eda/phase10_visualizations.py`
- [ ] Charts: (1) missing value heatmap, (2) news coverage by stock, (3) news count by day, (4) sentiment distribution, (5) return distribution, (6) volatility distribution, (7) rolling volatility, (8) correlation heatmap, (9) event study plots, (10) news count vs future volatility, (11) feature importance placeholder (skip if no model)
- [ ] All PNGs in respective `eda_output/<phase>/` dirs; index in `eda_output/report/charts_index.md`
**Verify:** ≥10 charts exist (11th conditional)
**FR:** FR-015 | **Phase:** 10

### Story 6.2: Final EDA report + candidate features
**Goal:** Assemble the deliverable report per EDA Guide "Final Report".
**Acceptance:**
- [ ] `src/eda/report.py` aggregates findings from all phases
- [ ] Sections: Executive Summary (data quality, major risks, key observations), Top Findings (with evidence), Recommended Candidate Features, Risks (leakage/sparse/imbalance/outliers), Recommended Next Steps
- [ ] Output: `eda_output/report/eda_final_report.md` + `candidate_features.csv`
- [ ] Candidate features cross-reference leakage status from Story 5.2
**Verify:** report.md references evidence (statistics/charts); candidate_features.csv non-empty
**FR:** FR-016 | **Phase:** Final Report

---

## Sprint Sizing (guidance)
- **Sprint 1:** Epic 1 + Epic 2 (foundation + price EDA) — unblocks all downstream phases.
- **Sprint 2:** Epic 3 + Epic 4 (news EDA + relationship/event study).
- **Sprint 3:** Epic 5 + Epic 6 (feature/leakage + viz/report) — produces final deliverable.
- **Sprint 4:** Epic 7 (modeling) — quantifies the news↔volatility relationship.

---

## Epic 7: Modeling — News Contribution to Parkinson Volatility

> Consumes EDA outputs. Aligns with the sibling `stock_vol_prediction01` baselines, which predict **Parkinson** vol using HAR features (daily/weekly/monthly rolling means). Goal: quantify how much news features improve vol prediction.

### Story 7.1: Modeling dataset (HAR + news + targets, leakage-safe split)
**Goal:** Build a train-ready feature matrix with a strict time-based split.
**Acceptance:**
- [ ] `src/modeling/dataset.py` → HAR features on parkinson_vol (1d/5d/22d rolling means) joined with news features (news_count_1d/3d/5d, days_since_last_news, sentiment_mean) + targets pk_t+1/+5/+10
- [ ] Time-based split: train ≤ 2024-12-31, test ≥ 2025-01-01 (NO random split, no shuffling)
- [ ] NaN handling: rows missing target dropped; news NaN preserved (model handles or impute-fit-on-train-only)
- [ ] Output: `eda_output/modeling/dataset_<ticker>.parquet` (or panel) + split summary
- [ ] Unit test: train dates all < test dates; HAR features use trailing windows only (no look-ahead)
**Verify:** split monotonic; HAR unit test passes
**FR:** FR-017, FR-013

### Story 7.2: Baseline models + news-contribution comparison
**Goal:** Train HAR (price-only) vs HAR+news; compare.
**Acceptance:**
- [ ] `src/modeling/baseline.py` → train both models per horizon (pk_t+1/+5/+10) on the 5-ticker panel
- [ ] Metrics: RMSE, MAE, R², QLIKE, directional accuracy (align with sibling baselines)
- [ ] Comparison table: per-horizon ΔRMSE/ΔR² (news vs no-news) → news contribution
- [ ] Output: `eda_output/modeling/metrics.csv` + `comparison_report.md`
- [ ] Unit test: metrics computed on synthetic data are correct; no data leakage (fit on train only)
**Verify:** comparison_report quantifies news contribution (positive/negative/neutral)
**FR:** FR-016 (candidate-feature validation), modeling extension

