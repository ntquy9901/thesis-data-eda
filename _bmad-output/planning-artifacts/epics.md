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

---

## Epic 8: Advanced Modeling — Does News Help with Nonlinear Models + Richer Features + More Data?

> Epic 7 found news features add ~no value to a LINEAR HAR-Ridge baseline. This epic tests the three levers that could change that verdict: (A) nonlinear models, (B) richer news features, (C) more data (30 tickers).

### Story 8.1: Scale EDA + modeling to all 30 VN30 tickers
**Goal:** Broaden the evidence base from 5 → 30 tickers.
**Acceptance:**
- [ ] `config.EDA_TICKERS` expanded to full VN30; pipeline regenerates artifacts on 30 tickers
- [ ] Modeling panel rebuilt (panel.parquet + split_summary) covering 30 tickers
- [ ] Verify runtime acceptable; note any tickers with sparse news
**Verify:** split_summary.n_tickers == 30; pipeline end-to-end runs
**FR:** FR-017

### Story 8.2: Advanced news features
**Goal:** Richer news signal than daily counts/mean sentiment.
**Acceptance:**
- [ ] `src/modeling/features.py` producing per-(ticker,date): event-weighted news count (Σ|sentiment|), sentiment strength (|sentiment_mean|, rolling std), topic-flag counts (earnings/dividend/M&A/.../macro/sector from the Phase-4 category map)
- [ ] Features added to the modeling panel (NaN when no news)
- [ ] Unit tests: weighting, topic-flag, NaN handling
**Verify:** new feature columns present + unit-tested
**FR:** FR-010, FR-013

### Story 8.3: Nonlinear model + full comparison
**Goal:** Test whether news helps under nonlinear capacity.
**Acceptance:**
- [ ] `src/modeling/baseline.py` extended: GradientBoostingRegressor alongside Ridge; models × {price, +news-basic, +news-advanced}
- [ ] Comparison matrix over pk_t+1/5/10: Ridge vs GBM × feature sets → ΔR²/ΔRMSE
- [ ] Output: `comparison_report.md` updated with the nonlinear verdict
**Verify:** report answers "does news help under nonlinear models + advanced features + 30 tickers?"

---

## Epic 9: Statistical Significance — Is the News Contribution Real?

> Epic 8 found negligible ΔR² from news. This epic tests whether that negligible effect is **statistically distinguishable from zero** (Diebold-Mariano + bootstrap CI) and whether it holds **per-ticker** (heterogeneity) — turning "small" into "not significant", a rigorous thesis-grade null.

### Story 9.1: Forecast significance (Diebold-Mariano + bootstrap CI)
**Goal:** Formally test if news-augmented forecasts differ from price-only.
**Acceptance:**
- [ ] `src/modeling/significance.py` → Diebold-Mariano test on squared-error loss differential (price vs +news_basic vs +news_adv) per target + horizon-aware
- [ ] Bootstrap CI (1000 resamples) on ΔRMSE and ΔR²
- [ ] Output: `eda_output/modeling/significance_report.md` + `significance.json`
- [ ] Unit test: DM stat/p on synthetic (identical → p high; clearly worse → p low)
**Verify:** report states per-target whether news improvement is significant

### Story 9.2: Per-ticker heterogeneity + event abnormal-vol t-test
**Goal:** Does news help ANY ticker? Are event abnormal-vols ≠ 0?
**Acceptance:**
- [ ] Per-ticker ΔR² (price vs +news) distribution + count of tickers where news helps
- [ ] t-test on event-study abnormal vol (Phase 6) per horizon (mean ≠ 0?)
- [ ] Output appended to `significance_report.md`
**Verify:** heterogeneity + event significance reported

---

## Epic 10: Web Dashboard — Interactive Visualization of Everything

> A Streamlit dashboard reading `eda_output/` artifacts so all findings are explorable interactively (ticker/horizon/model selectors, plotly charts). No re-computation — pure visualization of existing artifacts.

### Story 10.1: Dashboard scaffold + data layer + overview
**Goal:** Runnable `streamlit run src/dashboard/app.py` with a data-loading layer + overview page.
**Acceptance:**
- [ ] `src/dashboard/data.py` — typed loaders for every artifact (panel, metrics, significance, charts findings) with caching
- [ ] `src/dashboard/app.py` — multi-page Streamlit shell + Overview page (thesis conclusion, data profile, headline metrics)
- [ ] `streamlit` added to deps; `uv sync`
- [ ] Unit tests: loaders return expected shapes (mocked artifacts)
**Verify:** `streamlit run` launches; overview renders

### Story 10.2: Interactive pages (price, news, modeling, significance)
**Goal:** Per-domain pages with selectors + plotly charts.
**Acceptance:**
- [ ] Price page: ticker selector → price/returns/vol, rolling vol, ACF/PACF
- [ ] News page: sentiment dist/time-series/by-ticker, topics, coverage
- [ ] Modeling page: metrics table, R² by horizon × feature-set, model selector
- [ ] Significance page: DM p-values, bootstrap CI, per-ticker ΔR² bar
- [ ] Smoke test: app imports + each page builder runs on sample data
**Verify:** all pages render with real artifacts

---

## Epic 15: Extended Horizon (pk_t+22)

> Current targets stop at 10 trading days. Monthly horizon (~22 trading days) is standard in volatility forecasting literature. Add pk_t+22 to all pipeline stages.

### Story 15.1: Extended horizon target pk_t+22
**Goal:** Add 22-day forward Parkinson/realized volatility target through the entire pipeline.
**Acceptance:**
- [ ] `TARGET_HORIZONS += 22` → auto-generates pk_t+22 + rv_t+22 in price metrics
- [ ] All downstream TARGETS lists updated (modeling, EDA phases)
- [ ] Dashboard shows the new horizon
- [ ] Tests pass; no regressions
**Verify:** `uv run pytest tests/ -q --tb=short && uv run streamlit run src/dashboard/app.py`

---

## Epic 16: Advanced News Signal with Embedding + Ticker-Specific Models

> Build on the finding that daily rule-based sentiment is too weak. Move to embedding-first approach: both news groups (khach_quan + tong_hop), EWMA-smoothing, ticker-specific modeling, and rigorous ablation.

### Story 16.1: Embedding features from both groups + emb_norm
**Goal:** Replace single-group (tong_hop) embedding with dual-group (khach_quan + tong_hop) features + emb_norm.
**Acceptance:**
- [ ] `features.py` processes both groups → columns `kq_emb_0..31`, `th_emb_0..31`, `kq_emb_norm`, `th_emb_norm`
- [ ] Baseline updated with `price+news_adv_dual` feature set
- [ ] Tests pass; no regressions

### Story 16.2: EWMA aggregation for embedding features
**Goal:** Add EWMA-smoothed embedding features (30d half-life) to capture slow-moving news regimes (r=+0.27 finding).
**Acceptance:**
- [ ] `features.py` adds `ewma_emb_0..31` (30d half-life per-ticker)
- [ ] Added to `price+news_adv_dual` feature set
- [ ] Tests pass

### Story 16.3: Ticker clustering + entity embeddings
**Goal:** Cluster tickers by news-sensitivity; add per-ticker embedding coefficients.
**Acceptance:**
- [ ] `ticker_clusters.py` — compute ΔR² per ticker, cluster into sensitive/insensitive groups
- [ ] Entity embeddings per ticker learned inside the Ridge model
- [ ] Clusters saved to `eda_output/modeling/ticker_clusters.json`

### Story 16.4: Mixture-of-Experts (simplified gated model)
**Goal:** Gate that only activates news branch for tickers in the sensitive cluster.
**Acceptance:**
- [ ] `src/modeling/moe.py` — gated Ridge: price-only expert + price+news expert
- [ ] Per-ticker gate weight based on cluster membership
- [ ] Comparison vs standard Ridge in comparison_report.md

### Story 16.5: Ablation + permutation importance + OOS evaluation
**Goal:** Rigorous feature selection — only keep features that improve both statistically and economically.
**Acceptance:**
- [ ] Permutation importance for each news feature in `significance.py`
- [ ] OOS evaluation on a holdout period (2026H1)
- [ ] Final recommendations: which news features to keep/drop




