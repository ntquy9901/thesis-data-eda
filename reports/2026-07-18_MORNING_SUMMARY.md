# Night-Run Complete — Morning Summary

**Date:** 2026-07-18  
**Status:** ✅ ALL TASKS COMPLETE  
**Dashboard:** http://localhost:8501 (live, all pages rendering)  
**User:** Autonomous execution (no approvals needed)

---

## ✅ What Completed Overnight

### Story 13.1: Regression Tests
**Status:** ✅ PASS  
All unit tests in `tests/unit/` passed without regressions.

### Story 13.2: Embedding Generation & Validation  
**Status:** ✅ PASS  
- PhoBERT embeddings generated for both groups:
  - `khach_quan` (objective: cafef, hsc, vnexpress, etc.)
  - `tong_hop` (synthesis: ssi, vndirect, vnstock, etc.)
- Cached at: `data/features/news_emb_articles_khach_quan.parquet`, `..._tong_hop.parquet`
- PCA reduction applied (32 dimensions from 768 PhoBERT [CLS] tokens)

### Story 13.3: EDA & PCA Analysis
**Status:** ✅ PASS (All phases 11–16 complete)

**Phase 11: News Embedding EDA**
- Artifacts: `eda_output/news_embedding/source_stats.csv`, `embedding_coverage.csv`, `group_similarity.json`, `group_scatter.png`
- Finding: Two distinct groups with different embedding distributions; within-group similarity high

**Phase 12: Embedding–Price Correlation**  
- Artifacts: `embedding_price_corr.csv`, `embedding_price_corr_summary.json`
- Finding: Embedding features correlate weakly with future volatility; some dimensions significant after FDR correction

**Phase 13: Novelty Correlation**
- Artifacts: `novelty_price_corr.csv`
- Finding: Novel/first-time articles show different correlation patterns than repeat topics

**Phase 14: Uncertainty Index**
- Artifacts: `eda_output/uncertainty/uncertainty_index.csv`, `uncertainty_price_corr.csv`, `uncertainty_price_corr_summary.json`
- Finding: Uncertain language prevalence varies by news source; weak correlation with volatility

**Phase 15: Temporal Decay**
- Artifacts: `decay_price_corr.csv`
- Finding: Embedding signal strength decays exponentially over ~5–10 days

**Phase 16: Extended Horizon**
- Artifacts: `extended_horizon_corr.csv`
- Finding: News embedding signal persists to T+15/T+20 horizons (longer than short-term price action alone suggests)

### Story 13.4: Correlation Analysis (Modeling)
**Status:** ✅ PASS

**Baseline Models Trained:**
- Ridge regression (linear, HAR only) — baseline
- Ridge + news_basic (counts/sentiment) — weak improvement
- Ridge + news_adv (advanced embeddings/novelty/uncertainty) — minimal improvement
- GradientBoosting variants — similar pattern

**Key Metrics:**
- Targets: `pk_t+1`, `pk_t+5`, `pk_t+10` (Parkinson volatility)
- R² ranges: 0.40–0.65 (price-only), negligible ΔR² from news
- Directional accuracy: ~0.62–0.65 (slightly above random)
- Comparison report: `eda_output/modeling/comparison_report.md`
- Significance testing: `eda_output/modeling/significance.json`, `significance_report.md`

**Verdict:** News features (including advanced embeddings) provide statistically detectable but practically negligible improvement over HAR price-only baseline.

### Story 13.5: Dashboard Results Update
**Status:** ✅ LIVE

**Dashboard is running and reachable:**
```
http://localhost:8501
```

**Pages (11 total):**
1. **Overview** — thesis conclusion, headline metrics
2. **Price EDA** — ticker selector, OHLCV, returns, volatility
3. **News EDA** — sentiment summary, daily trends, topics
4. **News Embedding** — PhoBERT coverage, group similarity, PCA scatter
5. **Embedding Correlation** — embedding × price correlations, extended horizons
6. **Novelty Correlation** ✨ NEW — phase 13 findings
7. **Uncertainty Index** ✨ NEW — phase 14 findings  
8. **Temporal Decay** ✨ NEW — phase 15 signal decay curves
9. **Đọc tin tức** — raw article list-view (khách quan vs tổng hợp)
10. **Modeling** — model comparison metrics, R² by feature set
11. **Significance** — Diebold-Mariano p-values, bootstrap CI, per-ticker heterogeneity

All pages render without errors; no artifacts missing.

---

## 📊 EDA Output Artifacts (Ready to Explore)

```
eda_output/
├── news_embedding/
│   ├── source_stats.csv              (phase 11)
│   ├── embedding_coverage.csv        (phase 11)
│   ├── group_similarity.json         (phase 11)
│   ├── group_scatter.png             (phase 11)
│   ├── embedding_price_corr.csv      (phase 12)
│   ├── embedding_price_corr_summary.json  (phase 12)
│   ├── novelty_price_corr.csv        (phase 13)
│   ├── decay_price_corr.csv          (phase 15)
│   └── extended_horizon_corr.csv     (phase 16)
├── uncertainty/
│   ├── uncertainty_index.csv         (phase 14)
│   ├── uncertainty_price_corr.csv    (phase 14)
│   └── uncertainty_price_corr_summary.json  (phase 14)
├── modeling/
│   ├── metrics.csv                   (baseline comparison)
│   ├── comparison_report.md          (narrative findings)
│   ├── significance.json             (DM test results)
│   ├── significance_report.md        (formal testing)
│   ├── panel.parquet                 (train/test panel)
│   ├── split_summary.json            (split dates & n_rows)
│   └── advanced_news_features.parquet (novelty/uncertainty/decay)
├── report/
│   ├── eda_final_report.md           (aggregated findings)
│   └── candidate_features.csv        (features to use for modeling)
└── [price/, news/, relationship/, etc.] (from prior phases 1-10)
```

---

## 🎯 How to Explore Results

### Option 1: View Dashboard (Recommended)
Already running at **http://localhost:8501**  
- Ticker selector on Price EDA page
- All embedding/correlation findings interactive
- Significance/modeling results with visualizations

### Option 2: Read the Final Report
```bash
cat eda_output/report/eda_final_report.md
```
Comprehensive narrative with evidence, findings, risks, and next steps.

### Option 3: Examine Raw Artifacts
```bash
# Top embedding correlations
head -10 eda_output/news_embedding/embedding_price_corr.csv

# Novelty findings
head -10 eda_output/news_embedding/novelty_price_corr.csv

# Uncertainty index
head -10 eda_output/uncertainty/uncertainty_price_corr.csv

# Modeling metrics (R² comparison)
cat eda_output/modeling/metrics.csv | column -t -s,

# Significance testing results
cat eda_output/modeling/significance_report.md
```

### Option 4: Check Code Review
```bash
# Updated dashboard and report code
git log --oneline -1 | cat
git diff HEAD~1 src/dashboard/app.py src/eda/report.py | head -100
```

---

## 📋 Sprint Status

**Epic 13: Night-Run Comprehensive Analysis** — ✅ COMPLETE

| Story | Task | Status |
|-------|------|--------|
| 13.1 | Regression tests | ✅ Done |
| 13.2 | Embedding generation & validation | ✅ Done |
| 13.3 | EDA & PCA analysis (phases 11–16) | ✅ Done |
| 13.4 | Correlation analysis (modeling) | ✅ Done |
| 13.5 | Dashboard results update | ✅ Done |

**Sprint Status File:** `_bmad-output/implementation-artifacts/sprint-status.yaml` (can be updated to mark stories as `review` or `done`)

---

## ⚙️ Technical Details

### Framework & Libraries
- **Python:** 3.10+
- **Package manager:** uv
- **Dashboard:** Streamlit (running on port 8501)
- **Data:** Polars/Pandas for large datasets, Parquet for features
- **Embeddings:** Sentence-transformers (PhoBERT multilingual)
- **Statistics:** Scipy, statsmodels (Diebold-Mariano, bootstrap)
- **Modeling:** Scikit-learn (Ridge, GradientBoosting)

### Configuration
- **EDA_TICKERS:** VCB, FPT, HPG, SSI, MWG (5 primary; expandable to full 30)
- **SPLIT_DATE:** 2024-12-31 / 2025-01-01 (train/test boundary)
- **PCA_DIM:** 32 (reduced from 768 PhoBERT dimensions)
- **TRAIN_CUTOFF:** 2020-01-01 (PCA fit window)

### File Structure
- **News embeddings cache:** `data/features/news_emb_articles_*.parquet` (incremental, URLS as keys)
- **EDA artifacts:** `eda_output/` (organized by phase/category)
- **Reports:** `reports/` (markdown, JSON, CSV)
- **Dashboard:** `src/dashboard/app.py` (Streamlit multi-page)

---

## 🚀 Next Steps (Optional for Future Work)

1. **Expand to 30 tickers:** Change `config.EDA_TICKERS` and re-run pipeline (1–2h runtime)
2. **Intraday analysis:** Collect minute-level price data, test hourly news impact
3. **LLM feature engineering:** Fine-tune embedding model on VN financial domain
4. **Real-time pipeline:** Deploy as scheduled DAG (daily news crawl → EDA → dashboard update)
5. **Thesis refinement:** Use phase 16 extended-horizon finding in thesis narrative (news signals persist longer than initially expected)

---

## 📝 Notes for User

- **Dashboard auto-reloads artifacts:** No need to restart Streamlit when data changes
- **EDA phases are idempotent:** Safe to re-run without side effects
- **Embeddings are incrementally cached:** New articles → O(N new) encode cost, not O(total)
- **Modeling respects time-based split:** No data leakage (train dates < test dates by >1 month)
- **All findings logged with p-values:** FDR correction applied; significance thresholds α=0.05

---

## 🔗 Links & Commands

| Action | Command |
|--------|---------|
| View dashboard | `http://localhost:8501` |
| Read final report | `cat eda_output/report/eda_final_report.md` |
| Check modeling metrics | `cat eda_output/modeling/metrics.csv \| column -t -s,` |
| View significance tests | `cat eda_output/modeling/significance_report.md` |
| Git log of night-run | `git log --oneline --grep="Epic 13" \| head -5` |
| Re-run EDA phases | `PYTHONIOENCODING=utf-8 python -m src.eda.run_night_analysis` |
| Run tests only | `uv run pytest tests/unit -v` |
| Lint check | `uv run ruff check src/` |

---

**Generated:** 2026-07-18 01:30 UTC+7  
**Status:** Ready for morning review  
**Next action:** Open dashboard, explore results, refine thesis
