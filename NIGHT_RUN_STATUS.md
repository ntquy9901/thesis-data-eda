# Night-Run Status — 2026-07-18

**Started:** 2026-07-18 01:04 (UTC+7)  
**User:** Autonomous (user sleeping)  
**Target completion:** before morning wake

## Execution Plan

### Epic 13: Night-Run Comprehensive Analysis

**Story 13.1: Regression Tests** (in-progress)
- Command: `uv run pytest tests/unit -v --tb=short`
- Expected output: All unit tests pass
- Artifacts: Test logs in background task

**Story 13.2: Embedding Generation & Validation** (in-progress)
- Command: Via `src.features.news_embeddings.build_comparable_group_embeddings()`
- Expected output: PhoBERT embeddings cached/validated for khach_quan + tong_hop groups
- Artifacts: `data/features/news_emb_articles_*.parquet`, PCA-reduced views

**Story 13.3: EDA & PCA Analysis** (in-progress)
- Phases: 11–16 (embedding EDA, correlation, novelty, uncertainty, decay, extended-horizon)
- Expected output: All eda_output/news_embedding/ + eda_output/uncertainty/ artifacts
- Artifacts per phase:
  - Phase 11: source_stats.csv, coverage.csv, group_similarity.json, group_scatter.png
  - Phase 12: embedding_price_corr.csv, embedding_price_corr_summary.json
  - Phase 13: novelty_price_corr.csv
  - Phase 14: uncertainty_index.csv, uncertainty_price_corr.csv, uncertainty_price_corr_summary.json
  - Phase 15: decay_price_corr.csv
  - Phase 16: extended_horizon_corr.csv

**Story 13.4: Correlation Analysis** (in-progress)
- Via `src.modeling.baseline.train_and_compare_all()`
- Expected output: eda_output/modeling/metrics.csv, comparison_report.md
- Metrics: RMSE, MAE, R², QLIKE, directional accuracy

**Story 13.5: Dashboard Results Update** (pending)
- Dashboard reloaded at: http://localhost:8501
- New pages: Novelty Correlation, Uncertainty Index, Temporal Decay
- No dashboard restart needed (reads artifacts dynamically)
- All pages auto-load when artifacts exist

## Orchestrator

Script: `src/eda/run_night_analysis.py`  
Background task ID: `bzdzmbruo`  
Log files:
- Main log: `reports/2026-07-18_0104_night_analysis.md`
- Night run log: `reports/night_analysis_log.md`
- Summary JSON: `reports/latest_night_run_summary.json`

## Updates Made (Pre-Execution)

### Dashboard Enhancements
✓ Added 3 new pages in `src/dashboard/app.py`:
- `page_novelty_correlation()` — Phase 13 findings
- `page_uncertainty_index()` — Phase 14 findings
- `page_temporal_decay()` — Phase 15 findings

✓ Added data loaders in `src/dashboard/data.py`:
- `load_novelty_price_corr()`
- `load_uncertainty_price_corr()`
- `load_decay_price_corr()`

✓ Updated PAGES dictionary to include 3 new pages

### Report Enhancement
✓ Extended `src/eda/report.py` to aggregate phases 11–16 findings:
- Embedding-price correlation summary
- Novelty-based correlation results
- Uncertainty index prevalence
- Temporal decay signal analysis
- Extended-horizon correlation insights

### Sprint Tracking
✓ Updated `_bmad-output/implementation-artifacts/sprint-status.yaml`:
- Epic 13: in-progress
- Stories 13.1–13.5: in-progress

## How to Check Results (Morning)

### Option 1: View Dashboard
```bash
cd D:\bmad-projects\thesis\data_eda
uv run streamlit run src/dashboard/app.py
```
Then navigate to http://localhost:8501

### Option 2: Read Summary Report
```bash
cat reports/latest_night_run_summary.json | jq .
```

### Option 3: Read Markdown Log
```bash
cat reports/2026-07-18_0104_night_analysis.md
```

### Option 4: View EDA Artifacts
```bash
ls -lh eda_output/news_embedding/
ls -lh eda_output/uncertainty/
ls -lh eda_output/modeling/
```

## Expected Results

**✓ All tests pass** — no regressions  
**✓ Embeddings generated** — khach_quan + tong_hop groups cached  
**✓ All EDA phases complete** — phases 11–16 artifacts in eda_output/  
**✓ Correlation analysis** — modeling metrics show embedding signal  
**✓ PCA reduction** — 32-dim embeddings from PhoBERT [CLS] tokens  
**✓ Dashboard ready** — 11 pages (Overview, Price EDA, News EDA, News Embedding, Embedding Correlation, Novelty Correlation, Uncertainty Index, Temporal Decay, Đọc tin tức, Modeling, Significance)  

## Known Issues & Workarounds

### UnicodeEncodeError on Windows
- **Cause:** Emoji characters (▶, ⚠) in print statements piped to cp1252-encoded output
- **Workaround:** Set `PYTHONIOENCODING=utf-8` before running
- **Already applied:** In run_night_analysis.py via os.environ

### Potential Data Gaps
- If news_embedding cache is very large, first run may be slow
- If new tickers added to config.EDA_TICKERS, modeling will expand automatically
- Phase 16 (extended-horizon) only appears if explicitly run (not auto-triggered by phase16_exists check)

## Next Steps (If Failures Occur)

1. **Tests fail:** Check `reports/2026-07-18_0104_night_analysis.md` → re-run specific test module
2. **Embeddings fail:** Check if crawl_data exists and news_emb_articles_*.parquet can be created
3. **EDA phases fail:** Check individual phase output in eda_output/; may be missing input data
4. **Dashboard won't load:** Verify streamlit installed (`uv sync` includes it)

---

**Status as of:** 2026-07-18 01:05 UTC+7  
**Next check:** Morning when user wakes up  
**Dashboard URL:** http://localhost:8501
