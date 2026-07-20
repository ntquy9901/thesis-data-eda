# Night Analysis Run

Started: 2026-07-18T01:04:11.043610


======================================================================
NIGHT ANALYSIS PIPELINE - START
======================================================================

======================================================================
TASK 1: Running Regression Tests
======================================================================
[2026-07-18 01:04:11] [INFO] Starting: Full test suite
[2026-07-18 01:17:49] [INFO] PASS: Full test suite
[2026-07-18 01:17:49] [INFO] Output:
est_phase03_price_eda.py::test_parkinson_volatility_masks_invalid
tests/unit/test_phase03_price_eda.py::test_add_returns_masks_nonfinite_from_zero_price
  D:\bmad-projects\thesis\data_eda\.venv\Lib\site-packages\pandas\core\arraylike.py:402: RuntimeWarning: divide by zero encountered in log
    result = getattr(ufunc, method)(*inputs, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 240 passed, 3 warnings in 813.55s (0:13:33) =================


======================================================================
TASK 2: Embedding Generation & Validation
======================================================================
[2026-07-18 01:17:51] [INFO] Loading news embeddings...
[2026-07-18 01:18:41] [INFO]   Embeddings [khach_quan]: (4, 43) (rows, cols)
[2026-07-18 01:18:41] [INFO]   Embeddings [tong_hop]: (2221, 43) (rows, cols)

======================================================================
TASK 3: Running EDA Phases 11-16
======================================================================
[2026-07-18 01:18:41] [INFO] Running Phase 11: News Embedding EDA...
[2026-07-18 01:19:16] [INFO]   SUCCESS - Phase 11: News Embedding EDA
[2026-07-18 01:19:16] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\source_stats.csv
[2026-07-18 01:19:16] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\embedding_coverage.csv
[2026-07-18 01:19:16] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\group_similarity.json
[2026-07-18 01:19:16] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\group_scatter.png
[2026-07-18 01:19:16] [INFO] Running Phase 12: Embedding-Price Correlation...
[2026-07-18 01:19:26] [ERROR]   FAIL - Phase 12: Embedding-Price Correlation: ModuleNotFoundError: No module named 'statsmodels'
[2026-07-18 01:19:26] [INFO] Running Phase 13: Novelty Correlation...
[2026-07-18 01:20:05] [ERROR]   FAIL - Phase 13: Novelty Correlation: ModuleNotFoundError: No module named 'statsmodels'
[2026-07-18 01:20:05] [INFO] Running Phase 14: Uncertainty Index...
[2026-07-18 01:20:12] [ERROR]   FAIL - Phase 14: Uncertainty Index: ModuleNotFoundError: No module named 'statsmodels'
[2026-07-18 01:20:12] [INFO] Running Phase 15: Temporal Decay Correlation...
[2026-07-18 01:22:43] [ERROR]   FAIL - Phase 15: Temporal Decay Correlation: ModuleNotFoundError: No module named 'statsmodels'
[2026-07-18 01:22:43] [INFO] Running Phase 16: Extended Horizon Correlation...
[2026-07-18 01:22:48] [ERROR]   FAIL - Phase 16: Extended Horizon Correlation: ModuleNotFoundError: No module named 'statsmodels'

======================================================================
TASK 4: Modeling & Regression Analysis
======================================================================
[2026-07-18 01:22:48] [ERROR] Modeling error: ImportError: cannot import name 'train_and_compare_all' from 'src.modeling.baseline' (D:\bmad-projects\thesis\data_eda\src\modeling\baseline.py)

======================================================================
TASK 5: PCA Analysis
======================================================================
[2026-07-18 01:22:48] [INFO] Performing PCA reduction...
[2026-07-18 01:23:46] [INFO]   PCA[khach_quan]: 4 samples, 32 dims
[2026-07-18 01:23:46] [ERROR] PCA error: ValueError: Found array with 0 feature(s) (shape=(949, 0)) while a minimum of 1 is required by PCA.

======================================================================
TASK 6: Dashboard Update
======================================================================
[2026-07-18 01:23:46] [INFO] Starting: Dashboard startup check
[2026-07-18 01:23:57] [INFO] PASS: Dashboard startup check
[2026-07-18 01:23:57] [INFO] Output:

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.6:8501



======================================================================
TASK 7: Summary Report Generation
======================================================================
[2026-07-18 01:23:57] [ERROR] Report generation error: ImportError: cannot import name 'generate_final_report' from 'src.eda.report' (D:\bmad-projects\thesis\data_eda\src\eda\report.py)

======================================================================
NIGHT ANALYSIS PIPELINE - COMPLETE
======================================================================
[2026-07-18 01:23:57] [INFO] 
Duration: 1186.3s (19.8m)
[2026-07-18 01:23:57] [INFO] 
Summary:
[2026-07-18 01:23:57] [INFO] {
  "started": "2026-07-18T01:04:11.043610",
  "completed": "2026-07-18T01:23:57.307011",
  "duration_seconds": 1186.263401,
  "results": {
    "regression_tests": true,
    "embeddings": true,
    "eda_phases": {
      "phase11_news_embedding_eda": true,
      "phase12_embedding_price_correlation": false,
      "phase13_novelty_correlation": false,
      "phase14_uncertainty_index": false,
      "phase15_temporal_decay_correlation": false,
      "phase16_extended_horizon_correlation": false
    },
    "modeling": false,
    "pca": false,
    "dashboard": true,
    "report": false
  }
}
[2026-07-18 01:23:57] [INFO] 
Log file: D:\bmad-projects\thesis\data_eda\reports\2026-07-18_0104_night_analysis.md
[2026-07-18 01:23:57] [INFO] Summary JSON: D:\bmad-projects\thesis\data_eda\reports\latest_night_run_summary.json
[2026-07-18 01:23:57] [INFO] 
Results ready for dashboard at http://localhost:8501
[2026-07-18 01:23:57] [INFO] 
Passed: 4/13 tasks
