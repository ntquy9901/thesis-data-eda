# Night Analysis Run

Started: 2026-07-18T09:55:01.420034

Task 1 (regression tests) SKIPPED (--skip-tests)


======================================================================
NIGHT ANALYSIS PIPELINE - START
======================================================================

======================================================================
TASK 1: Running Regression Tests
======================================================================
[2026-07-18 09:55:01] [INFO] SKIPPED (--skip-tests flag set)

======================================================================
TASK 2: Embedding Generation & Validation
======================================================================
[2026-07-18 09:55:01] [INFO] Loading news embeddings...
[2026-07-18 09:55:30] [INFO]   Embeddings [khach_quan]: (4, 43) (rows, cols)
[2026-07-18 09:55:30] [INFO]   Embeddings [tong_hop]: (2221, 43) (rows, cols)

======================================================================
TASK 3: Running EDA Phases 11-16
======================================================================
[2026-07-18 09:55:30] [INFO] Running Phase 11: News Embedding EDA...
[2026-07-18 09:56:10] [INFO]   SUCCESS - Phase 11: News Embedding EDA
[2026-07-18 09:56:10] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\source_stats.csv
[2026-07-18 09:56:10] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\embedding_coverage.csv
[2026-07-18 09:56:10] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\group_similarity.json
[2026-07-18 09:56:10] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\group_scatter.png
[2026-07-18 09:56:10] [INFO] Running Phase 12: Embedding-Price Correlation...
[2026-07-18 09:56:17] [INFO]   SUCCESS - Phase 12: Embedding-Price Correlation
[2026-07-18 09:56:17] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\embedding_price_corr.csv
[2026-07-18 09:56:17] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\embedding_price_corr_summary.json
[2026-07-18 09:56:17] [INFO] Running Phase 13: Novelty Correlation...
[2026-07-18 09:56:40] [INFO]   SUCCESS - Phase 13: Novelty Correlation
[2026-07-18 09:56:40] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\novelty_price_corr.csv
[2026-07-18 09:56:40] [INFO] Running Phase 14: Uncertainty Index...
[2026-07-18 09:56:43] [INFO]   SUCCESS - Phase 14: Uncertainty Index
[2026-07-18 09:56:43] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\uncertainty\uncertainty_index.csv
[2026-07-18 09:56:43] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\uncertainty\uncertainty_price_corr.csv
[2026-07-18 09:56:43] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\uncertainty\uncertainty_price_corr_summary.json
[2026-07-18 09:56:43] [INFO] Running Phase 15: Temporal Decay Correlation...
[2026-07-18 09:58:01] [INFO]   SUCCESS - Phase 15: Temporal Decay Correlation
[2026-07-18 09:58:01] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\decay_price_corr.csv
[2026-07-18 09:58:01] [INFO] Running Phase 16: Extended Horizon Correlation...
[2026-07-18 09:58:03] [INFO]   SUCCESS - Phase 16: Extended Horizon Correlation
[2026-07-18 09:58:03] [INFO]     -> D:\bmad-projects\thesis\data_eda\eda_output\news_embedding\extended_horizon_corr.csv

======================================================================
TASK 4: Modeling & Regression Analysis
======================================================================
[2026-07-18 09:58:03] [INFO] Running baseline modeling comparison...
[2026-07-18 09:58:15] [INFO]   Baseline models complete: [WindowsPath('D:/bmad-projects/thesis/data_eda/eda_output/modeling/metrics.csv'), WindowsPath('D:/bmad-projects/thesis/data_eda/eda_output/modeling/comparison_report.md')]

======================================================================
TASK 5: PCA Analysis
======================================================================
[2026-07-18 09:58:15] [INFO] Performing PCA reduction...
[2026-07-18 09:58:39] [INFO]   PCA[khach_quan]: 4 samples, 32 dims
[2026-07-18 09:58:39] [INFO]   PCA[tong_hop]: 2221 samples, 32 dims

======================================================================
TASK 6: Dashboard Update
======================================================================
[2026-07-18 09:58:39] [INFO] Starting: Dashboard startup check
[2026-07-18 09:58:49] [INFO] PASS: Dashboard startup check
[2026-07-18 09:58:49] [INFO] Output:

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.6:8501



======================================================================
TASK 7: Summary Report Generation
======================================================================
[2026-07-18 09:58:49] [INFO] Generating final EDA report...
[2026-07-18 09:58:50] [INFO]   Report generated: [WindowsPath('D:/bmad-projects/thesis/data_eda/eda_output/report/candidate_features.csv'), WindowsPath('D:/bmad-projects/thesis/data_eda/eda_output/report/eda_final_report.md')]

======================================================================
NIGHT ANALYSIS PIPELINE - COMPLETE
======================================================================
[2026-07-18 09:58:50] [INFO] 
Duration: 228.6s (3.8m)
[2026-07-18 09:58:50] [INFO] 
Summary:
[2026-07-18 09:58:50] [INFO] {
  "started": "2026-07-18T09:55:01.420034",
  "completed": "2026-07-18T09:58:50.018041",
  "duration_seconds": 228.598007,
  "results": {
    "regression_tests": null,
    "embeddings": true,
    "eda_phases": {
      "phase11_news_embedding_eda": true,
      "phase12_embedding_price_correlation": true,
      "phase13_novelty_correlation": true,
      "phase14_uncertainty_index": true,
      "phase15_temporal_decay_correlation": true,
      "phase16_extended_horizon_correlation": true
    },
    "modeling": true,
    "pca": true,
    "dashboard": true,
    "report": true
  }
}
[2026-07-18 09:58:50] [INFO] 
Log file: D:\bmad-projects\thesis\data_eda\reports\2026-07-18_0955_night_analysis.md
[2026-07-18 09:58:50] [INFO] Summary JSON: D:\bmad-projects\thesis\data_eda\reports\latest_night_run_summary.json
[2026-07-18 09:58:50] [INFO] 
Results ready for dashboard at http://localhost:8501
[2026-07-18 09:58:50] [INFO] 
Passed: 6/12 tasks (1 skipped)
