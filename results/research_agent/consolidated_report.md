# Research Agent — Consolidated Report
Generated: 2026-07-22T23:23:34.401379+00:00
Total experiments: 32

## Experiments by Category

- **feature_compare**: 1 runs
- **horizon_compare**: 1 runs
- **model_compare**: 1 runs
- **sentiment**: 29 runs


## Category: feature_compare (1 runs)

| Method | Mean R² | Min R² | Max R² | Runs |
|--------|---------|--------|--------|------|
| feature_price_news_basic | 0.279845 | 0.279845 | 0.279845 | 1 |

**Best in feature_compare:** feature_price_news_basic (r2=0.279845)

## Category: horizon_compare (1 runs)

| Method | Mean R² | Min R² | Max R² | Runs |
|--------|---------|--------|--------|------|
| horizon_pk_t+1_price_news_basic | 0.279845 | 0.279845 | 0.279845 | 1 |

**Best in horizon_compare:** horizon_pk_t+1_price_news_basic (r2=0.279845)

## Category: model_compare (1 runs)

| Method | Mean R² | Min R² | Max R² | Runs |
|--------|---------|--------|--------|------|
| ridge_linear | 0.279886 | 0.279886 | 0.279886 | 1 |

**Best in model_compare:** ridge_linear (r2=0.279886)

## Category: sentiment (29 runs)

| Method | Mean R² | Min R² | Max R² | Runs |
|--------|---------|--------|--------|------|
| textblob_sentiment | 0.279150 | 0.279150 | 0.279150 | 14 |
| vader_sentiment | 0.279254 | 0.279254 | 0.279254 | 15 |

**Best in sentiment:** vader_sentiment (r2=0.279254)

## Top-5 Best Runs Overall

| Rank | Name | Category | R² | RMSE |
|------|------|----------|-----|------|
| 1 | ridge_linear | model_compare | 0.279886 | 0.00049072 |
| 2 | horizon_pk_t+1_price_news_basic | horizon_compare | 0.279845 | 0.00049073 |
| 3 | feature_price_news_basic | feature_compare | 0.279845 | 0.00049073 |
| 4 | vader_sentiment | sentiment | 0.279254 | 0.00049093 |
| 5 | vader_sentiment | sentiment | 0.279254 | 0.00049093 |

## Agent Status

| Agent | Log | Status |
|-------|-----|--------|
| Sentiment | agent_sentiment.log | `2026-07-23 06:21:49,329 [INFO] Cycle 1 done. Sleep 1800s` |
| Model | agent_model.log | `2026-07-23 06:20:43,277 [INFO]   Config: model=ridge set=price+news_adv_dual target=pk_t+1` |
| Feature | agent_feature.log | `2026-07-23 06:21:27,092 [INFO] Cycle 1 done. Sleep 3600s` |
| Horizon | agent_horizon.log | `2026-07-23 06:21:27,079 [INFO] Cycle 1 done. Sleep 3600s` |
| Ticker | agent_ticker.log | Not started |