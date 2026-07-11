# EDA Guide for Stock Volatility Prediction (Price + News)

## Objective

Perform a comprehensive Exploratory Data Analysis (EDA) before feature
engineering and model training for a stock volatility prediction
project.

Target examples:

-   T+1 realized volatility
-   T+5 realized volatility
-   T+10 realized volatility

------------------------------------------------------------------------

# Rules

-   Never modify raw data.
-   Work on copies.
-   Every finding must be supported by statistics or visualizations.
-   Detect possible data leakage.
-   Save every chart and summary.

Suggested output structure:

    eda_output/
        profiling/
        quality/
        price/
        news/
        relationship/
        feature_engineering/
        leakage/
        report/

------------------------------------------------------------------------

# Phase 1 -- Dataset Profiling

For every table report:

-   row count
-   column count
-   datatype
-   primary key
-   candidate key
-   memory usage

Produce a profiling table.

------------------------------------------------------------------------

# Phase 2 -- Data Quality

Check:

## Missing values

-   percentage
-   pattern
-   by stock
-   by date

## Duplicate

-   duplicated rows
-   duplicated news
-   duplicated IDs

## Invalid values

Examples:

-   negative volume
-   impossible prices
-   future timestamps
-   invalid stock codes

------------------------------------------------------------------------

# Phase 3 -- Price Data EDA

Create and analyze:

-   OHLCV summary
-   return
-   log return
-   realized volatility
-   ATR
-   rolling statistics

Visualizations:

-   histogram
-   boxplot
-   rolling volatility
-   ACF/PACF
-   correlation heatmap

Report:

-   outliers
-   volatility clustering
-   regime changes

------------------------------------------------------------------------

# Phase 4 -- News EDA

Measure:

## Coverage

-   news/day
-   news/stock
-   days without news
-   average news/day

## Publish Time

Analyze:

-   before market
-   during market
-   after market
-   weekend

Create effective_trading_date.

## Sentiment

Summaries:

-   mean
-   std
-   min
-   max
-   positive ratio
-   negative ratio

## Topics

If available:

-   earnings
-   dividend
-   M&A
-   management
-   regulation
-   macro
-   sector

## Sources

Analyze:

-   source distribution
-   duplicate news
-   source credibility
-   repost rate

------------------------------------------------------------------------

# Phase 5 -- Relationship Analysis

Evaluate:

Price \<-\> News

Examples:

-   news count vs future volatility
-   sentiment vs future volatility
-   topic vs volatility
-   negative news vs return

Methods:

-   Pearson
-   Spearman
-   Mutual Information
-   Granger Causality
-   Cross Correlation

------------------------------------------------------------------------

# Phase 6 -- Event Study

For each important news event calculate:

Before:

-   T-10
-   T-5
-   T-1

After:

-   T+1
-   T+5
-   T+10

Compare:

-   realized volatility
-   return
-   abnormal volatility

------------------------------------------------------------------------

# Phase 7 -- Sparse News Analysis

Calculate:

-   coverage ratio
-   days_since_last_news
-   news_count_1d
-   news_count_3d
-   news_count_5d

Do NOT replace missing news with sentiment=0 without distinguishing "no
news" from "neutral news".

Use:

-   news_available flag

------------------------------------------------------------------------

# Phase 8 -- Feature Engineering Validation

Evaluate:

-   missing
-   variance
-   redundancy
-   correlation
-   leakage
-   drift

Recommend removing:

-   constant features
-   duplicate features
-   highly collinear features

------------------------------------------------------------------------

# Phase 9 -- Leakage Detection

Verify:

-   publish timestamp vs trading timestamp
-   future information
-   rolling window correctness
-   target leakage
-   normalization leakage

Explicitly list every potential leakage.

------------------------------------------------------------------------

# Phase 10 -- Visualizations

Produce at minimum:

1.  Missing value heatmap
2.  News coverage by stock
3.  News count by day
4.  Sentiment distribution
5.  Return distribution
6.  Volatility distribution
7.  Rolling volatility
8.  Correlation heatmap
9.  Event study plots
10. News count vs future volatility
11. SHAP/importance (if model exists)

------------------------------------------------------------------------

# Final Report

Include:

## Executive Summary

-   data quality
-   major risks
-   key observations

## Findings

Top findings with evidence.

## Recommended Features

List candidate features for training.

## Risks

-   leakage
-   sparse data
-   imbalance
-   outliers

## Recommended Next Steps

Prioritized actions before model training.
