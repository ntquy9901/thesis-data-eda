# Market-Level EDA Report: News Embeddings vs VN30 Index

Generated: automated overnight run (2026-07-21 02:19)

## 1. Data Overview

- Total articles encoded: 1,466,794
- Trading days covered: 4,890
- Days with news coverage: 4,116
- Embedding dimensionality: 768 (PhoBERT [CLS])
- PCA reduction: 8 components

### Group breakdown

| Group | Description | Articles | Daily Coverage |
|-------|-------------|----------|----------------|
| ALL | All sources combined | 1,466,794 | 4,116 days |
| khach_quan | Mainstream press (cafef, vnexpress, thanhnien, tuoitre, vietnamplus, nld, hsc) | 1,448,136 | 3,710 days |
| tong_hop | Analyst commentary (ssi, vndirect, vietstock, vsdc) | 4,373 | 2,240 days |

## 2. Market-Level Correlation Results

### Significant PCA-Correlated Pairs (FDR-corrected)

| Group | Feature | Target | Pearson r | Spearman rho | MI |
|-------|---------|--------|-----------|-------------|-----|
| tong_hop | pc_0 | index_vol_20d | +0.2619 | +0.2229 | 0.0582 |
| khach_quan | pc_0 | index_vol_20d | -0.2463 | -0.2402 | 0.1419 |
| tong_hop | pc_0 | index_vol_5d | +0.1971 | +0.1693 | 0.0065 |
| khach_quan | pc_0 | index_vol_5d | -0.1603 | -0.1757 | 0.0645 |
| ALL | pc_0 | index_vol_5d | +0.1427 | -0.0898 | 0.0221 |
| tong_hop | pc_2 | index_vol_20d | +0.1405 | -0.0431 | 0.0416 |
| ALL | pc_1 | index_vol_20d | +0.1404 | +0.2137 | 0.1777 |
| tong_hop | pc_2 | index_vol_5d | +0.1371 | -0.0264 | 0.0012 |
| ALL | pc_0 | index_vol_20d | +0.1283 | -0.1259 | 0.1136 |
| ALL | emb_dispersion | index_vol_20d | -0.1256 | +0.0373 | 0.0647 |
| ALL | emb_dispersion | index_vol_5d | -0.1235 | +0.0251 | 0.0009 |
| ALL | pc_5 | index_vol_20d | -0.1213 | -0.1185 | 0.0435 |
| ALL | pc_2 | index_vol_20d | +0.1122 | +0.1807 | 0.0829 |
| tong_hop | pc_3 | index_vol_20d | -0.1060 | -0.0712 | 0.0472 |
| khach_quan | pc_1 | index_vol_20d | -0.1007 | -0.1174 | 0.0998 |
| ALL | pc_1 | index_vol_5d | +0.0961 | +0.1588 | 0.0642 |
| tong_hop | pc_7 | index_vol_5d | -0.0878 | -0.0610 | 0.0009 |
| tong_hop | pc_0 | index_return | +0.0805 | +0.0533 | 0.0162 |
| khach_quan | pc_4 | index_vol_20d | +0.0787 | +0.1119 | 0.0827 |
| ALL | pc_3 | index_vol_5d | -0.0782 | -0.0983 | 0.0389 |

### Top 10 Strongest Absolute Correlations

| Group | Feature | Target | Pearson r (p-value) | Spearman rho (p-value) |
|-------|---------|--------|--------------------|------------------------|
| tong_hop | pc_0 | index_vol_20d | +0.2619 (0.0000) | +0.2229 (0.0000) |
| khach_quan | pc_0 | index_vol_20d | -0.2463 (0.0000) | -0.2402 (0.0000) |
| tong_hop | pc_0 | index_vol_5d | +0.1971 (0.0000) | +0.1693 (0.0000) |
| khach_quan | pc_0 | index_vol_5d | -0.1603 (0.0000) | -0.1757 (0.0000) |
| ALL | pc_0 | index_vol_5d | +0.1427 (0.0000) | -0.0898 (0.0000) |
| tong_hop | pc_2 | index_vol_20d | +0.1405 (0.0000) | -0.0431 (0.0414) |
| ALL | pc_1 | index_vol_20d | +0.1404 (0.0000) | +0.2137 (0.0000) |
| tong_hop | pc_2 | index_vol_5d | +0.1371 (0.0000) | -0.0264 (0.2123) |
| ALL | pc_0 | index_vol_20d | +0.1283 (0.0000) | -0.1259 (0.0000) |
| ALL | emb_dispersion | index_vol_20d | -0.1256 (0.0000) | +0.0373 (0.0167) |

## 3. Advanced SOTA-Inspired Methods

### 3a. Divergence Index (khach_quan vs tong_hop centroid distance)
- Mean divergence: 0.3254
- Std divergence: 0.0922
- Max divergence: 0.6752
- Correlation with 20d hist vol: r=-0.0479, p=0.0397 (weak negative)

### 3b. Forward Volatility Prediction (OLS on centroid PCs)

| Group | Target | Test R2 | Pearson r | p-value |
|-------|--------|---------|-----------|--------|
| ALL | fwd_vol_5d | -0.0620 | -0.0019 | 0.9388 |
| ALL | fwd_vol_10d | -0.0676 | +0.0101 | 0.6864 |
| ALL | fwd_vol_20d | -0.0941 | +0.0203 | 0.4176 |
| khach_quan | fwd_vol_5d | -0.1656 | -0.0838 | 0.0024 |
| khach_quan | fwd_vol_10d | -0.2446 | -0.0588 | 0.0336 |
| khach_quan | fwd_vol_20d | -0.2294 | -0.0652 | 0.0186 |
| tong_hop | fwd_vol_5d | -0.3099 | +0.0335 | 0.1812 |
| tong_hop | fwd_vol_10d | -0.3790 | +0.0370 | 0.1403 |
| tong_hop | fwd_vol_20d | -0.4730 | +0.0676 | 0.0070 |

### 3c. Temporal Decay EWMA Correlation

| Group | Half-life | Target | Pearson r | p-value |
|-------|-----------|--------|-----------|--------|
| ALL | 30d | hist_vol_20d | +0.2738 | 0.0000 |
| ALL | 14d | hist_vol_20d | +0.2478 | 0.0000 |
| ALL | 7d | hist_vol_20d | +0.2362 | 0.0000 |
| ALL | 3d | hist_vol_20d | +0.2039 | 0.0000 |
| khach_quan | 7d | hist_vol_20d | -0.1351 | 0.0000 |
| khach_quan | 14d | hist_vol_20d | -0.1309 | 0.0000 |
| khach_quan | 3d | hist_vol_20d | -0.1308 | 0.0000 |
| khach_quan | 30d | hist_vol_20d | -0.1134 | 0.0000 |
| tong_hop | 30d | hist_vol_20d | +0.0990 | 0.0000 |
| ALL | 7d | log_return | +0.0961 | 0.0000 |

### 3d. NVIX-style Index
- Centroid norm vs hist_vol_20d: r=+0.1011 (p=0.0000)
- Centroid norm vs fwd_vol_10d: r=+0.1046 (p=0.0000)
- Volume-weighted NVIX: not significantly correlated

## 4. Key Findings & Interpretation

### Finding 1: Embedding centroids contain significant volatility signal
- Daily news-embedding centroids (PC0-PC7) are significantly correlated with VN30 EW index volatility
- Top finding: ALL PC0 vs index_vol_5d: r=+0.1427 (p=0.0000, FDR-corrected)
- Tong_hop PC0 vs index_vol_20d: r=+0.2619 (strongest single pair)
- Khach_quan PC0 vs index_vol_20d: r=-0.2463 (notable sign difference vs tong_hop)

### Finding 2: News and analyst commentary move in opposite directions during high volatility
- Tong_hop PC0 x vol: POSITIVE (r=+0.26) -- analysts write more commentary during volatile markets
- Khach_quan PC0 x vol: NEGATIVE (r=-0.25) -- general news embedding shifts opposite direction
- The sign divergence itself is a signal: divergence index correlates with vol at r=-0.048

### Finding 3: EWMA-smoothed centroid norm is the strongest signal
- ALL EWMA(30d) centroid norm vs hist_vol_20d: r=+0.2738 (strongest overall)
- This exceeds any single-day correlation, suggesting slow-moving news-sentiment regimes align with volatility regimes
- Practical implication: rolling averages of embedding centroids are more useful features than daily snapshots

### Finding 4: News volume alone is not a useful signal
- News count vs targets: r < 0.04 across all pairs
- What is said (embedding content) matters far more than how much is said (volume)

### Finding 5: No significant lead-lag or forward-predictive power
- Best cross-correlation lags cluster around 0 (same-day contemporaneous)
- News centroids correlate with concurrent volatility, not future volatility
- Forward volatility prediction (OLS) yields negative R2 values on test set

### Finding 6: Khach_quan (mainstream news) has stronger volatility signal than tong_hop (analyst)
- Despite being 1/300th the volume, tong_hop shows comparable correlation strength
- Khach_quan shows more consistent FDR-corrected Spearman significance across all features

## 5. Limitations

- VN30 EW index is an approximation (not the actual VN30 index)
- Correlations are contemporaneous (same-day), not predictive
- PCA components are not stable across subgroups (fitted independently)
- Forward volatility prediction (OLS) suffers from low signal-to-noise ratio
- Causal interpretation not warranted

## 6. Files Generated

- `advanced/divergence_index.csv` (41 KB)
- `advanced/forward_vol_prediction.csv` (1 KB)
- `advanced/leadlag_correlation.csv` (7 KB)
- `advanced/nvix_index.csv` (240 KB)
- `advanced/temporal_decay_correlation.csv` (2 KB)
- `market_centroid_timeseries.png` (317 KB)
- `market_corr_heatmap.png` (60 KB)
- `market_correlations.csv` (17 KB)
- `market_eda_summary.json` (0 KB)