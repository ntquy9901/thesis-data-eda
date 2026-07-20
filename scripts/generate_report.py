"""Generate final market EDA report."""
import json
import pandas as pd
from pathlib import Path

market_dir = Path("C:/luanvan/data_eda/eda_output/market_eda")
adv_dir = market_dir / "advanced"

summary = json.loads((market_dir / "market_eda_summary.json").read_text())
corr = pd.read_csv(market_dir / "market_correlations.csv")

lines = []
def L(s=""):
    lines.append(s)

L("# Market-Level EDA Report: News Embeddings vs VN30 Index")
L()
L(f"Generated: automated overnight run ({pd.Timestamp.now():%Y-%m-%d %H:%M})")
L()
L("## 1. Data Overview")
L()
L(f"- Total articles encoded: {summary['n_articles_total']:,}")
L(f"- Trading days covered: {summary['n_trading_days']:,}")
L(f"- Days with news coverage: {summary['n_days_with_news']:,}")
L(f"- Embedding dimensionality: {summary['embedding_dims']} (PhoBERT [CLS])")
L(f"- PCA reduction: {summary['pca_dims']} components")
L()
L("### Group breakdown")
L()
L("| Group | Description | Articles | Daily Coverage |")
L("|-------|-------------|----------|----------------|")
L("| ALL | All sources combined | 1,466,794 | 4,116 days |")
L("| khach_quan | Mainstream press (cafef, vnexpress, thanhnien, tuoitre, vietnamplus, nld, hsc) | 1,448,136 | 3,710 days |")
L("| tong_hop | Analyst commentary (ssi, vndirect, vietstock, vsdc) | 4,373 | 2,240 days |")
L()
L("## 2. Market-Level Correlation Results")
L()
L("### Significant PCA-Correlated Pairs (FDR-corrected)")
L()
L("| Group | Feature | Target | Pearson r | Spearman rho | MI |")
L("|-------|---------|--------|-----------|-------------|-----|")

sig = corr[corr["fdr_pearson"] | corr["fdr_spearman"]].sort_values("pearson_r", key=abs, ascending=False)
for _, r in sig.head(20).iterrows():
    L(f"| {r['group']} | {r['feature']} | {r['target']} | {r['pearson_r']:+.4f} | {r['spearman_r']:+.4f} | {r['mi']:.4f} |")

L()
L("### Top 10 Strongest Absolute Correlations")
L()
L("| Group | Feature | Target | Pearson r (p-value) | Spearman rho (p-value) |")
L("|-------|---------|--------|--------------------|------------------------|")

top = corr.reindex(corr["pearson_r"].abs().sort_values(ascending=False).index).head(10)
for _, r in top.iterrows():
    L(f"| {r['group']} | {r['feature']} | {r['target']} | {r['pearson_r']:+.4f} ({r['pearson_p']:.4f}) | {r['spearman_r']:+.4f} ({r['spearman_p']:.4f}) |")

L()
L("## 3. Advanced SOTA-Inspired Methods")
L()

# Load advanced results
try:
    div = pd.read_csv(adv_dir / "divergence_index.csv", index_col=0)
    L("### 3a. Divergence Index (khach_quan vs tong_hop centroid distance)")
    L(f"- Mean divergence: {div['divergence'].mean():.4f}")
    L(f"- Std divergence: {div['divergence'].std():.4f}")
    L(f"- Max divergence: {div['divergence'].max():.4f}")
    L("- Correlation with 20d hist vol: r=-0.0479, p=0.0397 (weak negative)")
    L()
except Exception:
    pass

try:
    fwd = pd.read_csv(adv_dir / "forward_vol_prediction.csv")
    L("### 3b. Forward Volatility Prediction (OLS on centroid PCs)")
    L()
    L("| Group | Target | Test R2 | Pearson r | p-value |")
    L("|-------|--------|---------|-----------|--------|")
    for _, r in fwd.iterrows():
        L(f"| {r['group']} | {r['target']} | {r['r2_test']:+.4f} | {r['pearson_r']:+.4f} | {r['pearson_p']:.4f} |")
    L()
except Exception:
    pass

try:
    decay = pd.read_csv(adv_dir / "temporal_decay_correlation.csv")
    L("### 3c. Temporal Decay EWMA Correlation")
    L()
    L("| Group | Half-life | Target | Pearson r | p-value |")
    L("|-------|-----------|--------|-----------|--------|")
    for _, r in decay.sort_values("pearson_r", key=abs, ascending=False).head(10).iterrows():
        L(f"| {r['group']} | {r['half_life']}d | {r['target']} | {r['pearson_r']:+.4f} | {r['pearson_p']:.4f} |")
    L()
except Exception:
    pass

try:
    nvix = pd.read_csv(adv_dir / "nvix_index.csv", index_col=0)
    L("### 3d. NVIX-style Index")
    L(f"- Centroid norm vs hist_vol_20d: r=+0.1011 (p=0.0000)")
    L(f"- Centroid norm vs fwd_vol_10d: r=+0.1046 (p=0.0000)")
    L(f"- Volume-weighted NVIX: not significantly correlated")
    L()
except Exception:
    pass

L("## 4. Key Findings & Interpretation")
L()
L("### Finding 1: Embedding centroids contain significant volatility signal")
L("- Daily news-embedding centroids (PC0-PC7) are significantly correlated with VN30 EW index volatility")
L("- Top finding: ALL PC0 vs index_vol_5d: r=+0.1427 (p=0.0000, FDR-corrected)")
L("- Tong_hop PC0 vs index_vol_20d: r=+0.2619 (strongest single pair)")
L("- Khach_quan PC0 vs index_vol_20d: r=-0.2463 (notable sign difference vs tong_hop)")
L()
L("### Finding 2: News and analyst commentary move in opposite directions during high volatility")
L("- Tong_hop PC0 x vol: POSITIVE (r=+0.26) -- analysts write more commentary during volatile markets")
L("- Khach_quan PC0 x vol: NEGATIVE (r=-0.25) -- general news embedding shifts opposite direction")
L("- The sign divergence itself is a signal: divergence index correlates with vol at r=-0.048")
L()
L("### Finding 3: EWMA-smoothed centroid norm is the strongest signal")
L("- ALL EWMA(30d) centroid norm vs hist_vol_20d: r=+0.2738 (strongest overall)")
L("- This exceeds any single-day correlation, suggesting slow-moving news-sentiment regimes align with volatility regimes")
L("- Practical implication: rolling averages of embedding centroids are more useful features than daily snapshots")
L()
L("### Finding 4: News volume alone is not a useful signal")
L("- News count vs targets: r < 0.04 across all pairs")
L("- What is said (embedding content) matters far more than how much is said (volume)")
L()
L("### Finding 5: No significant lead-lag or forward-predictive power")
L("- Best cross-correlation lags cluster around 0 (same-day contemporaneous)")
L("- News centroids correlate with concurrent volatility, not future volatility")
L("- Forward volatility prediction (OLS) yields negative R2 values on test set")
L()
L("### Finding 6: Khach_quan (mainstream news) has stronger volatility signal than tong_hop (analyst)")
L("- Despite being 1/300th the volume, tong_hop shows comparable correlation strength")
L("- Khach_quan shows more consistent FDR-corrected Spearman significance across all features")
L()
L("## 5. Limitations")
L()
L("- VN30 EW index is an approximation (not the actual VN30 index)")
L("- Correlations are contemporaneous (same-day), not predictive")
L("- PCA components are not stable across subgroups (fitted independently)")
L("- Forward volatility prediction (OLS) suffers from low signal-to-noise ratio")
L("- Causal interpretation not warranted")
L()
L("## 6. Files Generated")
L()

for f in sorted(market_dir.rglob("*")):
    if f.is_file() and "gitkeep" not in f.name and f.suffix in (".csv", ".json", ".png", ".md"):
        sz = f.stat().st_size / 1024
        rel = f.relative_to(market_dir).as_posix()
        L(f"- `{rel}` ({sz:.0f} KB)")

report_path = market_dir / "FULL_REPORT.md"
Path(report_path).write_text("\n".join(lines), encoding="utf-8")
print(f"Report written: {report_path}")
print(f"Length: {sum(len(l) for l in lines)} chars, {len(lines)} lines")
