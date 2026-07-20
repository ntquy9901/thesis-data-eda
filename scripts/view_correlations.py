"""View market EDA correlation results."""
import pandas as pd

corr = pd.read_csv("C:/luanvan/data_eda/eda_output/market_eda/market_correlations.csv")

print("=== SIGNIFICANT CORRELATIONS (FDR-corrected) ===\n")
sig = corr[corr["fdr_pearson"] | corr["fdr_spearman"]].sort_values(["group", "feature", "target"])
for _, r in sig.iterrows():
    p_sig = "P" if r["fdr_pearson"] else " "
    s_sig = "S" if r["fdr_spearman"] else " "
    print(f"{r['group']:12s} {r['feature']:20s} | {r['target']:20s} | "
          f"r={r['pearson_r']:+.4f} rho={r['spearman_r']:+.4f} mi={r['mi']:.4f} | {p_sig}{s_sig}")

print()
print("=== FEATURE SIGNIFICANCE COUNTS BY GROUP ===\n")
for g in ["ALL", "khach_quan", "tong_hop"]:
    gdf = corr[corr["group"] == g]
    p_sig = gdf["fdr_pearson"].sum()
    s_sig = gdf["fdr_spearman"].sum()
    print(f"{g:12s}: {p_sig}/{len(gdf)} Pearson-sig, {s_sig}/{len(gdf)} Spearman-sig")

print()
print("=== TOP CORRELATIONS (abs Pearson r) ===\n")
top = corr.reindex(corr["pearson_r"].abs().sort_values(ascending=False).index).head(15)
for _, r in top.iterrows():
    print(f"{r['group']:12s} {r['feature']:20s} | {r['target']:20s} | "
          f"r={r['pearson_r']:+.4f} (p={r['pearson_p']:.4f}) rho={r['spearman_r']:+.4f} (p={r['spearman_p']:.4f})")
