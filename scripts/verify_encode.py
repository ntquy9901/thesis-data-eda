"""Verify re-encode results."""
import os, pandas as pd
from pathlib import Path

d = Path("C:/luanvan/data_eda/data/features")
total = 0
for f in sorted(d.glob("news_emb_articles_*.parquet")):
    fsize = os.path.getsize(f) / 1e6
    df = pd.read_parquet(f)
    n = len(df)
    total += n
    src = f.stem.replace("news_emb_articles_", "")
    print(f"{src:35s} {n:>8,} rows  {fsize:7.1f} MB")

print("---")
print(f"{'TOTAL':35s} {total:>8,} rows")

# Check embedding stats on a sample
big = None
for f in d.glob("news_emb_articles_*.parquet"):
    df = pd.read_parquet(f)
    if len(df) > 1000:
        big = df.head(1000)
        break

if big is not None:
    emb_cols = [c for c in big.columns if c.startswith("raw_")]
    mat = big[emb_cols].values
    print(f"\nEmbedding shape: {mat.shape}")
    print(f"Range: [{mat.min():.4f}, {mat.max():.4f}]")
    print(f"Mean: {mat.mean():.4f}, Std: {mat.std():.4f}")
    print(f"First file schema cols: {list(big.columns[:5])}... ({len(emb_cols)} dims)")
