"""Standalone GPU-accelerated re-encode of ALL news articles with PhoBERT.

This script uses a separate Python 3.12 venv with torch CUDA to encode every
article (not just VN30-ticker-mentioning ones) and writes to the same per-source
parquet cache format as src.features.news_embeddings.

Usage: .venv-gpu\Scripts\python.exe scripts/reencode_all_articles_gpu.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from src.data.discover_news import discover_source_files, load_source
from src.nlp.embeddings import extract_phobert_embeddings

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
BATCH_SIZE = 64


def reencode_source(source: str, path: Path) -> int:
    """Encode ALL articles for one source, skip already-cached URLs."""
    cache_path = FEATURES_DIR / f"news_emb_articles_{source}.parquet"

    df = load_source(source, path)
    if df.empty:
        return 0

    title = df.get("title", pd.Series(index=df.index)).fillna("")
    lead = df.get("lead", pd.Series(index=df.index)).fillna("")
    df["_text"] = (title.astype(str) + " " + lead.astype(str)).str.strip()
    df = df[df["_text"].str.len() > 0].reset_index(drop=True)
    if "url" not in df.columns:
        df = df.reset_index(drop=False)
        if "url" not in df.columns:
            return 0
    df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)

    cached = pd.DataFrame({"url": []})
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
        except Exception:
            cached = pd.DataFrame({"url": []})

    known = set(cached["url"]) if not cached.empty else set()
    new_rows = df[~df["url"].isin(known)]
    if new_rows.empty:
        return len(cached)

    embs = extract_phobert_embeddings(new_rows["_text"].tolist(), batch_size=BATCH_SIZE)
    raw_cols = {f"raw_{i}": embs[:, i] for i in range(embs.shape[1])}
    new_df = pd.DataFrame({"url": new_rows["url"].values, **raw_cols})
    merged = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, index=False)
    n_new = len(new_df)
    n_total = len(merged)
    print(f"  {source}: +{n_new} new, {n_total} total, file={cache_path.name}")
    return n_new


def main():
    import torch
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    files = discover_source_files()
    print(f"Discovered {len(files)} sources\n")

    total_new = 0
    for source, path in files.items():
        print(f"[{source}] {path.name}...")
        try:
            n = reencode_source(source, path)
            total_new += n
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {total_new} new articles encoded across {len(files)} sources.")
    print(f"All cache files: {FEATURES_DIR}")


if __name__ == "__main__":
    main()
