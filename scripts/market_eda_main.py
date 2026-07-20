"""
Market-Level EDA: embed all articles -> daily centroids -> correlate with VN30.

Pipeline:
  1. Load cached embeddings + merge with article metadata (pub_date, source, lead)
  2. Map pub_date to trading date
  3. Compute daily centroids (mean embedding) per group (ALL, khach_quan, tong_hop)
  4. Load VN30 equal-weighted index returns and volatility
  5. Cross-correlation (Pearson/Spearman/MI) with FDR correction
  6. Visualizations & report
"""

import json, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore")

from src.data.discover_news import discover_source_files, load_source
from config import EDA_OUTPUT_DIR

EDA_OUTPUT_DIR = EDA_OUTPUT_DIR or PROJECT_ROOT / "eda_output"
MARKET_DIR   = EDA_OUTPUT_DIR / "market_eda"
MARKET_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "khach_quan": {"cafef","hsc","vnexpress","thanhnien","tuoitre","nld","vietnamplus",
                   "thanhnien_root","thanhnien_objective",
                   "tuoitre_root","tuoitre_objective",
                   "vietnamplus_root","vietnamplus_objective"},
    "tong_hop":   {"ssi","vndirect","vnstock","vietstock","vsdc"},
}

TRAIN_CUTOFF = np.datetime64("2020-01-01")


def load_all_articles() -> pd.DataFrame:
    """Load ALL articles with embeddings from per-source cache, merged with metadata."""
    import pandas as pd
    from src.features.news_embeddings import _article_cache_path

    files = discover_source_files()
    frames = []
    t0 = time.time()
    for source, path in files.items():
        try:
            df = load_source(source, path)
        except Exception:
            continue
        if df.empty:
            continue
        title = df.get("title", pd.Series(index=df.index)).fillna("")
        lead  = df.get("lead", pd.Series(index=df.index)).fillna("")
        df["_text"] = (title.astype(str) + " " + lead.astype(str)).str.strip()
        df = df[df["_text"].str.len() > 0].reset_index(drop=True)
        if "url" not in df.columns:
            continue
        df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)

        cache = _article_cache_path(source)
        if not cache.exists():
            continue
        embs = pd.read_parquet(cache)
        merged = df.merge(embs, on="url", how="inner")
        if merged.empty:
            continue
        merged["source"] = source
        frames.append(merged)
        print(f"  {source}: {len(merged)} articles merged")

    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"  TOTAL: {len(all_df)} unique articles in {time.time()-t0:.0f}s")
    return all_df


def map_to_trading_days(df: pd.DataFrame, trading_days: np.ndarray) -> pd.DataFrame:
    """Map each article's pub_date to nearest preceding trading day."""
    pub = pd.to_datetime(df["pub_date"], errors="coerce").dt.normalize()
    arr = pub.to_numpy(dtype="datetime64[ns]")
    out = np.empty(len(arr), dtype="datetime64[ns]")
    for i, d in enumerate(arr):
        if pd.isna(d):
            out[i] = np.datetime64("NaT")
            continue
        before = trading_days[trading_days <= d]
        out[i] = before[-1] if len(before) > 0 else np.datetime64("NaT")
    df["trading_date"] = out
    return df[df["trading_date"] != np.datetime64("NaT")].reset_index(drop=True)


def _load_trading_calendar() -> np.ndarray:
    """Build trading calendar from VN30 individual stock dates."""
    pdir = Path("C:/luanvan/stock_vol_prediction01/data/raw/prices")
    all_dates = []
    for f in sorted(pdir.glob("*_ohlcv.csv")):
        try:
            dates = pd.to_datetime(pd.read_csv(f, usecols=["date"])["date"], errors="coerce").dropna()
            all_dates.append(dates)
        except Exception:
            continue
    if all_dates:
        universe = pd.concat(all_dates, ignore_index=True).drop_duplicates().sort_values()
        return universe.to_numpy(dtype="datetime64[ns]")
    return np.array([], dtype="datetime64[ns]")


def _embedding_cols(df: pd.DataFrame) -> list[str]:
    import re
    return [c for c in df.columns if re.match(r"^raw_\d+$", c)]


def daily_centroids(df: pd.DataFrame, emb_cols: list[str]) -> dict:
    """Compute daily centroids and news volume for each group."""
    groups = {"ALL": df}
    for gname, srcs in GROUPS.items():
        mask = df["source"].isin(srcs)
        if mask.any():
            groups[gname] = df[mask].copy()

    results = {}
    for gname, gdf in groups.items():
        daily = gdf.groupby("trading_date")
        centroid = daily[emb_cols].mean()
        counts   = daily.size().to_frame("news_count")
        # embedding dispersion = mean L2 distance from centroid
        disp = []
        for dt, sub in daily:
            c = centroid.loc[dt].values  # 768-dim centroid
            mat = sub[emb_cols].values
            dists = np.linalg.norm(mat - c, axis=1)
            disp.append(dists.mean())
        counts["emb_dispersion"] = disp
        panel = counts.join(centroid, how="left")
        panel.index.name = "date"
        # rename raw_* -> cent_*
        panel = panel.rename(columns={c: f"cent_{i}" for i, c in enumerate(emb_cols)})
        results[gname] = panel
    return results


def _reduce_pca(df: pd.DataFrame) -> pd.DataFrame:
    """PCA-reduce cent_* columns to 8 dimensions for correlation analysis."""
    from sklearn.decomposition import PCA
    cent_cols = [c for c in df.columns if c.startswith("cent_")]
    if len(cent_cols) < 2:
        return df
    mat = df[cent_cols].values
    train_mask = df.index < TRAIN_CUTOFF
    n_train = int(train_mask.sum())
    if n_train < 2:
        return df
    d = min(8, mat.shape[1], max(1, n_train - 1))
    pca = PCA(n_components=d).fit(mat[train_mask])
    reduced = pca.transform(mat).astype(np.float32)
    for i in range(d):
        df[f"pc_{i}"] = reduced[:, i]
    df["pca_explained_var"] = pca.explained_variance_ratio_.sum()
    return df.drop(columns=cent_cols)


def _load_vn30_index() -> pd.DataFrame:
    """Build VN30 equal-weighted index from individual stock prices."""
    import pandas as pd
    pdir = Path("C:/luanvan/stock_vol_prediction01/data/raw/prices")
    frames = []
    for f in sorted(pdir.glob("*_ohlcv.csv")):
        ticker = f.stem.replace("_ohlcv", "")
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            df["ticker"] = ticker
            frames.append(df[["date", "ticker", "close", "volume"]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    all_prices = pd.concat(frames, ignore_index=True)
    all_prices["date"] = pd.to_datetime(all_prices["date"]).dt.normalize()
    # EW index: equal-weight mean of close prices
    ew = all_prices.groupby("date").agg(index_price=("close", "mean"), tot_volume=("volume", "sum")
    ).reset_index()
    ew["index_return"] = ew["index_price"].pct_change()
    ew["index_vol_5d"] = ew["index_return"].rolling(5).std()
    ew["index_vol_20d"] = ew["index_return"].rolling(20).std()
    # log return
    ew["log_return"] = np.log(ew["index_price"] / ew["index_price"].shift(1))
    return ew.set_index("date").dropna()


def pearson_spearman(a: pd.Series, b: pd.Series) -> dict:
    from scipy.stats import pearsonr, spearmanr
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    p = pearsonr(a[mask], b[mask])
    s = spearmanr(a[mask], b[mask])
    return {"pearson_r": p.statistic, "pearson_p": p.pvalue, "spearman_r": s.statistic, "spearman_p": s.pvalue}


def mutual_information(a: pd.Series, b: pd.Series, bins: int = 20) -> float | None:
    from sklearn.feature_selection import mutual_info_regression
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return None
    return float(mutual_info_regression(a[mask].values.reshape(-1, 1), b[mask].values, random_state=0)[0])


def fdr_correct(pvals: list[float]) -> list[bool]:
    from statsmodels.stats.multitest import multipletests
    if not pvals:
        return []
    return [bool(x) for x in multipletests([p for p in pvals if p is not None], method="fdr_bh")[0]]


def correlate(df: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows, pears_p, spea_p = [], [], []
    for feat in features:
        if feat not in df.columns:
            continue
        for tgt in targets:
            if tgt not in df.columns:
                continue
            ps = pearson_spearman(df[feat], df[tgt])
            mi = mutual_information(df[feat], df[tgt])
            rows.append({"feature": feat, "target": tgt, "mi": mi, **ps})
            pears_p.append(ps["pearson_p"])
            spea_p.append(ps["spearman_p"])
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["fdr_pearson"] = fdr_correct(pears_p)
        corr["fdr_spearman"] = fdr_correct(spea_p)
    return corr


def plot_timeseries(panels: dict, targets: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = {"ALL": "black", "khach_quan": "blue", "tong_hop": "red"}

    # 1. News volume
    ax = axes[0]
    for g, p in panels.items():
        ax.plot(p.index, p["news_count"], label=g, color=colors.get(g), alpha=0.7, lw=0.5)
    ax.set_ylabel("Daily news count")
    ax.legend(fontsize=8)
    ax.set_title("News Volume by Group")

    # 2. Embedding dispersion
    ax = axes[1]
    for g, p in panels.items():
        if "emb_dispersion" in p.columns:
            ax.plot(p.index, p["emb_dispersion"], label=g, color=colors.get(g), alpha=0.7, lw=0.5)
    ax.set_ylabel("Embedding dispersion")
    ax.legend(fontsize=8)
    ax.set_title("Within-Day Embedding Dispersion (mean L2 from centroid)")

    # 3. PC0 vs index return
    ax = axes[2]
    common = targets.index.intersection(panels["ALL"].index)
    if len(common) > 0 and "pc_0" in panels["ALL"].columns:
        ax_twin = ax.twinx()
        ax.plot(common, panels["ALL"].loc[common, "pc_0"], "b-", alpha=0.5, lw=0.5, label="PC0")
        ax_twin.plot(common, targets.loc[common, "log_return"], "r-", alpha=0.3, lw=0.5, label="Log return")
        ax.set_ylabel("PC0 (centroid)", color="b")
        ax_twin.set_ylabel("Log return", color="r")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.set_title("Daily Market Centroid PC0 vs VN30 EW Return")

    # 4. PC0 vs 20d vol
    ax = axes[3]
    if len(common) > 0 and "pc_0" in panels["ALL"].columns:
        ax_twin = ax.twinx()
        ax.plot(common, panels["ALL"].loc[common, "pc_0"], "b-", alpha=0.5, lw=0.5, label="PC0")
        ax_twin.plot(common, targets.loc[common, "index_vol_20d"], "g-", alpha=0.3, lw=0.5, label="20d vol")
        ax.set_ylabel("PC0 (centroid)", color="b")
        ax_twin.set_ylabel("20d vol", color="g")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.set_title("Daily Market Centroid PC0 vs VN30 EW 20d Vol")

    fig.tight_layout()
    path = MARKET_DIR / "market_centroid_timeseries.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_corr_heatmap(corr_df: pd.DataFrame, out_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if corr_df.empty:
        return
    pivot = corr_df.pivot_table(index="feature", columns="target", values="pearson_r", aggfunc="first")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Market Centroid PCs vs VN30 EW Index Targets")
    fig.tight_layout()
    path = MARKET_DIR / out_name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    t_start = time.time()
    print("=" * 60)
    print("MARKET-LEVEL EDA: News Embedding Centroids vs VN30 Index")
    print("=" * 60)

    # Step 1: Load trading calendar
    print("\n[1/6] Loading trading calendar...")
    trading = _load_trading_calendar()
    print(f"  {len(trading)} trading days")

    # Step 2: Load all articles with embeddings (with cache)
    cache_file = MARKET_DIR / "all_articles_merged.parquet"
    print("\n[2/6] Loading articles + embeddings...")
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        all_articles = pd.read_parquet(cache_file)
    else:
        all_articles = load_all_articles()
        if all_articles.empty:
            print("  ERROR: No articles loaded!")
            return
        all_articles.to_parquet(cache_file, index=False)
        print(f"  Cached to {cache_file}")

    # Step 3: Map to trading dates
    print("\n[3/6] Mapping to trading days...")
    if "trading_date" not in all_articles.columns:
        all_articles = map_to_trading_days(all_articles, trading)
        all_articles.to_parquet(cache_file, index=False)
    print(f"  {len(all_articles)} articles with valid trading dates")

    # Step 4: Compute daily centroids
    print("\n[4/6] Computing daily centroids...")
    emb_cols = _embedding_cols(all_articles)
    print(f"  Embedding dims: {len(emb_cols)}")
    panels = daily_centroids(all_articles, emb_cols)
    for g, p in panels.items():
        n_days = len(p)
        print(f"  [{g}]: {n_days} trading days, {int(p['news_count'].sum())} articles")

    # Step 4b: PCA reduce centroids
    for g in panels:
        panels[g] = _reduce_pca(panels[g])
        pc_cols = [c for c in panels[g].columns if c.startswith("pc_")]
        ev = panels[g].get('pca_explained_var', pd.Series([0.0]))
        ev_val = float(ev.iloc[0]) if hasattr(ev, 'iloc') else float(ev)
        print(f"  [{g}] PCA: {len(pc_cols)} PCs, explained var {ev_val:.2%}")

    # Step 5: Load VN30 index & correlate
    print("\n[5/6] Loading VN30 EW index & correlating...")
    index = _load_vn30_index()
    print(f"  VN30 EW index: {len(index)} trading days ({index.index.min().date()} to {index.index.max().date()})")
    index_targets = ["index_return", "log_return", "index_vol_5d", "index_vol_20d"]

    all_corr_frames = []
    for gname, panel in panels.items():
        joined = panel.join(index[index_targets], how="inner")
        features = [c for c in panel.columns if c.startswith("pc_")]
        features += ["news_count", "emb_dispersion"]
        features = [f for f in features if f in joined.columns]
        corr = correlate(joined, features, index_targets)
        corr["group"] = gname
        all_corr_frames.append(corr)
        sig = corr[corr["fdr_spearman"] | corr["fdr_pearson"]]
        print(f"  [{gname}]: {len(corr)} (feat, target) pairs, {len(sig)} significant")

    all_corr = pd.concat(all_corr_frames, ignore_index=True)
    all_corr.to_csv(MARKET_DIR / "market_correlations.csv", index=False)
    print(f"  Saved market_correlations.csv")

    # Save summary
    summary = {
        "n_articles_total": int(len(all_articles)),
        "n_trading_days": int(len(trading)),
        "n_days_with_news": int(panels["ALL"]["news_count"].gt(0).sum()),
        "n_groups": len(panels),
        "n_corr_pairs": int(len(all_corr)),
        "n_significant": int(all_corr["fdr_spearman"].sum() + all_corr["fdr_pearson"].sum()),
        "embedding_dims": len(emb_cols),
        "pca_dims": 8,
    }
    with open(MARKET_DIR / "market_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved market_eda_summary.json")

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")
    plot_timeseries(panels, index)
    plot_corr_heatmap(all_corr, "market_corr_heatmap.png")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"MARKET EDA COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Output: {MARKET_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
