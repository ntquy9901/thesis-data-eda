"""
Advanced Market-Level Analysis: SOTA-inspired methods.

Implements:
  1. Divergence Index — cosine distance between khach_quan and tong_hop centroids
  2. NVIX-style Index — volume-weighted centroid norm
  3. Lead-Lag multi-day cross-correlation
  4. Forward volatility prediction using centroids
  5. Temporal decay EWMA centroids
  6. Uncertainty index analysis

Outputs -> eda_output/market_eda/advanced/
"""

import json, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore")

from src.data.discover_news import discover_source_files, load_source
from src.features.news_embeddings import _article_cache_path
from config import EDA_OUTPUT_DIR

OUTDIR = (EDA_OUTPUT_DIR or PROJECT_ROOT / "eda_output") / "market_eda" / "advanced"
OUTDIR.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "khach_quan": {"cafef","hsc","vnexpress","thanhnien","tuoitre","nld","vietnamplus",
                   "thanhnien_root","thanhnien_objective",
                   "tuoitre_root","tuoitre_objective",
                   "vietnamplus_root","vietnamplus_objective"},
    "tong_hop":   {"ssi","vndirect","vnstock","vietstock","vsdc"},
}


def load_vn30_index() -> pd.DataFrame:
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
    all_prices = pd.concat(frames, ignore_index=True)
    all_prices["date"] = pd.to_datetime(all_prices["date"]).dt.normalize()
    ew = all_prices.groupby("date").agg(
        index_price=("close", "mean"), tot_volume=("volume", "sum")
    ).reset_index()
    ew["log_return"] = np.log(ew["index_price"] / ew["index_price"].shift(1))
    for d in [5, 10, 20]:
        ew[f"hist_vol_{d}d"] = ew["log_return"].rolling(d).std()
    # forward vol (needs careful shift to avoid lookahead in predictor context)
    for d in [5, 10, 20]:
        ew[f"fwd_vol_{d}d"] = ew["log_return"].shift(-d).rolling(d).std().shift(d - 1)
    return ew.set_index("date").dropna(subset=["log_return"])


def load_embedding_panels() -> dict[str, pd.DataFrame]:
    """Reuse cached merged articles, compute daily centroids per group."""
    cache_file = Path("C:/luanvan/data_eda/eda_output/market_eda/all_articles_merged.parquet")
    if not cache_file.exists():
        raise FileNotFoundError("Run market_eda_main.py first to generate cached articles")
    df = pd.read_parquet(cache_file)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()

    emb_cols = [c for c in df.columns if re.match(r"^raw_\d+$", c)]
    groups = {"ALL": df}
    for gname, srcs in GROUPS.items():
        mask = df["source"].isin(srcs)
        if mask.any():
            groups[gname] = df[mask].copy()

    panels = {}
    for gname, gdf in groups.items():
        daily = gdf.groupby("trading_date")
        centroid = daily[emb_cols].mean()
        counts = daily.size().to_frame("news_count")
        disp = []
        for dt, sub in daily:
            c = centroid.loc[dt].values
            mat = sub[emb_cols].values
            dists = np.linalg.norm(mat - c, axis=1)
            disp.append(dists.mean())
        counts["emb_dispersion"] = disp
        panel = counts.join(centroid, how="left")
        panel.index.name = "date"
        panels[gname] = panel
    return panels


def _reduce_to_pc(panel: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Reduce raw_* columns to n PCs using PCA fit on pre-2020 data."""
    from sklearn.decomposition import PCA
    emb = [c for c in panel.columns if re.match(r"^raw_\d+$", c)]
    if len(emb) < 2:
        return panel
    mat = panel[emb].values
    train = panel.index < np.datetime64("2020-01-01")
    n_train = int(train.sum())
    if n_train < 2:
        return panel
    d = min(n, mat.shape[1], max(1, n_train - 1))
    pca = PCA(n_components=d).fit(mat[train])
    reduced = pca.transform(mat).astype(np.float32)
    for i in range(d):
        panel[f"pc_{i}"] = reduced[:, i]
    panel["pca_ev"] = pca.explained_variance_ratio_.sum()
    return panel  # keep raw cols too


def compute_divergence_index(panels: dict) -> pd.Series | pd.DataFrame:
    """Cosine distance between khach_quan and tong_hop centroids on shared-trading days."""
    if "khach_quan" not in panels or "tong_hop" not in panels:
        return pd.Series(dtype=float)
    kq = panels["khach_quan"]
    th = panels["tong_hop"]
    common = kq.index.intersection(th.index)
    if len(common) < 2:
        return pd.Series(dtype=float)
    emb = [c for c in kq.columns if re.match(r"^raw_\d+$", c)]
    kq_v = kq.loc[common, emb].values
    th_v = th.loc[common, emb].values
    norms = np.clip(np.linalg.norm(kq_v, axis=1) * np.linalg.norm(th_v, axis=1), 1e-9, None)
    cos_sim = (kq_v * th_v).sum(axis=1) / norms
    return pd.Series(1.0 - cos_sim, index=common, name="divergence")


def compute_nvix_index(panels: dict) -> pd.DataFrame:
    """NVIX-style: daily centroid L2 norm weighted by sqrt(news_count)."""
    panel = panels["ALL"]
    emb = [c for c in panel.columns if re.match(r"^raw_\d+$", c)]
    centroid_norm = np.linalg.norm(panel[emb].values, axis=1)
    nvix = pd.DataFrame({
        "centroid_norm": centroid_norm,
        "nvix_raw": centroid_norm * np.sqrt(panel["news_count"].values + 1),
    }, index=panel.index)
    # Z-score for interpretability
    train = nvix.index < np.datetime64("2020-01-01")
    if train.any():
        mu = nvix.loc[train, "nvix_raw"].mean()
        sd = nvix.loc[train, "nvix_raw"].std()
        nvix["nvix_z"] = (nvix["nvix_raw"] - mu) / sd
    return nvix


def compute_lead_lag(
    panels: dict, index: pd.DataFrame, max_lag: int = 20
) -> pd.DataFrame:
    """Cross-correlation at multiple lags: centroid PC0 vs log_return."""
    from scipy.stats import pearsonr
    rows = []
    for gname, panel in panels.items():
        if "pc_0" not in panel.columns:
            continue
        joined = panel[["pc_0"]].join(index[["log_return", "hist_vol_20d"]], how="inner")
        pc = joined["pc_0"].values
        ret = joined["log_return"].values
        vol = joined["hist_vol_20d"].values
        n = len(pc)
        for lag in range(-max_lag, max_lag + 1):
            if n - abs(lag) < 10:
                continue
            if lag > 0:
                a_pc, b_ret = pc[:-lag], ret[lag:]
                a_vl, b_vl = pc[:-lag], vol[lag:]
            elif lag < 0:
                a_pc, b_ret = pc[-lag:], ret[:lag]
                a_vl, b_vl = pc[-lag:], vol[:lag]
            else:
                a_pc, b_ret, a_vl, b_vl = pc, ret, pc, vol
            r_p, p_p = pearsonr(a_pc, b_ret) if np.std(a_pc) > 0 and np.std(b_ret) > 0 else (np.nan, 1.0)
            rows.append({"group": gname, "lag_days": lag, "target": "log_return",
                         "pearson_r": r_p, "pearson_p": p_p})
            r_v, p_v = pearsonr(a_vl, b_vl) if np.std(a_vl) > 0 and np.std(b_vl) > 0 else (np.nan, 1.0)
            rows.append({"group": gname, "lag_days": lag, "target": "hist_vol_20d",
                         "pearson_r": r_v, "pearson_p": p_v})
    return pd.DataFrame(rows)


def forward_vol_prediction(
    panels: dict, index: pd.DataFrame
) -> pd.DataFrame:
    """Predict forward volatility using centroids (OLS for each group + targets)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr
    rows = []
    for gname, panel in panels.items():
        pc = [c for c in panel.columns if c.startswith("pc_")]
        if len(pc) < 1:
            continue
        for tgt in ["fwd_vol_5d", "fwd_vol_10d", "fwd_vol_20d"]:
            if tgt not in index.columns:
                continue
            joined = panel[pc].join(index[[tgt]], how="inner").dropna()
            if len(joined) < 30:
                continue
            train = joined.index < np.datetime64("2020-01-01")
            if train.sum() < 10:
                continue
            X_train = joined.loc[train, pc].values
            y_train = joined.loc[train, tgt].values
            X_test = joined.loc[~train, pc].values
            y_test = joined.loc[~train, tgt].values

            m = LinearRegression().fit(X_train, y_train)
            y_pred = m.predict(X_test)
            r, p = pearsonr(y_pred.flatten(), y_test) if np.std(y_pred) > 0 and np.std(y_test) > 0 else (np.nan, 1.0)
            rows.append({
                "group": gname, "target": tgt,
                "n_train": len(X_train), "n_test": len(X_test),
                "r2_test": r2_score(y_test, y_pred),
                "pearson_r": r, "pearson_p": p,
            })
    return pd.DataFrame(rows)


def temporal_decay_corr(
    panels: dict, index: pd.DataFrame, half_lives: list = [3, 7, 14, 30]
) -> pd.DataFrame:
    """EWMA-weighted centroid vs log_return at multiple decay rates."""
    from scipy.stats import pearsonr
    rows = []
    for gname, panel in panels.items():
        raw = [c for c in panel.columns if re.match(r"^raw_\d+$", c)]
        if len(raw) < 1:
            continue
        for hl in half_lives:
            alpha = 1 - 0.5 ** (1 / hl)
            ewma = panel[raw].ewm(alpha=alpha, adjust=False).mean()
            if ewma.empty:
                continue
            ewma_norm = np.linalg.norm(ewma.values, axis=1)
            s = pd.Series(ewma_norm, index=panel.index, name=f"ewma_norm_hl{hl}")
            joined = s.to_frame().join(index[["log_return", "hist_vol_20d"]], how="inner").dropna()
            for tgt in ["log_return", "hist_vol_20d"]:
                r, p = pearsonr(joined[s.name], joined[tgt]) if np.std(joined[s.name]) > 0 and np.std(joined[tgt]) > 0 else (np.nan, 1.0)
                rows.append({
                    "group": gname, "target": tgt, "half_life": hl,
                    "pearson_r": r, "pearson_p": p,
                })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("=" * 60)
    print("ADVANCED MARKET ANALYSIS: SOTA Methods")
    print("=" * 60)

    # Load data
    print("\n[1] Loading panel data...")
    panels = load_embedding_panels()
    for g in panels:
        panels[g] = _reduce_to_pc(panels[g])
        n_pc = sum(1 for c in panels[g].columns if c.startswith("pc_"))
        print(f"  {g}: {len(panels[g])} days, {n_pc} PCs")

    index = load_vn30_index()
    print(f"  VN30 EW index: {len(index)} days")

    # 1. Divergence Index
    print("\n[2] Divergence Index...")
    divergence = compute_divergence_index(panels)
    if len(divergence) > 0:
        diverg_df = divergence.to_frame("divergence")
        diverg_df.to_csv(OUTDIR / "divergence_index.csv")
        joined = diverg_df.join(index[["log_return", "hist_vol_20d"]], how="inner").dropna()
        from scipy.stats import pearsonr
        for tgt in ["log_return", "hist_vol_20d"]:
            r, p = (pearsonr(joined["divergence"], joined[tgt])
                    if np.std(joined["divergence"]) > 0 and np.std(joined[tgt]) > 0 else (np.nan, 1.0))
            print(f"  Divergence vs {tgt:20s}: r={r:+.4f} p={p:.4f}")

    # 2. NVIX-style
    print("\n[3] NVIX-style Index...")
    nvix = compute_nvix_index(panels)
    nvix.to_csv(OUTDIR / "nvix_index.csv")
    joined = nvix.join(index[["log_return", "hist_vol_20d", "fwd_vol_10d"]], how="inner").dropna()
    from scipy.stats import pearsonr
    for col in ["centroid_norm", "nvix_raw", "nvix_z"]:
        for tgt in ["log_return", "hist_vol_20d", "fwd_vol_10d"]:
            r, p = (
                pearsonr(joined[col], joined[tgt])
                if np.std(joined[col]) > 0 and np.std(joined[tgt]) > 0
                else (np.nan, 1.0)
            )
            print(f"  {col:15s} vs {tgt:15s}: r={r:+.4f} p={p:.4f}")

    # 3. Lead-Lag
    print("\n[4] Lead-Lag Cross-Correlation...")
    leadlag = compute_lead_lag(panels, index, max_lag=10)
    leadlag.to_csv(OUTDIR / "leadlag_correlation.csv", index=False)
    for g in ["ALL", "khach_quan", "tong_hop"]:
        sub = leadlag[(leadlag["group"] == g) & (leadlag["target"] == "log_return")]
        if not sub.empty:
            best = sub.loc[sub["pearson_r"].abs().idxmax()]
            print(f"  {g:12s} best lag: {best['lag_days']:+.0f}d r={best['pearson_r']:+.4f}")

    # 4. Forward volatility prediction
    print("\n[5] Forward Volatility Prediction (OLS on PCs)...")
    fwd = forward_vol_prediction(panels, index)
    fwd.to_csv(OUTDIR / "forward_vol_prediction.csv", index=False)
    if not fwd.empty:
        best = fwd.loc[fwd["pearson_r"].abs().idxmax()]
        print(f"  Best: {best['group']} {best['target']}: r={best['pearson_r']:.4f} R2={best['r2_test']:.4f}")
        for g in ["ALL", "khach_quan", "tong_hop"]:
            sub = fwd[fwd["group"] == g]
            for _, r in sub.iterrows():
                print(f"  {g:12s} {r['target']:15s}: R2={r['r2_test']:.4f} r={r['pearson_r']:+.4f} p={r['pearson_p']:.4f}")

    # 5. Temporal decay
    print("\n[6] Temporal Decay EWMA...")
    decay = temporal_decay_corr(panels, index, half_lives=[3, 7, 14, 30])
    decay.to_csv(OUTDIR / "temporal_decay_correlation.csv", index=False)
    for g in ["ALL", "khach_quan", "tong_hop"]:
        sub = decay[decay["group"] == g]
        if not sub.empty:
            best = sub.loc[sub["pearson_r"].abs().idxmax()]
            print(f"  {g:12s} best: hl={best['half_life']}d {best['target']:15s} r={best['pearson_r']:+.4f}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ADVANCED ANALYSIS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Output: {OUTDIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
