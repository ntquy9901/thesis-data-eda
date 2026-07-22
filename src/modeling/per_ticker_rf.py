"""Story 23-1 — Per-ticker RF baseline evaluation on price-only features.

Trains RandomForest per ticker on 1-year train window (2025), tests on Jan 2026.
Computes full metrics: R², RMSE, MAE, QLIKE, DirAcc, MAPE, Theil's U, Pearson r, Spearman r.
Output: ``results/modeling/per_ticker_rf_baseline_{target}.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor

from config import EDA_OUTPUT_DIR, EDA_TICKERS, PROJECT_ROOT
from src.modeling.dataset import TARGETS

PRICE_FEATURES = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
TRAIN_START = "2025-01-02"
TRAIN_END = "2025-12-31"
TEST_START = "2026-01-02"
TEST_END = "2026-01-31"
EPS = 1e-12
RESULTS_DIR = PROJECT_ROOT / "results" / "modeling"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray | None = None) -> dict:
    n = len(y_true)
    if n == 0:
        return {k: None for k in ["rmse", "mae", "r2", "qlike", "dir_acc", "mape", "theils_u", "pearson_r", "spearman_r"]}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    yt = np.clip(y_true, EPS, None)
    yp = np.clip(y_pred, EPS, None)
    qlike = float(np.mean(yt / yp - np.log(yt / yp) - 1.0))
    dir_acc = None
    if y_prev is not None and n > 0:
        dir_acc = float(np.mean((y_true > y_prev) == (y_pred > y_prev)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), EPS, None)))) * 100
    denom = float(np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2)))
    theils_u = rmse / denom if denom > 0 else float("nan")
    pearson_r_val = None
    spearman_r_val = None
    if n > 1:
        try:
            pearson_r_val = float(pearsonr(y_true, y_pred).statistic)
        except Exception:
            pearson_r_val = None
        try:
            spearman_r_val = float(spearmanr(y_true, y_pred).statistic)
        except Exception:
            spearman_r_val = None
    return {
        "rmse": round(rmse, 8), "mae": round(mae, 8), "r2": round(float(r2), 6),
        "qlike": round(qlike, 6), "dir_acc": round(dir_acc, 4) if dir_acc is not None else None,
        "mape": round(mape, 6), "theils_u": round(theils_u, 6),
        "pearson_r": round(pearson_r_val, 6) if pearson_r_val is not None else None,
        "spearman_r": round(spearman_r_val, 6) if spearman_r_val is not None else None,
    }


def evaluate_per_ticker_rf(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)
    avail_feats = [c for c in PRICE_FEATURES if c in panel.columns]
    rows = []
    for ticker in sorted(EDA_TICKERS):
        tdf = panel[panel["ticker"] == ticker].dropna(subset=[target]).copy()
        tr = tdf[(tdf["date"] >= TRAIN_START) & (tdf["date"] <= TRAIN_END)]
        te = tdf[(tdf["date"] >= TEST_START) & (tdf["date"] <= TEST_END)]
        if len(tr) < 20 or len(te) < 5:
            continue
        rf.fit(tr[avail_feats], tr[target])
        y_pred = rf.predict(te[avail_feats])
        y_true = te[target].to_numpy()
        y_prev = te[target].shift(1).to_numpy()
        m = compute_metrics(y_true, y_pred, y_prev)
        rows.append({"ticker": ticker, "n_train": len(tr), "n_test": len(te), **m})
    return pd.DataFrame(rows)


def run(targets: list[str] | None = None) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = EDA_OUTPUT_DIR / "modeling" / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    targets = targets or TARGETS
    written: list[Path] = []
    for target in targets:
        if target not in panel.columns:
            continue
        df = evaluate_per_ticker_rf(panel, target)
        if df.empty:
            continue
        path = RESULTS_DIR / f"per_ticker_rf_baseline_{target}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        written.append(path)
        metric_cols = ["rmse", "mae", "r2", "qlike", "dir_acc", "mape", "theils_u", "pearson_r", "spearman_r"]
        avail = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
        print(f"\n=== {target} summary (30 tickers) ===")
        for col in avail:
            vals = df[col].dropna()
            print(f"  {col:12s} mean={vals.mean():.6f}  std={vals.std():.6f}  min={vals.min():.6f}  "
                  f"p50={vals.median():.6f}  max={vals.max():.6f}")
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
