"""Story 7-2 — Baseline models + news-contribution comparison.

Trains two HAR-style regressors per Parkinson target (pk_t+1/+5/+10) on the
modeling panel, with a strict time-based split (no shuffle, preprocessing fit
on TRAIN only):

- **Model A (price-only):** HAR features (har_daily/weekly/monthly) + ATR + realized vol
- **Model B (price + news):** Model A features + news_count_1d/3d/5d + days_since_last_news + sentiment_mean

Metrics (aligned with the sibling baselines): RMSE, MAE, R², QLIKE, directional
accuracy. The comparison (ΔRMSE, ΔR²) quantifies the **news contribution**.

Outputs: ``eda_output/modeling/metrics.csv`` + ``comparison_report.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.common import EDA_OUTPUT_DIR
from src.modeling.dataset import SPLIT_DATE, TARGETS, build_panel
from src.modeling.features import (
    ADV_FEATURES as NEWS_ADVANCED,  # Story 11-1: embedding + topic features
)

PRICE_FEATURES = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
NEWS_FEATURES = ["news_count_1d", "news_count_3d", "news_count_5d", "days_since_last_news", "sentiment_mean"]
FEATURE_SETS = {
    "price": PRICE_FEATURES,
    "price+news_basic": PRICE_FEATURES + NEWS_FEATURES,
    "price+news_adv": PRICE_FEATURES + NEWS_FEATURES + NEWS_ADVANCED,
}
MODEL_TYPES = ["ridge", "gbm"]
EPS = 1e-12


# ---------- pure helpers (unit-tested) ----------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray | None = None) -> dict:
    """RMSE, MAE, R², QLIKE, directional accuracy (vs ``y_prev`` if given).

    RMSE/MAE/R² use the raw prediction (Ridge can predict ≤0; flooring it would
    distort error metrics). QLIKE clips ``y_pred`` to EPS (it divides + logs).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_raw = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"rmse": None, "mae": None, "r2": None, "qlike": None, "dir_acc": None}
    rmse = float(np.sqrt(np.mean((y_true - y_pred_raw) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred_raw)))
    ss_res = float(np.sum((y_true - y_pred_raw) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # QLIKE guards y_true==0 (Parkinson vol can be 0 when high==low) and y_pred<=0
    yt = np.clip(y_true, EPS, None)
    yp = np.clip(y_pred_raw, EPS, None)
    ratio = yt / yp
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    dir_acc = None
    if y_prev is not None and n > 0:
        actual_up = y_true > y_prev
        pred_up = y_pred_raw > y_prev
        dir_acc = float(np.mean(actual_up == pred_up))
    return {"rmse": round(rmse, 8), "mae": round(mae, 8), "r2": round(float(r2), 6),
            "qlike": round(qlike, 6), "dir_acc": round(dir_acc, 4) if dir_acc is not None else None}


def make_pipeline(model_type: str = "ridge"):
    """Preprocessing (fit on TRAIN only) + regressor.

    ``ridge`` → median-impute + standardize + Ridge. ``gbm`` → HistGradientBoosting
    (NaN-native, no impute/scale needed; fast histogram GBM).
    """
    if model_type == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=0)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def _split_xy(panel: pd.DataFrame, features: list[str], target: str, split: str):
    """Time-split + assemble X/y. Reuses ``dataset.time_split`` (single source).

    ``y_prev`` for directional accuracy is computed per-ticker (groupby shift) so
    it never bleeds across ticker boundaries in the concatenated panel.
    """
    from src.modeling.dataset import time_split

    df = panel.dropna(subset=[target]).copy()
    train, test = time_split(df, split)
    feats = [c for c in features if c in df.columns]
    y_prev = test.groupby("ticker")[target].shift(1).to_numpy() if not test.empty else None
    return train[feats], train[target], test[feats], test[target], feats, y_prev


# ---------- runner ----------
def run_models() -> pd.DataFrame:
    """Train each (model × feature_set × target) → metrics rows (18 combos)."""
    panel = build_panel()
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for target in TARGETS:
        if target not in panel.columns:
            continue
        for fset_name, feats in FEATURE_SETS.items():
            Xtr, ytr, Xte, yte, used, yprev = _split_xy(panel, feats, target, SPLIT_DATE)
            if len(Xtr) == 0 or len(Xte) == 0:
                continue
            for mtype in MODEL_TYPES:
                pipe = make_pipeline(mtype).fit(Xtr, ytr)
                m = compute_metrics(yte.to_numpy(), pipe.predict(Xte), yprev)
                rows.append({"target": target, "model": mtype, "feature_set": fset_name, "n_features": len(used), **m})
    return pd.DataFrame(rows)


def run() -> list[Path]:
    from src.eda.common import ensure_output_dirs

    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = run_models()
    if metrics.empty:
        return []
    written: list[Path] = []

    mpath = outdir / "metrics.csv"
    metrics.to_csv(mpath, index=False, encoding="utf-8")
    written.append(mpath)

    lines = ["# Modeling Comparison — News Contribution to Parkinson Vol\n"]
    lines.append(
        f"Split: train < {SPLIT_DATE}, test >= {SPLIT_DATE}. Models: ridge (linear HAR) + gbm "
        f"(HistGradientBoosting). Feature sets: price / +news_basic / +news_adv. 30 tickers."
    )
    lines.append("\n## Metrics (model × feature_set × target)\n")
    lines.append("```\n" + metrics.to_string(index=False) + "\n```")

    lines.append("\n\n## News contribution (ΔR² vs price-only; >0 = news helps)\n")
    for mtype in MODEL_TYPES:
        lines.append(f"\n### {mtype}")
        for target in TARGETS:
            base = metrics[(metrics.model == mtype) & (metrics.feature_set == "price") & (metrics.target == target)]
            if base.empty:
                continue
            base = base.iloc[0]
            for fset in ["price+news_basic", "price+news_adv"]:
                row = metrics[(metrics.model == mtype) & (metrics.feature_set == fset) & (metrics.target == target)]
                if row.empty:
                    continue
                row = row.iloc[0]
                d_r2 = (row["r2"] - base["r2"]) if base["r2"] is not None and row["r2"] is not None else None
                d_rmse = (row["rmse"] - base["rmse"]) if base["rmse"] is not None else None
                verdict = "HELPS" if (d_r2 is not None and d_r2 > 0) else ("neutral" if d_r2 == 0 else "no effect")
                lines.append(f"- {target} [{fset}]: ΔR²={d_r2:+.4f}  ΔRMSE={d_rmse:+.6f} → {verdict}")

    rep = outdir / "comparison_report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    written.append(rep)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
