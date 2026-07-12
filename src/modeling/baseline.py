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

PRICE_FEATURES = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
NEWS_FEATURES = ["news_count_1d", "news_count_3d", "news_count_5d", "days_since_last_news", "sentiment_mean"]
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


def make_pipeline():
    """Preprocessing (fit on TRAIN only) + Ridge regressor."""
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
    """Train Model A (price) and Model B (price+news) per target → metrics rows."""
    panel = build_panel()
    if panel.empty:
        return pd.DataFrame()
    rows = []
    for target in TARGETS:
        if target not in panel.columns:
            continue
        # Model A: price only
        Xtr, ytr, Xte, yte, feats, yprev = _split_xy(panel, PRICE_FEATURES, target, SPLIT_DATE)
        if len(Xtr) == 0 or len(Xte) == 0:
            continue
        pa = make_pipeline().fit(Xtr, ytr)
        ma = compute_metrics(yte.to_numpy(), pa.predict(Xte), yprev)
        rows.append({"target": target, "model": "A_price_only", "n_features": len(feats), **ma})
        # Model B: price + news (guard like Model A — empty split would crash fit)
        Xtr2, ytr2, Xte2, yte2, feats2, yprev2 = _split_xy(panel, PRICE_FEATURES + NEWS_FEATURES, target, SPLIT_DATE)
        if len(Xtr2) == 0 or len(Xte2) == 0:
            continue
        pb = make_pipeline().fit(Xtr2, ytr2)
        mb = compute_metrics(yte2.to_numpy(), pb.predict(Xte2), yprev2)
        rows.append({"target": target, "model": "B_price_plus_news", "n_features": len(feats2), **mb})
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

    # comparison: pivot per target, compute Δ (B vs A)
    lines = ["# Modeling Comparison — News Contribution to Parkinson Vol\n"]
    lines.append(f"Split: train < {SPLIT_DATE}, test >= {SPLIT_DATE}. Model: Ridge (HAR-style, median impute + standardize fit on train).\n")
    lines.append("\n## Metrics by target × model\n")
    lines.append("```\n" + metrics.to_string(index=False) + "\n```")

    lines.append("\n\n## News contribution (Model B − Model A; positive ΔR² / negative ΔRMSE = news helps)\n")
    for target in TARGETS:
        a = metrics[(metrics.target == target) & (metrics.model == "A_price_only")]
        b = metrics[(metrics.target == target) & (metrics.model == "B_price_plus_news")]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        d_rmse = (b["rmse"] - a["rmse"]) if a["rmse"] is not None else None
        d_r2 = (b["r2"] - a["r2"]) if a["r2"] is not None else None
        verdict = "news HELPS" if (d_r2 is not None and d_r2 > 0) else ("neutral" if d_r2 == 0 else "news does NOT help")
        lines.append(f"- **{target}**: ΔRMSE={d_rmse:+.6f}  ΔR²={d_r2:+.4f}  → {verdict}")

    rep = outdir / "comparison_report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    written.append(rep)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
