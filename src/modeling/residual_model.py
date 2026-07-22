from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.modeling.baseline import PRICE_FEATURES, NEWS_FEATURES, TARGETS, compute_metrics
from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
from src.modeling.features import ADV_FEATURES_DUAL

MIN_RESIDUAL_SAMPLES = 100

NewsFeatSpec = list[str]


def generate_oof_residuals(panel: pd.DataFrame, target: str, n_splits: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    df = panel.dropna(subset=[target]).copy().sort_values("date")
    avail = [c for c in PRICE_FEATURES if c in df.columns]
    if len(df) < n_splits + 1 or not avail:
        df["oof_pred_price"] = np.nan
        df["oof_residual"] = np.nan
        return df, df["oof_residual"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    df["oof_pred_price"] = np.nan
    for train_idx, val_idx in tscv.split(df):
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe.fit(train[avail], train[target])
        df.iloc[val_idx, df.columns.get_loc("oof_pred_price")] = pipe.predict(val[avail])
    df["oof_residual"] = df[target] - df["oof_pred_price"]
    return df, df["oof_residual"]


def _safe_delta(current: dict | None, baseline: dict, key: str = "r2"):
    if current is None or current.get(key) is None or baseline.get(key) is None:
        return None
    return round(current[key] - baseline[key], 6)


def evaluate_residual_model(panel: pd.DataFrame, target: str, news_feats: NewsFeatSpec | None = None) -> dict:
    if news_feats is None:
        news_feats = NEWS_FEATURES + ADV_FEATURES_DUAL

    df, residuals = generate_oof_residuals(panel, target)
    avail_price = [c for c in PRICE_FEATURES if c in df.columns]
    avail_news = [c for c in news_feats if c in df.columns]
    avail_all = list(dict.fromkeys(avail_price + avail_news))

    train, test = time_split(df, SPLIT_DATE)

    pipe_price = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_price.fit(train[avail_price], train[target])
    pred_price = pipe_price.predict(test[avail_price])
    m_price = compute_metrics(test[target].to_numpy(), pred_price)

    pipe_direct = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_direct.fit(train[avail_all], train[target])
    pred_direct = pipe_direct.predict(test[avail_all])
    m_direct = compute_metrics(test[target].to_numpy(), pred_direct)

    train_resid = train.dropna(subset=["oof_residual"]).copy()
    if len(train_resid) >= MIN_RESIDUAL_SAMPLES and avail_news:
        pipe_resid = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe_resid.fit(train_resid[avail_news], train_resid["oof_residual"])
        pred_resid = pipe_resid.predict(test[avail_news])
        pred_combined = pipe_price.predict(test[avail_price]) + pred_resid
        m_resid = compute_metrics(test[target].to_numpy(), pred_combined)
    else:
        m_resid = {"r2": None, "rmse": None, "mae": None}

    return {
        "target": target,
        "price_only": {"r2": m_price["r2"], "rmse": m_price["rmse"], "mae": m_price["mae"]},
        "price_news_direct": {"r2": m_direct["r2"], "rmse": m_direct["rmse"], "mae": m_direct["mae"]},
        "residual_news_model": {"r2": m_resid["r2"], "rmse": m_resid["rmse"], "mae": m_resid["mae"]},
        "delta_r2_resid_vs_price": _safe_delta(m_resid, m_price),
        "delta_r2_direct_vs_price": _safe_delta(m_direct, m_price),
        "n_train": len(train),
        "n_test": len(test),
    }


def _format_metric(val, fmt: str = ".6f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    panel = build_panel()
    if panel.empty:
        return written

    results = {}
    lines = ["# HAR Residual News Model\n"]
    for target in TARGETS:
        res = evaluate_residual_model(panel, target)
        results[target] = res
        lines.append(f"\n## {target}\n")
        for model_key in ["price_only", "price_news_direct", "residual_news_model"]:
            m = res[model_key]
            r2_str = _format_metric(m["r2"])
            rmse_str = _format_metric(m["rmse"], ".8f")
            lines.append(f"- {model_key:25s}: R2={r2_str}, RMSE={rmse_str}")
        d_resid = _format_metric(res["delta_r2_resid_vs_price"])
        d_dir = _format_metric(res["delta_r2_direct_vs_price"])
        lines.append(f"  dR2 residual vs price  : {d_resid}")
        lines.append(f"  dR2 direct vs price    : {d_dir}")

    jpath = outdir / "residual_analysis.json"
    jpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    written.append(jpath)

    rpath = outdir / "residual_analysis_report.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    written.append(rpath)
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
