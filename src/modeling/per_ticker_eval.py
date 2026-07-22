"""Per-ticker evaluation + volatility regime analysis (Priority 1 items 9, 10).

Story 17-4: no_news_mask feature.
Story 18-3: per-ticker + regime evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.modeling.baseline import PRICE_FEATURES, NEWS_FEATURES, TARGETS, compute_metrics
from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
from src.modeling.features import ADV_FEATURES_DUAL


def per_ticker_evaluation(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    """ΔR² per ticker: how many tickers does news actually help?"""
    all_feats = PRICE_FEATURES + NEWS_FEATURES + ADV_FEATURES_DUAL
    df = panel.dropna(subset=[target]).copy()
    train, test = time_split(df, SPLIT_DATE)
    avail_price = [c for c in PRICE_FEATURES if c in df.columns]
    avail_all = [c for c in all_feats if c in df.columns]

    rows = []
    for ticker in sorted(df["ticker"].unique()):
        tr = train[train["ticker"] == ticker]
        te = test[test["ticker"] == ticker]
        if len(tr) < 20 or len(te) < 5:
            continue
        pipe_p = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipe_n = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipe_p.fit(tr[avail_price], tr[target])
        pipe_n.fit(tr[avail_all], tr[target])
        r2_p = float(pipe_p.score(te[avail_price], te[target]))
        r2_n = float(pipe_n.score(te[avail_all], te[target]))
        rows.append({"ticker": ticker, "price_r2": round(r2_p, 6), "news_r2": round(r2_n, 6),
                     "delta_r2": round(r2_n - r2_p, 6), "news_helps": bool(r2_n > r2_p)})
    return pd.DataFrame(rows)


def volatility_regime_evaluation(panel: pd.DataFrame, target: str,
                                  vol_col: str = "realized_vol_20d") -> pd.DataFrame:
    """News contribution broken down by volatility regime."""
    all_feats = PRICE_FEATURES + NEWS_FEATURES + ADV_FEATURES_DUAL
    df = panel.dropna(subset=[target, vol_col]).copy()
    train, test = time_split(df, SPLIT_DATE)
    avail_price = [c for c in PRICE_FEATURES if c in df.columns]
    avail_all = [c for c in all_feats if c in df.columns]

    pipe_p = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe_n = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe_p.fit(train[avail_price], train[target])
    pipe_n.fit(train[avail_all], train[target])

    test = test.copy()
    vol = test[vol_col]
    q33 = vol.quantile(0.33)
    q66 = vol.quantile(0.66)
    low_mask = vol <= q33
    mid_mask = (vol > q33) & (vol <= q66)
    high_mask = vol > q66

    rows = []
    for regime, mask in [("low", low_mask), ("normal", mid_mask), ("high", high_mask)]:
        sub = test[mask]
        if len(sub) < 10:
            continue
        r2_p = float(pipe_p.score(sub[avail_price], sub[target]))
        r2_n = float(pipe_n.score(sub[avail_all], sub[target]))
        rows.append({"regime": regime, "n": len(sub), "price_r2": round(r2_p, 6),
                     "news_r2": round(r2_n, 6), "delta_r2": round(r2_n - r2_p, 6)})
    return pd.DataFrame(rows)


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    panel = build_panel()
    if panel.empty:
        return written

    lines = ["# Per-ticker and Volatility Regime Evaluation\n"]

    for target in TARGETS:
        pt = per_ticker_evaluation(panel, target)
        if pt.empty:
            continue
        p_path = outdir / f"per_ticker_{target}.csv"
        pt.to_csv(p_path, index=False, encoding="utf-8")
        written.append(p_path)
        n_help = int(pt["news_helps"].sum())
        lines.append(f"\n## {target}: News helps {n_help}/{len(pt)} tickers")
        lines.append(f"  Mean delta_r2={pt['delta_r2'].mean():.6f}")
        lines.append(f"  Max delta_r2={pt['delta_r2'].max():.6f}")
        lines.append(f"  Min delta_r2={pt['delta_r2'].min():.6f}")
        if pt["delta_r2"].notna().any():
            best = pt.loc[pt["delta_r2"].idxmax()]
            lines.append(f"  Best ticker: {best['ticker']} ({best['delta_r2']:+.6f})")

        vr = volatility_regime_evaluation(panel, target)
        if not vr.empty:
            v_path = outdir / f"vol_regime_{target}.csv"
            vr.to_csv(v_path, index=False, encoding="utf-8")
            written.append(v_path)
            lines.append(f"\n  Volatility regime breakdown:")
            for _, r in vr.iterrows():
                lines.append(f"    {r['regime']:8s} (n={r['n']:5d}): delta_r2={r['delta_r2']:+.6f}")

    rpath = outdir / "per_ticker_regime_report.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    written.append(rpath)
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
