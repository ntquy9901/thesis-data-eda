from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs
from src.modeling.baseline import PRICE_FEATURES, TARGETS, compute_metrics, make_pipeline
from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
from src.modeling.features import ADV_FEATURES_DUAL

MINI_FEATURES = [
    "news_count_1d", "news_count_3d", "days_since_last_news",
    "kq_emb_0", "th_emb_0",
    "kq_emb_norm", "th_emb_norm",
    "kq_novelty_30d", "th_novelty_30d",
    "kq_dispersion", "th_dispersion",
    "kq_max_semantic_shock", "th_max_semantic_shock",
]


def _exante_high_vol_flag(series: pd.Series, threshold: float = 0.8) -> pd.Series:
    exp_min = series.expanding().min()
    exp_max = series.expanding().max()
    denom = (exp_max - exp_min).clip(lower=1e-12)
    return (series - exp_min) / denom >= threshold


def _align_flags(flags: pd.Series, target_df: pd.DataFrame) -> pd.Series:
    return flags.reindex(target_df.index, fill_value=False)


class ConditionalVolModel:
    def __init__(self, threshold: float = 0.8, shrink_lambda: float = 0.3,
                 clip_q: float = 0.01, vol_col: str = "realized_vol_20d",
                 gate_type: str = "hard"):
        self.threshold = threshold
        self.shrink_lambda = shrink_lambda
        self.clip_q = clip_q
        self.vol_col = vol_col
        self.gate_type = gate_type
        self.har_model: Pipeline | None = None
        self.news_model: Pipeline | None = None
        self.price_feats: list[str] = []
        self.news_feats: list[str] = []

    def fit(self, panel: pd.DataFrame, target: str) -> None:
        df = panel.dropna(subset=[target, self.vol_col]).copy()
        train, _ = time_split(df, SPLIT_DATE)
        avail_price = [c for c in PRICE_FEATURES if c in df.columns]
        avail_news = [c for c in PRICE_FEATURES + MINI_FEATURES if c in df.columns]
        self.price_feats = avail_price
        self.news_feats = avail_news
        self.har_model = make_pipeline().fit(train[avail_price], train[target])
        self.news_model = make_pipeline().fit(train[avail_news], train[target])

    def _gate(self, flags: pd.Series) -> np.ndarray:
        if self.gate_type == "soft":
            return 1.0 / (1.0 + np.exp(-5.0 * (flags.astype(float) - 0.5)))
        return flags.astype(float).to_numpy()

    def predict(self, panel: pd.DataFrame, target: str) -> np.ndarray:
        if self.har_model is None:
            raise RuntimeError("fit first")
        df = panel.dropna(subset=[target, self.vol_col]).copy()
        pred_har = self.har_model.predict(df[self.price_feats])
        pred_news = self.news_model.predict(df[self.news_feats])
        delta = pred_news - pred_har
        q = float(np.percentile(np.abs(delta), self.clip_q * 100)) if self.clip_q > 0 else np.inf
        delta = np.clip(delta, -q, q)
        delta *= self.shrink_lambda
        flags_df = pd.DataFrame(
            _exante_high_vol_flag(df.groupby("ticker")[self.vol_col].transform(lambda s: s),
                                  self.threshold)
        )
        flags_s = _align_flags(flags_df.iloc[:, 0], df)
        g = self._gate(flags_s)
        return pred_har + g * delta

    def evaluate(self, panel: pd.DataFrame, target: str) -> dict:
        df = panel.dropna(subset=[target, self.vol_col]).copy()
        self.fit(df, target)
        train, test = time_split(df, SPLIT_DATE)
        pred = self.predict(test, target)
        y_true = test[target].to_numpy()
        m = compute_metrics(y_true, pred)

        pred_har = self.har_model.predict(test[self.price_feats])
        m_har = compute_metrics(y_true, pred_har)

        flags_df = pd.DataFrame(
            _exante_high_vol_flag(df.groupby("ticker")[self.vol_col].transform(lambda s: s),
                                  self.threshold)
        )
        flags_s = _align_flags(flags_df.iloc[:, 0], test)
        high_mask = flags_s
        low_mask = ~flags_s

        m_high = compute_metrics(y_true[high_mask], pred[high_mask])
        m_low = compute_metrics(y_true[low_mask], pred[low_mask])
        har_high = compute_metrics(y_true[high_mask], pred_har[high_mask])
        har_low = compute_metrics(y_true[low_mask], pred_har[low_mask])

        return {
            "target": target, "threshold": self.threshold, "gate_type": self.gate_type,
            "shrink_lambda": self.shrink_lambda,
            "all": {"r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"]},
            "high_vol": {"r2": m_high["r2"], "rmse": m_high["rmse"], "mae": m_high["mae"],
                         "n": int(high_mask.sum())},
            "low_vol": {"r2": m_low["r2"], "rmse": m_low["rmse"], "mae": m_low["mae"],
                        "n": int(low_mask.sum())},
            "har_baseline_all": {"r2": m_har["r2"], "rmse": m_har["rmse"], "mae": m_har["mae"]},
            "har_baseline_high": {"r2": har_high["r2"], "rmse": har_high["rmse"], "mae": har_high["mae"]},
            "har_baseline_low": {"r2": har_low["r2"], "rmse": har_low["rmse"], "mae": har_low["mae"]},
        }


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    panel = build_panel()
    if panel.empty:
        return written

    results = {}
    lines = ["# Epic 20 — Conditional Lightweight Model\n"]
    lines.append(f"\nReduced feature set ({len(MINI_FEATURES)} features): {MINI_FEATURES}\n")

    for gate_type in ["hard", "soft"]:
        for target in TARGETS:
            cv = ConditionalVolModel(gate_type=gate_type)
            r = cv.evaluate(panel, target)
            results[f"{gate_type}_{target}"] = r
            lines.append(f"\n## {gate_type} gate — {target}\n")
            lines.append(f"- All: R²={r['all']['r2']}, RMSE={r['all']['rmse']}")
            lines.append(f"- High-vol (n={r['high_vol']['n']}): R²={r['high_vol']['r2']}")
            lines.append(f"- Low-vol  (n={r['low_vol']['n']}): R²={r['low_vol']['r2']}")
            har = r["har_baseline_all"]
            lines.append(f"- HAR baseline: R²={har['r2']}, RMSE={har['rmse']}")
            dr2_all = round(r["all"]["r2"] - har["r2"], 6) if r["all"]["r2"] is not None and har["r2"] is not None else None
            dr2_high = round(r["high_vol"]["r2"] - r["har_baseline_high"]["r2"], 6) if r["high_vol"]["r2"] is not None and r["har_baseline_high"]["r2"] is not None else None
            lines.append(f"- ΔR² all: {dr2_all}")
            lines.append(f"- ΔR² high-vol: {dr2_high}")

    jpath = outdir / "conditional_model_results.json"
    jpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    written.append(jpath)

    rpath = outdir / "conditional_model_report.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    written.append(rpath)
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
