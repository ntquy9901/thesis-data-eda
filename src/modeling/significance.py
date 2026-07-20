"""Epic 9 — Statistical significance of the news contribution.

Turns Epic-8's "negligible ΔR²" into a rigorous claim:
- **Diebold-Mariano** test on the squared-error loss differential (price vs +news).
- **Bootstrap CI** on ΔRMSE and ΔR² (1000 resamples).
- **Per-ticker heterogeneity** — does news help ANY individual ticker?
- **Event abnormal-vol t-test** — is the Phase-6 abnormal vol ≠ 0?

Outputs: ``eda_output/modeling/significance_report.md`` + ``significance.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda.common import EDA_OUTPUT_DIR
from src.modeling.baseline import (
    FEATURE_SETS,
    SPLIT_DATE,
    TARGETS,
    _split_xy,
    compute_metrics,
    make_pipeline,
)
from src.modeling.dataset import build_panel


# ---------- pure helpers (unit-tested) ----------
def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> dict:
    """Diebold-Mariano test on squared-error loss differential (e1 vs e2).

    Positive DM → model 1 (e1) has larger loss (worse). Two-sided p from normal.
    """
    from scipy.stats import norm

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = e1**2 - e2**2
    n = len(d)
    if n < 2:
        return {"dm_stat": None, "dm_pvalue": None, "n": int(n)}
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return {"dm_stat": 0.0, "dm_pvalue": 1.0, "n": int(n)}
    dm = float(d.mean() / (sd / np.sqrt(n)))
    p = float(2 * (1 - norm.cdf(abs(dm))))
    return {"dm_stat": round(dm, 4), "dm_pvalue": round(p, 4), "n": int(n)}


def bootstrap_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI for ΔRMSE (=RMSE_b − RMSE_a) and ΔR² (=R²_b − R²_a)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 10:
        return {"delta_rmse_ci": None, "delta_r2_ci": None, "n_boot": 0}
    drm, dr2 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mb = compute_metrics(y[idx], np.asarray(pred_b)[idx])
        ma = compute_metrics(y[idx], np.asarray(pred_a)[idx])
        drm.append(mb["rmse"] - ma["rmse"])
        dr2.append(mb["r2"] - ma["r2"])
    return {
        "delta_rmse_ci": [round(float(np.percentile(drm, 2.5)), 8), round(float(np.percentile(drm, 97.5)), 8)],
        "delta_r2_ci": [round(float(np.percentile(dr2, 2.5)), 6), round(float(np.percentile(dr2, 97.5)), 6)],
        "n_boot": n_boot,
    }


def _fit_pair(panel: pd.DataFrame, target: str, model_type: str = "ridge", compare_fset: str = "price+news_adv"):
    """Fit price vs ``compare_fset`` models; return test y + both predictions."""
    out = {}
    for fset in ["price", compare_fset]:
        Xtr, ytr, Xte, yte, _, _ = _split_xy(panel, FEATURE_SETS[fset], target, SPLIT_DATE)
        if len(Xtr) == 0 or len(Xte) == 0:
            return None
        pipe = make_pipeline(model_type).fit(Xtr, ytr)
        out[fset] = {"y": yte.to_numpy(), "pred": pipe.predict(Xte)}
    return out


def per_ticker_delta_r2(panel: pd.DataFrame, target: str, model_type: str = "ridge",
                         compare_fset: str = "price+news_adv") -> pd.Series:
    """ΔR² (``compare_fset`` − price) per ticker; NaN where a ticker can't be fit."""
    deltas = {}
    for ticker, sub in panel.groupby("ticker"):
        sub = sub.dropna(subset=[target])
        if len(sub) < 100:
            continue
        try:
            res = _fit_pair(sub, target, model_type, compare_fset)
            if res is None:
                continue
            ra = compute_metrics(res["price"]["y"], res["price"]["pred"])["r2"]
            rb = compute_metrics(res[compare_fset]["y"], res[compare_fset]["pred"])["r2"]
            deltas[ticker] = round(rb - ra, 4)
        except Exception:
            continue
    return pd.Series(deltas)


def event_abnormal_ttest(event_csv: Path) -> dict:
    """One-sample t-test per horizon: is mean abnormal_vol ≠ 0?"""
    from scipy.stats import ttest_1samp

    if not event_csv.exists():
        return {}
    df = pd.read_csv(event_csv)
    out = {}
    for h, grp in df.groupby("horizon"):
        a = grp["abnormal_vol"].dropna()
        if len(a) < 5:
            continue
        t, p = ttest_1samp(a, 0.0)
        out[int(h)] = {"mean": round(float(a.mean()), 6), "t": round(float(t), 3),
                       "pvalue": round(float(p), 4), "significant": bool(p < 0.05)}
    return out


def _ablation_block(panel: pd.DataFrame, fsets: list[str], model_type: str,
                     lines: list[str], section_title: str) -> dict:
    """DM test + bootstrap CI (price vs each of ``fsets``) for one model type; appends a
    markdown section to ``lines`` in place and returns the same structure for the JSON report."""
    out: dict = {}
    lines.append(f"\n## {section_title}\n")
    for fset in fsets:
        out[fset] = {}
        lines.append(f"\n### {fset}")
        for target in TARGETS:
            if target not in panel.columns:
                continue
            res = _fit_pair(panel, target, model_type, compare_fset=fset)
            if res is None:
                continue
            dm = diebold_mariano(res["price"]["y"] - res["price"]["pred"],
                                 res[fset]["y"] - res[fset]["pred"])
            boot = bootstrap_delta(res[fset]["y"], res["price"]["pred"], res[fset]["pred"])
            out[fset][target] = {"dm": dm, "bootstrap": boot}
            sig = "NOT significant" if dm["dm_pvalue"] is None or dm["dm_pvalue"] > 0.05 else "significant"
            ci = boot["delta_r2_ci"]
            lines.append(f"- **{target}**: DM p={dm['dm_pvalue']} → {sig}; "
                         f"ΔR² 95% CI [{ci[0] if ci else None}, {ci[1] if ci else None}]")
    return out


# ---------- runner ----------
def run() -> list[Path]:
    panel = build_panel()
    if panel.empty:
        return []
    results = {"per_target": {}, "per_ticker_delta_r2": {}, "event_ttest": {}}
    lines = ["# Statistical Significance — News Contribution\n"]

    lines.append("\n## Diebold-Mariano + bootstrap (price vs price+news_adv, Ridge)\n")
    for target in TARGETS:
        if target not in panel.columns:
            continue
        res = _fit_pair(panel, target, "ridge")
        if res is None:
            continue
        dm = diebold_mariano(res["price"]["y"] - res["price"]["pred"],
                             res["price+news_adv"]["y"] - res["price+news_adv"]["pred"])
        boot = bootstrap_delta(res["price+news_adv"]["y"], res["price"]["pred"], res["price+news_adv"]["pred"])
        results["per_target"][target] = {"dm": dm, "bootstrap": boot}
        sig = "NOT significant" if dm["dm_pvalue"] > 0.05 else "significant"
        lines.append(f"- **{target}**: DM p={dm['dm_pvalue']} → {sig}; "
                     f"ΔR² 95% CI [{boot['delta_r2_ci'][0]}, {boot['delta_r2_ci'][1]}]")

    lines.append("\n## Per-ticker heterogeneity (ΔR² = +news_adv − price; >0 = news helps)\n")
    for target in TARGETS:
        if target not in panel.columns:
            continue
        deltas = per_ticker_delta_r2(panel, target, "ridge")
        if deltas.empty:
            continue
        n_help = int((deltas > 0).sum())
        results["per_ticker_delta_r2"][target] = deltas.to_dict()
        lines.append(f"- **{target}**: news helps in {n_help}/{len(deltas)} tickers; "
                     f"ΔR² median={deltas.median():.4f}, max={deltas.max():.4f}")

    evt = event_abnormal_ttest(EDA_OUTPUT_DIR / "relationship" / "event_study.csv")
    results["event_ttest"] = evt
    lines.append("\n## Event abnormal-volatility t-test (mean ≠ 0?)\n")
    if evt:
        for h, v in sorted(evt.items()):
            lines.append(f"- horizon {h}: mean abnormal vol={v['mean']}, p={v['pvalue']} → "
                         f"{'significant' if v['significant'] else 'not significant'}")
    else:
        lines.append("- (no event_study.csv)")

    # Story 14-1 — per-family ablation: isolate sentiment5/event_type from the bundled
    # "news_adv" set above, so their individual OOS contribution (DM test + bootstrap CI) is
    # visible rather than hidden inside one aggregate ΔR² (guideline Level-1/Gate-F requirement).
    fam_fsets = ["price+sentiment5", "price+event_type", "price+sentiment5+event_type"]
    results["per_family"] = _ablation_block(panel, fam_fsets, "ridge", lines,
                                             "Per-family ablation (Level-1 guideline: sentiment / event-type, Ridge)")

    # Same ablation with GBM (nonlinear, invariant to multicollinearity) — checks whether the
    # Ridge-only null on the COMBINED set (sentiment5+event_type) is a multicollinearity
    # artifact (trees don't need decorrelated inputs) or GBM independently ignores these
    # features too (as it already does for the bundled news_adv set — see metrics.csv, where
    # GBM predictions are byte-identical across price/price+sentiment5/price+event_type/
    # price+sentiment5+event_type, i.e. it never splits on any of them).
    results["per_family_gbm"] = _ablation_block(panel, fam_fsets, "gbm", lines,
                                                 "Per-family ablation (Level-1 guideline: sentiment / event-type, GBM — nonlinear, checks multicollinearity hypothesis)")

    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "significance.json"
    jpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    rpath = outdir / "significance_report.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    return [jpath, rpath]


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
