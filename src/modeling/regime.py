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
from src.modeling.baseline import PRICE_FEATURES, NEWS_FEATURES, TARGETS, compute_metrics
from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
from src.modeling.features import ADV_FEATURES_DUAL

NEWS_FSET = PRICE_FEATURES + NEWS_FEATURES + ADV_FEATURES_DUAL
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
N_WALK_FOLDS = 3
N_PLACEBO = 20


def _exante_high_vol_flag(series: pd.Series, threshold: float) -> pd.Series:
    """Expanding-window percentile — no look-ahead. NaN until enough history."""
    exp_min = series.expanding().min()
    exp_max = series.expanding().max()
    denom = (exp_max - exp_min).clip(lower=1e-12)
    pct = (series - exp_min) / denom
    return pct >= threshold


def compute_exante_flags(panel: pd.DataFrame, vol_col: str = "realized_vol_20d",
                         thresholds: list[float] | None = None) -> dict[float, pd.DataFrame]:
    """Per-ticker ex-ante high-vol flags at each threshold. Returns {threshold: flag_series}."""
    if thresholds is None:
        thresholds = THRESHOLDS
    result: dict[float, pd.DataFrame] = {}
    for th in thresholds:
        all_flags = []
        for ticker in panel["ticker"].unique():
            sub = panel[panel["ticker"] == ticker].copy().sort_values("date")
            flag = _exante_high_vol_flag(sub[vol_col], th)
            all_flags.append(flag)
        result[th] = pd.concat(all_flags)
    return result


def threshold_sensitivity(panel: pd.DataFrame, target: str,
                          thresholds: list[float] | None = None) -> pd.DataFrame:
    """ΔR² at each threshold — does the high-vol news signal depend on cutoff choice?"""
    if thresholds is None:
        thresholds = THRESHOLDS
    avail_price = [c for c in PRICE_FEATURES if c in panel.columns]
    avail_all = [c for c in NEWS_FSET if c in panel.columns]
    df = panel.dropna(subset=[target]).copy()
    train, test = time_split(df, SPLIT_DATE)

    pipe_p = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_n = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_p.fit(train[avail_price], train[target])
    pipe_n.fit(train[avail_all], train[target])

    rows = []
    flags = compute_exante_flags(df, thresholds=thresholds)
    for th in thresholds:
        mask = flags[th].reindex(test.index, fill_value=False)
        high = test[mask]
        low = test[~mask]
        for regime, sub in [("high", high), ("low", low), ("all", test)]:
            if len(sub) < 10:
                continue
            r2_p = float(pipe_p.score(sub[avail_price], sub[target]))
            r2_n = float(pipe_n.score(sub[avail_all], sub[target]))
            rows.append({
                "threshold": th, "regime": regime, "n": len(sub),
                "price_r2": round(r2_p, 6), "news_r2": round(r2_n, 6),
                "delta_r2": round(r2_n - r2_p, 6),
            })
    return pd.DataFrame(rows)


def high_vol_subset_model(panel: pd.DataFrame, target: str,
                          threshold: float = 0.8) -> dict:
    """Train and evaluate model on high-vol subset only."""
    avail_price = [c for c in PRICE_FEATURES if c in panel.columns]
    avail_all = [c for c in NEWS_FSET if c in panel.columns]
    df = panel.dropna(subset=[target]).copy()
    flags = compute_exante_flags(df, thresholds=[threshold])[threshold]
    df_high = df[flags.reindex(df.index, fill_value=False)].copy()
    if len(df_high) < 50:
        return {"threshold": threshold, "n_high": 0, "error": "insufficient high-vol samples"}
    train_h, test_h = time_split(df_high, SPLIT_DATE)
    if len(train_h) < 20 or len(test_h) < 10:
        return {"threshold": threshold, "n_high": len(df_high), "n_train": len(train_h),
                "n_test": len(test_h), "error": "insufficient split samples"}

    pipe_p = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_n = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_p.fit(train_h[avail_price], train_h[target])
    pipe_n.fit(train_h[avail_all], train_h[target])
    m_p = compute_metrics(test_h[target].to_numpy(), pipe_p.predict(test_h[avail_price]))
    m_n = compute_metrics(test_h[target].to_numpy(), pipe_n.predict(test_h[avail_all]))

    return {
        "threshold": threshold, "n_high": len(df_high), "n_train": len(train_h),
        "n_test": len(test_h),
        "price_only": {"r2": m_p["r2"], "rmse": m_p["rmse"], "mae": m_p["mae"]},
        "price_news": {"r2": m_n["r2"], "rmse": m_n["rmse"], "mae": m_n["mae"]},
        "delta_r2": round(m_n["r2"] - m_p["r2"], 6) if m_n["r2"] is not None and m_p["r2"] is not None else None,
    }


def _dm_test_high_vol(y_true: np.ndarray, pred_price: np.ndarray,
                       pred_news: np.ndarray) -> dict:
    from scipy.stats import norm
    e1 = (y_true - pred_price) ** 2
    e2 = (y_true - pred_news) ** 2
    d = e1 - e2
    n = len(d)
    if n < 2:
        return {"dm_stat": None, "dm_pvalue": None, "n": n}
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return {"dm_stat": 0.0, "dm_pvalue": 1.0, "n": n}
    dm = float(d.mean() / (sd / np.sqrt(n)))
    p = float(2 * (1 - norm.cdf(abs(dm))))
    return {"dm_stat": round(dm, 4), "dm_pvalue": round(p, 4), "n": n}


def _block_bootstrap_ci(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                         block_size: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """Moving block bootstrap for serially dependent errors."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < block_size + 5:
        return {"delta_r2_ci": None, "delta_rmse_ci": None}
    n_blocks = n - block_size + 1
    dr2_vals = []
    for _ in range(n_boot):
        start_idx = rng.integers(0, n_blocks, int(np.ceil(n / block_size)))
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in start_idx])[:n]
        mb = compute_metrics(y_true[idx], pred_b[idx])
        ma = compute_metrics(y_true[idx], pred_a[idx])
        if mb["r2"] is not None and ma["r2"] is not None:
            dr2_vals.append(mb["r2"] - ma["r2"])
    if not dr2_vals:
        return {"delta_r2_ci": None, "delta_rmse_ci": None}
    return {
        "delta_r2_ci": [round(float(np.percentile(dr2_vals, 2.5)), 6),
                        round(float(np.percentile(dr2_vals, 97.5)), 6)],
        "n_boot": n_boot, "block_size": block_size,
    }


def statistical_tests_high_vol(panel: pd.DataFrame, target: str,
                                threshold: float = 0.8) -> dict:
    """DM test + block bootstrap CI on high-vol subset."""
    avail_price = [c for c in PRICE_FEATURES if c in panel.columns]
    avail_all = [c for c in NEWS_FSET if c in panel.columns]
    df = panel.dropna(subset=[target]).copy()
    train, test = time_split(df, SPLIT_DATE)
    flags = compute_exante_flags(df, thresholds=[threshold])[threshold]
    test_high = test[flags.reindex(test.index, fill_value=False)]
    if len(test_high) < 10:
        return {"threshold": threshold, "n_high_vol_test": 0, "error": "insufficient high-vol test samples"}

    pipe_p = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_n = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_p.fit(train[avail_price], train[target])
    pipe_n.fit(train[avail_all], train[target])

    y_true = test_high[target].to_numpy()
    pred_p = pipe_p.predict(test_high[avail_price])
    pred_n = pipe_n.predict(test_high[avail_all])

    dm = _dm_test_high_vol(y_true, pred_p, pred_n)
    boot = _block_bootstrap_ci(y_true, pred_p, pred_n)
    m = compute_metrics(y_true, pred_n - pred_p)

    return {
        "threshold": threshold, "n_high_vol_test": len(test_high),
        "dm": dm, "block_bootstrap": boot,
        "delta_rmse": m["rmse"], "delta_mae": m["mae"],
    }


def placebo_tests(panel: pd.DataFrame, target: str, threshold: float = 0.8) -> dict:
    """Placebo 1: block-shuffle news. Placebo 2: time-shift. Returns dict of results."""
    avail_price = [c for c in PRICE_FEATURES if c in panel.columns]
    avail_all = [c for c in NEWS_FSET if c in panel.columns]
    df = panel.dropna(subset=[target]).copy()
    train, test = time_split(df, SPLIT_DATE)
    news_idx = [c for c in avail_all if c not in avail_price]
    flags = compute_exante_flags(df, thresholds=[threshold])[threshold]
    test_high = test[flags.reindex(test.index, fill_value=False)]
    pipe_p = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_p.fit(train[avail_price], train[target])

    results = {}
    y_true = test_high[target].to_numpy()
    pred_p = pipe_p.predict(test_high[avail_price])

    rng = np.random.default_rng(42)

    # Placebo 1: block-shuffle news (block size = 5)
    block_size = 5
    n_blocks = int(np.ceil(len(train) / block_size))
    dr2_block = []
    for _ in range(N_PLACEBO):
        train_perm = train.copy()
        block_idx = rng.permutation(n_blocks)
        perm_news = train_perm[news_idx].copy()
        for j, col in enumerate(news_idx):
            vals = perm_news[col].to_numpy()
            blocks = [vals[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
            vals_perm = np.concatenate([blocks[b] for b in block_idx])[:len(vals)]
            train_perm[col] = vals_perm
        pipe_n = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe_n.fit(train_perm[avail_all], train_perm[target])
        pred_n = pipe_n.predict(test_high[avail_all])
        mb = compute_metrics(y_true, pred_n)
        dr2_block.append(mb["r2"] - compute_metrics(y_true, pred_p)["r2"]
                         if mb["r2"] is not None else 0)
    results["placebo_block_shuffle"] = {
        "mean_delta_r2": round(float(np.mean(dr2_block)), 6),
        "std_delta_r2": round(float(np.std(dr2_block, ddof=1)), 6),
    }

    # Placebo 2: time-shift news features
    for shift in [-10, -5, 5, 10]:
        df_shift = df.copy()
        for col in news_idx:
            df_shift[col] = df_shift.groupby("ticker")[col].shift(shift)
        train_s, test_s = time_split(df_shift, SPLIT_DATE)
        ts_high = test_s[flags.reindex(test_s.index, fill_value=False)]
        if len(ts_high) < 5:
            continue
        pipe_ns = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe_ns.fit(train_s[avail_all], train_s[target])
        pred_ns = pipe_ns.predict(ts_high[avail_all])
        m_ns = compute_metrics(ts_high[target].to_numpy(), pred_ns)
        m_p = compute_metrics(ts_high[target].to_numpy(),
                              pipe_p.predict(ts_high[avail_price]))
        dr2 = round(m_ns["r2"] - m_p["r2"], 6) if m_ns["r2"] is not None and m_p["r2"] is not None else None
        results[f"placebo_time_shift_{shift}d"] = {
            "delta_r2": dr2, "n": len(ts_high),
        }

    return results


def cross_analysis(panel: pd.DataFrame, target: str, threshold: float = 0.8,
                   cluster_path: str | Path | None = None) -> pd.DataFrame:
    """Sensitive vs non-sensitive ticker × high/low vol regime."""
    if cluster_path is None:
        cluster_path = EDA_OUTPUT_DIR / "modeling" / "ticker_clusters.json"
    cluster_path = Path(cluster_path)
    if cluster_path.exists():
        data = json.loads(cluster_path.read_text(encoding="utf-8"))
        per_ticker = data.get("per_ticker", {})
        sensitive = {t for t, v in per_ticker.items() if v.get("cluster") == "sensitive"}
    else:
        sensitive = set()

    avail_price = [c for c in PRICE_FEATURES if c in panel.columns]
    avail_all = [c for c in NEWS_FSET if c in panel.columns]
    df = panel.dropna(subset=[target]).copy()
    train, test = time_split(df, SPLIT_DATE)
    flags_all = compute_exante_flags(df, thresholds=[threshold])[threshold]
    flags_test = flags_all[test.index]

    pipe_p = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_n = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_p.fit(train[avail_price], train[target])
    pipe_n.fit(train[avail_all], train[target])

    rows = []
    for ticker_group, tickers in [("sensitive", sensitive), ("non_sensitive", set(panel["ticker"].unique()) - sensitive)]:
        for regime_name, regime_mask in [("high", flags_test), ("low", ~flags_test)]:
            regime_mask_aligned = regime_mask.reindex(test.index, fill_value=False)
            sub = test[test["ticker"].isin(tickers) & regime_mask_aligned]
            if len(sub) < 5:
                continue
            r2_p = float(pipe_p.score(sub[avail_price], sub[target]))
            r2_n = float(pipe_n.score(sub[avail_all], sub[target]))
            rows.append({
                "ticker_group": ticker_group, "regime": regime_name,
                "n": len(sub), "price_r2": round(r2_p, 6),
                "news_r2": round(r2_n, 6), "delta_r2": round(r2_n - r2_p, 6),
            })
    return pd.DataFrame(rows)


def walk_forward_evaluation(panel: pd.DataFrame, target: str, n_folds: int = N_WALK_FOLDS,
                             threshold: float = 0.8) -> pd.DataFrame:
    """Walk-forward (expanding window) validation on high-vol subset."""
    avail_price = [c for c in PRICE_FEATURES if c in panel.columns]
    avail_all = [c for c in NEWS_FSET if c in panel.columns]
    df = panel.dropna(subset=[target]).copy().sort_values("date")
    dates = sorted(df["date"].unique())
    if len(dates) < n_folds + 1:
        return pd.DataFrame()
    chunk_size = len(dates) // (n_folds + 1)

    rows = []
    for fold in range(n_folds):
        split_date = dates[(fold + 1) * chunk_size]
        train_fold = df[df["date"] < split_date]
        test_fold = df[df["date"] >= split_date]
        flags = compute_exante_flags(pd.concat([train_fold, test_fold]),
                                     thresholds=[threshold])[threshold]
        test_high = test_fold[flags.reindex(test_fold.index, fill_value=False)]
        if len(train_fold) < 50 or len(test_high) < 5:
            continue
        pipe_p = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe_n = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe_p.fit(train_fold[avail_price], train_fold[target])
        pipe_n.fit(train_fold[avail_all], train_fold[target])
        r2_p = float(pipe_p.score(test_high[avail_price], test_high[target]))
        r2_n = float(pipe_n.score(test_high[avail_all], test_high[target]))
        rows.append({
            "fold": fold, "split_date": str(split_date.date()),
            "n_train": len(train_fold), "n_test_high": len(test_high),
            "price_r2": round(r2_p, 6), "news_r2": round(r2_n, 6),
            "delta_r2": round(r2_n - r2_p, 6),
        })
    return pd.DataFrame(rows)


def make_keep_drop_decision(results: dict) -> dict:
    """Evaluate all evidence and produce a Keep/Drop recommendation per (horizon, regime)."""
    decisions = {}
    for horizon_key, h_results in results.items():
        evidence = h_results.get("evidence", {})
        n_folds_positive = evidence.get("n_folds_positive", 0)
        n_folds_total = evidence.get("n_folds_total", 0)
        placebo_better = evidence.get("placebo_better", True)
        dm_significant = evidence.get("dm_significant", False)
        ci_excludes_zero = evidence.get("ci_excludes_zero", False)
        multi_ticker = evidence.get("multi_ticker", False)
        delta_r2 = evidence.get("delta_r2", 0)

        score = 0
        if delta_r2 > 0:
            score += 1
        if n_folds_total > 0 and n_folds_positive / n_folds_total > 0.5:
            score += 1
        if dm_significant:
            score += 1
        if ci_excludes_zero:
            score += 1
        if not placebo_better:
            score += 1
        if multi_ticker:
            score += 1

        if score >= 4:
            decision = "KEEP"
        elif score >= 2:
            decision = "CONDITIONAL_KEEP"
        else:
            decision = "DROP"

        decisions[horizon_key] = {
            "decision": decision,
            "evidence_score": score,
            "max_score": 6,
            "delta_r2": delta_r2,
            "n_folds_positive": n_folds_positive,
            "n_folds_total": n_folds_total,
            "placebo_better": placebo_better,
            "dm_significant": dm_significant,
            "ci_excludes_zero": ci_excludes_zero,
            "multi_ticker": multi_ticker,
        }
    return decisions


def run() -> list[Path]:
    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    panel = build_panel()
    if panel.empty:
        return written

    all_results: dict = {}
    lines = ["# Epic 19 — Regime-Conditional News Validation\n"]

    lines.append("\n## 19-1: Ex-ante Volatility Regimes\n")
    flags = compute_exante_flags(panel)
    for th in THRESHOLDS:
        n_high = int(flags[th].sum())
        n_total = int(len(flags[th]))
        lines.append(f"- Threshold {th:.0%}: {n_high}/{n_total} high-vol observations")

    lines.append("\n## 19-2: Threshold Sensitivity\n")
    for target in TARGETS:
        ts = threshold_sensitivity(panel, target)
        if not ts.empty:
            csv_path = outdir / f"threshold_sensitivity_{target}.csv"
            ts.to_csv(csv_path, index=False, encoding="utf-8")
            written.append(csv_path)
            lines.append(f"\n### {target}\n")
            lines.append(f"| Threshold | Regime | n | price_r2 | news_r2 | delta_r2 |")
            lines.append(f"|-----------|--------|---|----------|---------|----------|")
            for _, r in ts.iterrows():
                lines.append(f"| {r['threshold']:.0%} | {r['regime']:6s} | {r['n']} | {r['price_r2']:.6f} | {r['news_r2']:.6f} | {r['delta_r2']:+.6f} |")

    lines.append("\n## 19-3: High-Vol Subset Model\n")
    for target in TARGETS:
        hv = high_vol_subset_model(panel, target)
        all_results.setdefault(target, {})["high_vol_subset"] = hv
        lines.append(f"\n### {target}\n")
        if hv.get("error"):
            lines.append(f"- {hv['error']}")
        else:
            hv_r2 = hv["price_news"]["r2"]
            p_r2 = hv["price_only"]["r2"]
            lines.append(f"- n_high={hv['n_high']}, train={hv['n_train']}, test={hv['n_test']}")
            lines.append(f"- Price-only R²={p_r2}")
            lines.append(f"- Price+news R²={hv_r2}")
            lines.append(f"- ΔR²={hv['delta_r2']:+.6f}")

    lines.append("\n## 19-4: Sensitive × High-Vol Cross Analysis\n")
    for target in TARGETS:
        ca = cross_analysis(panel, target)
        if not ca.empty:
            csv_path = outdir / f"cross_analysis_{target}.csv"
            ca.to_csv(csv_path, index=False, encoding="utf-8")
            written.append(csv_path)
            lines.append(f"\n### {target}\n")
            for _, r in ca.iterrows():
                lines.append(f"- {r['ticker_group']:15s} / {r['regime']:6s} (n={r['n']}): ΔR²={r['delta_r2']:+.6f}")

    lines.append("\n## 19-5: Walk-Forward Validation\n")
    for target in TARGETS:
        wf = walk_forward_evaluation(panel, target)
        if not wf.empty:
            csv_path = outdir / f"walk_forward_{target}.csv"
            wf.to_csv(csv_path, index=False, encoding="utf-8")
            written.append(csv_path)
            lines.append(f"\n### {target}\n")
            n_pos = int((wf["delta_r2"] > 0).sum())
            all_results.setdefault(target, {}).setdefault("evidence", {})["n_folds_positive"] = n_pos
            all_results.setdefault(target, {}).setdefault("evidence", {})["n_folds_total"] = len(wf)
            lines.append(f"- Folds with positive ΔR²: {n_pos}/{len(wf)}")
            for _, r in wf.iterrows():
                lines.append(f"  Fold {r['fold']}: ΔR²={r['delta_r2']:+.6f} (n_test_high={r['n_test_high']})")

    lines.append("\n## 19-6: Statistical Tests + Placebos\n")
    for target in TARGETS:
        st = statistical_tests_high_vol(panel, target)
        all_results.setdefault(target, {}).setdefault("evidence", {}).update({
            "dm_pvalue": st.get("dm", {}).get("dm_pvalue"),
            "dm_significant": st.get("dm", {}).get("dm_pvalue") is not None and st["dm"]["dm_pvalue"] < 0.05,
            "ci_excludes_zero": (st.get("block_bootstrap", {}).get("delta_r2_ci") is not None and
                                 st["block_bootstrap"]["delta_r2_ci"][0] > 0),
        })
        lines.append(f"\n### {target}\n")
        lines.append(f"- DM stat={st.get('dm', {}).get('dm_stat')}, p={st.get('dm', {}).get('dm_pvalue')}")
        ci = st.get("block_bootstrap", {}).get("delta_r2_ci")
        lines.append(f"- Block-bootstrap 95% CI: [{ci[0] if ci else 'N/A'}, {ci[1] if ci else 'N/A'}]")
        lines.append(f"- n_high_vol_test={st.get('n_high_vol_test', 0)}")

        plc = placebo_tests(panel, target)
        all_results.setdefault(target, {}).setdefault("evidence", {})["placebo_better"] = (
            any(
                v.get("delta_r2", v.get("mean_delta_r2", 0)) is not None
                and v.get("delta_r2", v.get("mean_delta_r2", 0)) > 0
                for v in plc.values()
            )
        )
        lines.append(f"\n  Placebo tests:")
        for k, v in plc.items():
            lines.append(f"  - {k}: ΔR²={v.get('delta_r2', v.get('mean_delta_r2', 'N/A'))}")

    jpath = outdir / "regime_analysis.json"
    jpath.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    written.append(jpath)

    lines.append("\n## 19-7: Keep/Drop Decision\n")
    decisions = make_keep_drop_decision(all_results)
    for horizon, dec in decisions.items():
        lines.append(f"- {horizon}: **{dec['decision']}** (score={dec['evidence_score']}/{dec['max_score']}, "
                     f"ΔR²={dec['delta_r2']})")
    lines.append(f"\n### Decision rationale\n")
    for horizon, dec in decisions.items():
        reasons = []
        if dec["n_folds_positive"] > 0 and dec["n_folds_total"] > 0:
            reasons.append(f"{dec['n_folds_positive']}/{dec['n_folds_total']} folds positive")
        if dec.get("dm_significant"):
            reasons.append("DM test significant")
        else:
            reasons.append("DM test NOT significant")
        if dec.get("ci_excludes_zero"):
            reasons.append("Bootstrap CI excludes zero")
        else:
            reasons.append("Bootstrap CI includes zero")
        reasons.append(f"Placebo {'better' if dec.get('placebo_better') else 'not better'} than real")
        lines.append(f"- {horizon}: {', '.join(reasons)}")

    rpath = outdir / "regime_validation_report.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    written.append(rpath)
    return written


if __name__ == "__main__":
    for p in run():
        print(f"Wrote {p}")
