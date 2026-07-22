"""Phase 9 — Leakage Detection (per EDA Guide).

Enumerates every potential data-leakage source with an explicit status
(fixed / accepted / open) and mitigation. The headline deliverable is
``leakage_list.md`` — a human-readable audit; ``leakage_checks.json`` is the
machine-readable companion.

Checks: timestamp ordering; future-information in features; rolling-window
look-ahead; target leakage (a feature correlating >0.95 with a target); and
time-based (not random) train/test splitting.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.eda.common import ensure_output_dirs, phase_output_dir
from src.eda.phase08_feature_validation import FEATURE_COLS, _build_feature_matrix

TARGETS = ["pk_t+1", "pk_t+5", "pk_t+10", "pk_t+22", "rv_t+1", "rv_t+5", "rv_t+10", "rv_t+22"]
TARGET_LEAKAGE_R = 0.95
TRAIN_TEST_SPLIT = "2025-01-01"  # ≤2024 train, ≥2025 test


# ---------- pure helpers (unit-tested) ----------
def dates_monotonic_per_group(df: pd.DataFrame, group_col: str, date_col: str) -> bool:
    """True if ``date_col`` is non-decreasing within every group."""
    if group_col not in df.columns or date_col not in df.columns:
        return False
    return bool(df.groupby(group_col)[date_col].apply(lambda s: s.is_monotonic_increasing).all())


def target_leakage_flags(df: pd.DataFrame, targets: list[str], threshold: float = TARGET_LEAKAGE_R) -> list[dict]:
    """Feature×target pairs whose |corr| > threshold (possible target leakage)."""
    num = df.select_dtypes(include=[np.number])
    feats = [c for c in FEATURE_COLS if c in num.columns]
    out = []
    for tgt in targets:
        if tgt not in num.columns:
            continue
        for feat in feats:
            if feat == tgt:
                continue
            r = num[feat].corr(num[tgt])
            if pd.notna(r) and abs(r) > threshold:
                out.append({"feature": feat, "target": tgt, "corr": round(float(r), 4)})
    return out


# ---------- phase runner ----------
def _structural_findings(fm: pd.DataFrame) -> list[dict]:
    """Audit each feature/target's construction for leakage (structural, not data-driven)."""
    monotonic = dates_monotonic_per_group(fm, "ticker", "date")
    return [
        {
            "item": "Prediction targets rv_t+1/+5/+10 use ONLY future log-returns (shift -h)",
            "status": "fixed", "evidence": "phase03 volatility_targets; proven by mutation unit test",
        },
        {
            "item": "Prediction targets pk_t+1/+5/+10 = parkinson.shift(-h) (future Parkinson)",
            "status": "fixed", "evidence": "phase03 parkinson_targets; matches baselines' convention",
        },
        {
            "item": "Trailing features (atr_14, realized_vol_5d/20d) use trailing rolling only",
            "status": "fixed", "evidence": "phase03; rolling default trailing, no shift",
        },
        {
            "item": "News features aligned via effective_trading_date (news ≤ close → same day)",
            "status": "fixed", "evidence": "phase04 effective_trading_date; after-close → next day",
        },
        {
            "item": "sentiment_mean filled only when news_available=1; NaN otherwise (no masking)",
            "status": "fixed", "evidence": "phase07 sparse_features; unit-tested",
        },
        {
            "item": "Date ordering monotonic per ticker in the feature matrix",
            "status": "fixed" if monotonic else "open",
            "evidence": f"is_monotonic_increasing per ticker = {monotonic}",
        },
        {
            "item": "Train/test split must be TIME-BASED (≤2024 train, ≥2025 test), never random",
            "status": "accepted", "evidence": f"recommended split = {TRAIN_TEST_SPLIT}; enforced downstream",
        },
        {
            "item": "Normalization leakage: EDA does not fit scalers on full data (modeling concern)",
            "status": "accepted", "evidence": "no normalizers fit in EDA; baselines use per-split StandardScaler",
        },
    ]


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("leakage")
    fm = _build_feature_matrix()
    if fm.empty:
        return []
    written = []

    structural = _structural_findings(fm)
    leakage = target_leakage_flags(fm, TARGETS)
    checks = {"structural": structural, "target_leakage_suspects": leakage,
              "target_leakage_threshold": TARGET_LEAKAGE_R}
    (outdir / "leakage_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    written.append(outdir / "leakage_checks.json")

    # Explicit markdown leakage list
    lines = ["# Leakage List (Phase 9)\n", "> Every potential leakage source, explicit status.\n"]
    lines.append("\n## Structural checks\n")
    for f in structural:
        lines.append(f"- **[{f['status'].upper()}]** {f['item']} — {f['evidence']}")
    lines.append(f"\n## Target-leakage suspects (|corr| > {TARGET_LEAKAGE_R:.2f} with a target)\n")
    if leakage:
        for lk in leakage:
            lines.append(f"- **[{lk['feature']} ↔ {lk['target']}]** r={lk['corr']} — REVIEW before modeling")
    else:
        lines.append(f"- None found (no feature correlates > {TARGET_LEAKAGE_R:.2f} with any target).")
    lines.append("\n## Recommendation\n")
    lines.append("- Use a strict time-based train/test split (≤2024 / ≥2025).")
    lines.append("- Drop or redesign any target-leakage suspect listed above.")
    lines.append("- When modeling, fit all normalizers/encoders on TRAIN only.")
    (outdir / "leakage_list.md").write_text("\n".join(lines), encoding="utf-8")
    written.append(outdir / "leakage_list.md")
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
