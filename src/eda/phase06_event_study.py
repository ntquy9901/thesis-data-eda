"""Phase 6 — Event Study (per EDA Guide).

For each important news event, compare realized/Parkinson volatility, return,
and abnormal volatility in pre (T-10/T-5/T-1) vs post (T+1/T+5/T+10) windows.

Outputs (under ``eda_output/relationship/``):
- ``event_study.csv`` — per-event × per-horizon metrics
- ``event_study_plot.png`` — average abnormal vol by horizon

Reads: ``eda_output/news/sparse_news_features.parquet`` + per-ticker
``price_metrics_<ticker>.parquet``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.eda.common import ensure_output_dirs, phase_output_dir
from src.eda.phase05_relationship import _load_joined_panel  # noqa: F401 (reuse)

HORIZONS = (1, 5, 10)
EVENTS_PER_TICKER = 20


# ---------- pure helpers (unit-tested) ----------
def window_mean(series: pd.Series, start: int, end: int) -> float | None:
    """Mean of series.iloc[start:end+1], NaN-dropped. None if empty/out-of-range."""
    if start < 0 or end >= len(series):
        return None
    seg = series.iloc[start : end + 1].dropna()
    return float(seg.mean()) if len(seg) else None


def window_sum(series: pd.Series, start: int, end: int) -> float | None:
    """Sum of series.iloc[start:end+1], NaN-dropped. None if empty/out-of-range."""
    if start < 0 or end >= len(series):
        return None
    seg = series.iloc[start : end + 1].dropna()
    return float(seg.sum()) if len(seg) else None


def realized_vol_window(log_returns: pd.Series, start: int, end: int) -> float | None:
    """Realized vol = sqrt(sum of squared log returns) over [start, end]."""
    if start < 0 or end >= len(log_returns):
        return None
    seg = log_returns.iloc[start : end + 1].dropna()
    return float(np.sqrt((seg**2).sum())) if len(seg) else None


def event_window_metrics(event_idx: int, parkinson: pd.Series, log_returns: pd.Series,
                         horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Pre/post Parkinson vol + realized vol + return + abnormal vol per horizon.

    ``pre_vol``/``post_vol`` are mean Parkinson; ``pre_realized``/``post_realized``
    are sqrt(sum of squared log returns) — both per the EDA Guide spec.
    """
    rows = []
    for h in horizons:
        pre_v = window_mean(parkinson, event_idx - h, event_idx - 1)
        post_v = window_mean(parkinson, event_idx + 1, event_idx + h)
        pre_rv = realized_vol_window(log_returns, event_idx - h, event_idx - 1)
        post_rv = realized_vol_window(log_returns, event_idx + 1, event_idx + h)
        pre_r = window_sum(log_returns, event_idx - h, event_idx - 1)
        post_r = window_sum(log_returns, event_idx + 1, event_idx + h)
        abnormal = (post_v - pre_v) if (pre_v is not None and post_v is not None) else None
        rows.append({
            "horizon": h, "pre_vol": pre_v, "post_vol": post_v,
            "pre_realized": pre_rv, "post_realized": post_rv,
            "abnormal_vol": abnormal, "pre_return": pre_r, "post_return": post_r,
        })
    return pd.DataFrame(rows)


def select_event_indices(score: pd.Series, top_n: int = EVENTS_PER_TICKER) -> list[int]:
    """Indices of the ``top_n`` rows with the largest |score| (NaN ignored)."""
    s = score.abs().dropna()
    if s.empty:
        return []
    return list(s.nlargest(min(top_n, len(s))).index)


# ---------- phase runner ----------
def _load_price(ticker: str) -> pd.DataFrame:
    from src.eda.common import EDA_OUTPUT_DIR

    pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
    if not pq.exists():
        return pd.DataFrame()
    df = pd.read_parquet(pq)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _plot_event_study(events: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    if events.empty:
        return
    agg = events.groupby("horizon")["abnormal_vol"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    agg.plot.bar(ax=ax, color="darkorange")
    ax.set_title("Event study: mean abnormal Parkinson vol by horizon")
    ax.set_ylabel("post_vol − pre_vol")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_phase() -> list:
    from src.eda.common import configure_plots

    ensure_output_dirs()
    configure_plots()  # Agg backend + plot style for headless PNG
    outdir = phase_output_dir("relationship")
    panel = _load_joined_panel()
    if panel.empty:
        return []
    written = []
    all_events = []

    for ticker, sub in panel.groupby("ticker"):
        price = _load_price(ticker)
        if price.empty or "parkinson_vol" not in price or "log_returns" not in price:
            continue
        date_to_idx = {d: i for i, d in enumerate(price["date"])}
        # align log_returns + parkinson to the sorted price positional index
        lr = price["log_returns"].reset_index(drop=True)
        pk = price["parkinson_vol"].reset_index(drop=True)

        # event score: |sentiment| (top-magnitude sentiment days, per spec);
        # fall back to news_count if sentiment unavailable
        sub_idx = sub.set_index("trading_date")
        if "sentiment_mean" in sub_idx.columns:
            score = sub_idx["sentiment_mean"]
        else:
            score = sub_idx.get("news_count_1d", pd.Series(dtype=float))
        if score is None or score.empty:
            continue
        event_dates = select_event_indices(score)  # abs + dropna inside
        for d in event_dates:
            idx = date_to_idx.get(pd.Timestamp(d).normalize())
            if idx is None:
                continue
            metrics = event_window_metrics(idx, pk, lr)
            metrics.insert(0, "ticker", ticker)
            metrics.insert(1, "event_date", pd.Timestamp(d).normalize())
            all_events.append(metrics)

    if not all_events:
        return []
    events = pd.concat(all_events, ignore_index=True)
    csv_path = outdir / "event_study.csv"
    events.to_csv(csv_path, index=False, encoding="utf-8")
    written.append(csv_path)
    plot_path = outdir / "event_study_plot.png"
    _plot_event_study(events, plot_path)
    written.append(plot_path)

    summary = {
        "n_events": int(events["event_date"].nunique()),
        "mean_abnormal_vol_by_horizon": events.groupby("horizon")["abnormal_vol"].mean().round(5).to_dict(),
    }
    (outdir / "event_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(outdir / "event_study_summary.json")
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
