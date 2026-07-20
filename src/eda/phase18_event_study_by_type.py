"""Phase 18 — Level 2 Event Study, segmented by event type (Story 14-2).

Per ``docs/gpt-guide/news_feature_evaluation_guideline.md`` Level 2: "Treat each event
independently" (earnings, dividend, M&A, management, regulation, macro, sector — the existing
``TOPIC_CATEGORIES`` taxonomy) rather than Phase 6's single top-|sentiment|-magnitude selection
pooling ALL event types together. Adds **abnormal return** (vs. an equal-weighted VN30 market
benchmark) and **Cumulative Abnormal Return (CAR)** — Phase 6 only had abnormal volatility.

Window: T-5 .. T0 .. T+10 (horizons 1, 5, 10), matching Phase 6's HORIZONS for comparability.

Outputs -> eda_output/event_study_by_type/:
- ``event_study_by_type.csv`` — per (ticker, event_type, event_date, horizon) metrics
- ``event_study_by_type_summary.json`` — per (event_type, horizon): mean abnormal vol/CAR + t-test
"""

from __future__ import annotations

import json

import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir
from src.eda.phase04_news_eda import TOPIC_CATEGORIES
from src.eda.phase06_event_study import HORIZONS, window_mean, window_sum
from src.features.sentiment_scores import _explode_tickers, build_article_sentiment

EVENT_TYPES = list(TOPIC_CATEGORIES)


# ---------- pure helpers (unit-tested) ----------
def market_benchmark_returns(price_by_ticker: dict[str, pd.Series]) -> pd.Series:
    """Equal-weighted market benchmark: mean log_return across tickers, per date (index).

    ``price_by_ticker`` maps ticker -> a date-indexed log_returns Series. Dates present for
    fewer than 2 tickers are still included (mean of whatever is available that date)."""
    if not price_by_ticker:
        return pd.Series(dtype=float)
    aligned = pd.concat(price_by_ticker, axis=1, sort=True)
    return aligned.mean(axis=1, skipna=True)


def event_type_window_metrics(
    event_idx: int, parkinson: pd.Series, log_returns: pd.Series, abnormal_returns: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Pre/post Parkinson vol + return + abnormal vol + CAR per horizon, for one event."""
    rows = []
    for h in horizons:
        pre_v = window_mean(parkinson, event_idx - h, event_idx - 1)
        post_v = window_mean(parkinson, event_idx + 1, event_idx + h)
        pre_r = window_sum(log_returns, event_idx - h, event_idx - 1)
        post_r = window_sum(log_returns, event_idx + 1, event_idx + h)
        pre_car = window_sum(abnormal_returns, event_idx - h, event_idx - 1)
        post_car = window_sum(abnormal_returns, event_idx + 1, event_idx + h)
        abnormal_vol = (post_v - pre_v) if (pre_v is not None and post_v is not None) else None
        rows.append({
            "horizon": h, "pre_vol": pre_v, "post_vol": post_v, "abnormal_vol": abnormal_vol,
            "pre_return": pre_r, "post_return": post_r,
            "pre_car": pre_car, "post_car": post_car,
        })
    return pd.DataFrame(rows)


def event_days_by_type(exploded: pd.DataFrame, ticker: str, event_type: str) -> list:
    """Distinct dates where ``ticker`` had >=1 article flagged with ``event_type`` (deduped)."""
    col = f"event_{event_type}"
    if exploded.empty or col not in exploded.columns:
        return []
    sub = exploded[(exploded["ticker"] == ticker) & (exploded[col] > 0)]
    return sorted(sub["date"].unique().tolist())


# ---------- phase runner ----------
def _load_price(ticker: str) -> pd.DataFrame:
    pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
    if not pq.exists():
        return pd.DataFrame()
    df = pd.read_parquet(pq)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _one_sample_ttest(values: pd.Series) -> dict:
    from scipy.stats import ttest_1samp

    # window_mean/window_sum return None (not NaN) at series edges, which makes a mixed
    # column dtype=object -> scipy's ttest chokes on the object dtype rather than NaN values.
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 5:
        return {"mean": None, "t": None, "pvalue": None, "significant": False, "n": int(len(v))}
    t, p = ttest_1samp(v, 0.0)
    return {"mean": round(float(v.mean()), 6), "t": round(float(t), 3),
            "pvalue": round(float(p), 4), "significant": bool(p < 0.05), "n": int(len(v))}


def run_phase() -> list:
    from src.eda.common import configure_plots

    ensure_output_dirs()
    configure_plots()
    outdir = phase_output_dir("event_study_by_type")

    prices = {t: _load_price(t) for t in EDA_TICKERS}
    prices = {t: df for t, df in prices.items() if not df.empty and "log_returns" in df.columns}
    if not prices:
        return []

    log_ret_by_ticker = {t: df.set_index("date")["log_returns"] for t, df in prices.items()}
    market = market_benchmark_returns(log_ret_by_ticker)

    exploded = _explode_tickers(build_article_sentiment())
    if exploded.empty:
        return []

    all_rows = []
    for ticker, price in prices.items():
        if "parkinson_vol" not in price.columns:
            continue
        date_to_idx = {d: i for i, d in enumerate(price["date"])}
        lr = price["log_returns"].reset_index(drop=True)
        pk = price["parkinson_vol"].reset_index(drop=True)
        mkt_aligned = price["date"].map(market).reset_index(drop=True)
        abnormal = (lr - mkt_aligned).reset_index(drop=True)

        for event_type in EVENT_TYPES:
            for d in event_days_by_type(exploded, ticker, event_type):
                idx = date_to_idx.get(pd.Timestamp(d).normalize())
                if idx is None:
                    continue
                metrics = event_type_window_metrics(idx, pk, lr, abnormal)
                metrics.insert(0, "ticker", ticker)
                metrics.insert(1, "event_type", event_type)
                metrics.insert(2, "event_date", pd.Timestamp(d).normalize())
                all_rows.append(metrics)

    if not all_rows:
        return []
    events = pd.concat(all_rows, ignore_index=True)
    written = []
    csv_path = outdir / "event_study_by_type.csv"
    events.to_csv(csv_path, index=False, encoding="utf-8")
    written.append(csv_path)

    summary = {}
    for (etype, h), grp in events.groupby(["event_type", "horizon"]):
        summary[f"{etype}_h{h}"] = {
            "n_events": int(grp["event_date"].nunique()),
            "abnormal_vol_ttest": _one_sample_ttest(grp["abnormal_vol"]),
            "post_car_ttest": _one_sample_ttest(grp["post_car"]),
        }
    summary_path = outdir / "event_study_by_type_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.append(summary_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
