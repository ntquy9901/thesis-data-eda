"""Agent 5/5: Per-ticker analysis — tests per-ticker RF with different windows."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src.research_agent.base import Registry
from src.research_agent.storage import save_result, get_all_results
from config import PROJECT_ROOT, EDA_OUTPUT_DIR

logger = logging.getLogger("agent_ticker")
handler = logging.FileHandler("results/research_agent/logs/agent_ticker.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

CYCLE_SLEEP_S = 3600

WINDOWS = [
    {"name": "1yr", "train_start": "2025-01-02", "train_end": "2025-12-31"},
    {"name": "3yr", "train_start": "2023-01-02", "train_end": "2025-12-31"},
    {"name": "5yr", "train_start": "2021-01-04", "train_end": "2025-12-31"},
]


def _run_ticker_window(target: str, window: dict, top_n: int = 5):
    from sklearn.ensemble import RandomForestRegressor
    from src.modeling.dataset import TARGETS as TGT, build_panel

    PRICE_FEATS = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
    TEST_START = "2026-01-02"
    TEST_END = "2026-01-31"
    from config import EDA_TICKERS

    panel = build_panel()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    avail = [c for c in PRICE_FEATS if c in panel.columns]
    rows = []
    for ticker in sorted(EDA_TICKERS):
        tdf = panel[panel["ticker"] == ticker].dropna(subset=[target]]
        tr = tdf[(tdf["date"] >= window["train_start"]) & (tdf["date"] <= window["train_end"])]
        te = tdf[(tdf["date"] >= TEST_START) & (tdf["date"] <= TEST_END)]
        if len(tr) < 20 or len(te) < 5:
            continue
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0).fit(tr[avail], tr[target])
        yt = te[target].to_numpy()
        yp = rf.predict(te[avail])
        from sklearn.metrics import r2_score
        r2 = float(r2_score(yt, yp))
        rows.append({"ticker": ticker, "r2": round(r2, 6), "n_train": len(tr), "n_test": len(te)})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    top = df.nlargest(top_n, "r2")
    return {
        "n_tickers": len(df),
        "mean_r2": round(float(df["r2"].mean()), 6),
        "median_r2": round(float(df["r2"].median()), 6),
        f"top1_r2_{df.iloc[0]['ticker']}": float(df.iloc[0]["r2"]),
        f"top1_ticker": df.iloc[0]["ticker"],
        f"worst_r2_{df.iloc[-1]['ticker']}": float(df.iloc[-1]["r2"]),
    }


def run_cycle(cycle: int):
    logger.info(f"\n{'='*50}\nCycle {cycle} — {datetime.now(timezone.utc).isoformat()}")
    idx = cycle % len(WINDOWS)
    window = WINDOWS[idx]
    target = "pk_t+1"
    logger.info(f"  Testing window={window['name']} target={target}")

    try:
        metrics = _run_ticker_window(target, window)
        if metrics:
            eid = save_result(
                name=f"ticker_{window['name']}", category="ticker_compare",
                params={"window": window["name"], "target": target, "agent_cycle": cycle},
                metrics=metrics, started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=0, status="done",
            )
            logger.info(f"  mean_r2={metrics.get('mean_r2', 'N/A')}  "
                        f"n_tickers={metrics.get('n_tickers', 'N/A')}  id={eid}")
        else:
            logger.warning("  No ticker results (empty)")
    except Exception as e:
        logger.error(f"  Ticker cycle FAILED: {e}")

    logger.info(f"Cycle {cycle} done. Sleep {CYCLE_SLEEP_S}s")
    time.sleep(CYCLE_SLEEP_S)


def main():
    logger.info("=" * 50)
    logger.info("AGENT TICKER STARTED")
    logger.info("=" * 50)
    from src.research_agent.runner import ResearchAgent
    ra = ResearchAgent()
    ra._load_all_experiments()
    cycle = 0
    while True:
        cycle += 1
        run_cycle(cycle)


if __name__ == "__main__":
    main()
