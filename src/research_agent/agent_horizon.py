"""Agent 4/5: Horizon analysis — tests pk_t+1/+5/+10/+22 across feature sets."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.research_agent.base import Registry
from src.research_agent.report import generate_comparison_report
from src.research_agent.storage import save_result

logger = logging.getLogger("agent_horizon")
handler = logging.FileHandler("results/research_agent/logs/agent_horizon.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

CYCLE_SLEEP_S = 3600
TARGETS = ["pk_t+1", "pk_t+5", "pk_t+10", "pk_t+22"]
FEATURE_SETS = ["price", "price+news_basic", "price+news_adv_dual", "price+news_adv_full"]


def _run_cfg(model: str, target: str, fset: str):
    from src.modeling.baseline import FEATURE_SETS as FS, make_pipeline, compute_metrics as base_metrics
    from src.modeling.dataset import SPLIT_DATE, build_panel, time_split

    panel = build_panel()
    feats = [c for c in FS[fset] if c in panel.columns]
    train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
    pipe = make_pipeline(model).fit(train[feats], train[target])
    y_pred = pipe.predict(test[feats])
    y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
    m = base_metrics(test[target].to_numpy(), y_pred, y_prev)
    return {k: float(v) if v is not None else float("nan") for k, v in m.items()}, len(feats)


def run_cycle(cycle: int):
    logger.info(f"\n{'='*50}\nCycle {cycle} — {datetime.now(timezone.utc).isoformat()}")
    idx = cycle % (len(TARGETS) * len(FEATURE_SETS))
    t_idx = idx // len(FEATURE_SETS)
    f_idx = idx % len(FEATURE_SETS)
    target = TARGETS[t_idx]
    fset = FEATURE_SETS[f_idx]
    logger.info(f"  Testing target={target} feature_set={fset}")

    try:
        metrics, nf = _run_cfg("ridge", target, fset)
        eid = save_result(
            name=f"horizon_{target}_{fset.replace('+', '_')}", category="horizon_compare",
            params={"target": target, "feature_set": fset, "n_features": nf,
                    "model": "ridge", "agent_cycle": cycle},
            metrics=metrics, started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0, status="done",
        )
        logger.info(f"  r2={metrics.get('r2', 'N/A'):.6f}  rmse={metrics.get('rmse', 'N/A'):.8f}  id={eid}")
    except Exception as e:
        logger.error(f"  target={target} fset={fset} FAILED: {e}")

    try:
        path = generate_comparison_report("horizon_compare")
        logger.info(f"  Report: {path}")
    except Exception as e:
        logger.warning(f"  Report failed: {e}")

    logger.info(f"Cycle {cycle} done. Sleep {CYCLE_SLEEP_S}s")
    time.sleep(CYCLE_SLEEP_S)


def main():
    logger.info("=" * 50)
    logger.info("AGENT HORIZON STARTED")
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
