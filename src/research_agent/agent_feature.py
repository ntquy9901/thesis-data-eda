"""Agent 3/5: Feature engineering — tests feature set combinations systematically."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.research_agent.base import Registry
from src.research_agent.report import generate_comparison_report
from src.research_agent.storage import save_result

logger = logging.getLogger("agent_feature")
handler = logging.FileHandler("results/research_agent/logs/agent_feature.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

CYCLE_SLEEP_S = 3600

FEATURE_SETS_TO_TEST = [
    "price",
    "price+news_basic",
    "price+news_adv",
    "price+news_adv_dual",
    "price+news_adv_dual_ewma30",
    "price+news_adv_novelty",
    "price+news_adv_multi_ewma",
    "price+news_adv_full",
    "price+sentiment5",
    "price+event_type",
    "price+sentiment5+event_type",
]


def _test_fset(fset_name: str, target: str = "pk_t+1", model: str = "ridge"):
    from src.modeling.baseline import FEATURE_SETS, make_pipeline, compute_metrics as base_metrics
    from src.modeling.dataset import SPLIT_DATE, build_panel, time_split

    panel = build_panel()
    feats = [c for c in FEATURE_SETS[fset_name] if c in panel.columns]
    train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
    pipe = make_pipeline(model).fit(train[feats], train[target])
    y_pred = pipe.predict(test[feats])
    y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
    m = base_metrics(test[target].to_numpy(), y_pred, y_prev)
    return {k: float(v) if v is not None else float("nan") for k, v in m.items()}, len(feats)


def run_cycle(cycle: int):
    logger.info(f"\n{'='*50}\nCycle {cycle} — {datetime.now(timezone.utc).isoformat()}")
    idx = cycle % len(FEATURE_SETS_TO_TEST)
    fset = FEATURE_SETS_TO_TEST[idx]
    logger.info(f"  Testing feature_set={fset} (idx={idx}/{len(FEATURE_SETS_TO_TEST)})")

    try:
        metrics, nf = _test_fset(fset)
        eid = save_result(
            name=f"feature_{fset.replace('+', '_')}", category="feature_compare",
            params={"feature_set": fset, "n_features": nf, "agent_cycle": cycle, "model": "ridge", "target": "pk_t+1"},
            metrics=metrics, started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            duration_s=0, status="done",
        )
        logger.info(f"  n_features={nf}  r2={metrics.get('r2', 'N/A'):.6f}  "
                    f"rmse={metrics.get('rmse', 'N/A'):.8f}  id={eid}")
    except Exception as e:
        logger.error(f"  {fset} FAILED: {e}")

    try:
        path = generate_comparison_report("feature_compare")
        logger.info(f"  Report: {path}")
    except Exception as e:
        logger.warning(f"  Report failed: {e}")

    logger.info(f"Cycle {cycle} done. Sleep {CYCLE_SLEEP_S}s")
    time.sleep(CYCLE_SLEEP_S)


def main():
    logger.info("=" * 50)
    logger.info("AGENT FEATURE STARTED")
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
