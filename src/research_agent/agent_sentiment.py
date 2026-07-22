"""Agent 1/5: Sentiment methods — tests VADER, TextBlob, lexicon approaches."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.research_agent.base import Registry
from src.research_agent.report import generate_comparison_report
from src.research_agent.storage import save_result

logger = logging.getLogger("agent_sentiment")
handler = logging.FileHandler("results/research_agent/logs/agent_sentiment.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

CYCLE_SLEEP_S = 1800


def run_cycle(cycle: int):
    logger.info(f"\n{'='*50}\nCycle {cycle} — {datetime.now(timezone.utc).isoformat()}")
    for name in ["vader_sentiment", "textblob_sentiment"]:
        exp_cls = Registry.get(name)
        if exp_cls is None:
            logger.warning(f"  {name}: not registered")
            continue
        try:
            exp = exp_cls()
            params = exp.default_params()
            result = exp.run(**params)
            eid = save_result(
                name=result.name, category=result.category,
                params={**result.params, "agent_cycle": cycle},
                metrics=result.metrics, started_at=result.started_at,
                finished_at=result.finished_at, duration_s=result.duration_s,
                status=result.status, error=result.error,
            )
            logger.info(f"  {name}: r2={result.metrics.get('r2', 'N/A'):.6f}  "
                        f"dur={result.duration_s:.1f}s  id={eid}  {result.status}")
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}")

    try:
        path = generate_comparison_report("sentiment")
        logger.info(f"  Report: {path}")
    except Exception as e:
        logger.warning(f"  Report failed: {e}")

    logger.info(f"Cycle {cycle} done. Sleep {CYCLE_SLEEP_S}s")
    time.sleep(CYCLE_SLEEP_S)


def main():
    logger.info("=" * 50)
    logger.info("AGENT SENTIMENT STARTED")
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
