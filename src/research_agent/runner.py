"""Continuous research agent — lặp vô hạn: research → experiment → log → sleep."""

from __future__ import annotations

import importlib
import logging
import time
from datetime import datetime, timezone

from src.research_agent.base import Registry
from src.research_agent.report import generate_comparison_report
from src.research_agent.research import run_research_cycle
from src.research_agent.storage import save_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self, sleep_minutes: int = 60):
        self.sleep_seconds = sleep_minutes * 60
        self.cycle = 0
        self._load_all_experiments()

    def _load_all_experiments(self):
        for mod_name in [
            "src.research_agent.experiments.sentiment_methods",
            "src.research_agent.experiments.model_compare",
            "src.research_agent.experiments.news_features",
        ]:
            try:
                importlib.import_module(mod_name)
                logger.info(f"Loaded experiments from {mod_name}")
            except Exception as e:
                logger.warning(f"Failed to load {mod_name}: {e}")

    def run_cycle(self):
        self.cycle += 1
        logger.info(f"\n{'='*60}\nCycle {self.cycle} — {datetime.now(timezone.utc).isoformat()}\n{'='*60}")

        # Phase 1: Research
        logger.info("Phase 1: Researching new methods...")
        try:
            findings = run_research_cycle()
            logger.info(f"Found {len(findings)} papers/articles")
        except Exception as e:
            logger.warning(f"Research phase failed: {e}")

        # Phase 2: Run all registered experiments
        logger.info("Phase 2: Running experiments...")
        for name in Registry.list():
            exp_cls = Registry.get(name)
            if exp_cls is None:
                continue
            try:
                exp = exp_cls()
                params = exp.default_params()
                logger.info(f"  Running {name} ({exp.category})...")
                result = exp.run(**params)
                eid = save_result(
                    name=result.name, category=result.category,
                    params=result.params, metrics=result.metrics,
                    started_at=result.started_at, finished_at=result.finished_at,
                    duration_s=result.duration_s, status=result.status,
                    error=result.error, artifacts=result.artifacts,
                )
                logger.info(f"    Result: r2={result.metrics.get('r2', 'N/A')}  "
                            f"dur={result.duration_s:.1f}s  id={eid}  status={result.status}")
            except Exception as e:
                logger.error(f"  Experiment {name} failed: {e}")

        # Phase 3: Generate comparison reports
        logger.info("Phase 3: Generating reports...")
        for cat in Registry.categories():
            try:
                path = generate_comparison_report(cat)
                logger.info(f"  Report: {path}")
            except Exception as e:
                logger.warning(f"  Report for {cat} failed: {e}")

        logger.info(f"Cycle {self.cycle} complete. Sleeping {self.sleep_seconds}s...")

    def run_forever(self, max_cycles: int | None = None):
        logger.info(f"Research Agent started. Sleep={self.sleep_seconds}s, max_cycles={max_cycles or 'infinite'}")
        while True:
            self.run_cycle()
            if max_cycles and self.cycle >= max_cycles:
                logger.info(f"Reached max_cycles={max_cycles}. Stopping.")
                break
            time.sleep(self.sleep_seconds)


def run(targets: list[str] | None = None, feature_sets: list[str] | None = None,
        models: list[str] | None = None, sleep_minutes: int = 0):
    """Single-shot run (không loop). Chạy experiments và in comparison."""
    agent = ResearchAgent()
    agent._load_all_experiments()

    logger.info("Running all experiments...")
    for name in Registry.list():
        exp_cls = Registry.get(name)
        if exp_cls is None:
            continue
        try:
            exp = exp_cls()
            result = exp.run()
            save_result(
                name=result.name, category=result.category,
                params=result.params, metrics=result.metrics,
                started_at=result.started_at, finished_at=result.finished_at,
                duration_s=result.duration_s, status=result.status,
                error=result.error,
            )
            logger.info(f"  {name}: r2={result.metrics.get('r2', 'N/A')}  status={result.status}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    logger.info("\nComparison reports:")
    for cat in Registry.categories():
        path = generate_comparison_report(cat)
        logger.info(f"  {cat}: {path}")


if __name__ == "__main__":
    import sys
    if "--forever" in sys.argv:
        mins = 60
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            mins = int(sys.argv[2])
        agent = ResearchAgent(sleep_minutes=mins)
        agent.run_forever()
    else:
        # Single-shot (dùng trong CI / dev)
        run()
