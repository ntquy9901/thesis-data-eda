"""Agent 2/5: Model architectures — tests Ridge, RF, HGB, XGBoost with varying params."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.research_agent.base import Experiment, ExperimentResult, Registry
from src.research_agent.report import generate_comparison_report
from src.research_agent.storage import save_result

logger = logging.getLogger("agent_model")
handler = logging.FileHandler("results/research_agent/logs/agent_model.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

CYCLE_SLEEP_S = 3600


@Registry.register
class XGBoostExperiment(Experiment):
    name = "xgboost"
    category = "model_compare"
    description = "XGBoost regressor — gradient boosting alternative"

    def run(self, **kwargs) -> ExperimentResult:
        import time as _time
        from datetime import timezone as _tz
        started = datetime.now(_tz.utc).isoformat()
        t0 = _time.time()
        try:
            import xgboost as xgb
            from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
            from src.modeling.baseline import FEATURE_SETS, compute_metrics as base_metrics
            target = kwargs.get("target", "pk_t+1")
            fset = kwargs.get("feature_set", "price+news_adv_dual")
            panel = build_panel()
            feats = [c for c in FEATURE_SETS[fset] if c in panel.columns]
            train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
            model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=0)
            model.fit(train[feats], train[target])
            y_pred = model.predict(test[feats])
            y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
            metrics = base_metrics(test[target].to_numpy(), y_pred, y_prev)
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"model": "XGBoost", "target": target, "feature_set": fset},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(_tz.utc).isoformat(),
                duration_s=round(_time.time() - t0, 2), status="done")
        except ImportError:
            return ExperimentResult(
                name=self.name, category=self.category, params={},
                metrics={}, started_at=started,
                finished_at=datetime.now(_tz.utc).isoformat(),
                duration_s=round(_time.time() - t0, 2), status="error",
                error="xgboost not installed (pip install xgboost)")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={},
                metrics={}, started_at=started,
                finished_at=datetime.now(_tz.utc).isoformat(),
                duration_s=round(_time.time() - t0, 2), status="error", error=str(e))


CONFIG_CYCLES = [
    {"model": "ridge", "target": "pk_t+1", "feature_set": "price"},
    {"model": "ridge", "target": "pk_t+1", "feature_set": "price+news_adv_dual"},
    {"model": "gbm", "target": "pk_t+1", "feature_set": "price"},
    {"model": "gbm", "target": "pk_t+1", "feature_set": "price+news_adv_dual"},
    {"model": "rf", "target": "pk_t+1", "feature_set": "price"},
    {"model": "rf", "target": "pk_t+1", "feature_set": "price+news_adv_dual"},
]


def _run_model_cfg(model_type: str, target: str, feature_set: str) -> ExperimentResult:
    from src.modeling.baseline import FEATURE_SETS, make_pipeline, compute_metrics as base_metrics
    from src.modeling.dataset import SPLIT_DATE, build_panel, time_split

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    try:
        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            panel = build_panel()
            feats = [c for c in FEATURE_SETS[feature_set] if c in panel.columns]
            train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)
            model.fit(train[feats], train[target])
            y_pred = model.predict(test[feats])
        else:
            panel = build_panel()
            feats = [c for c in FEATURE_SETS[feature_set] if c in panel.columns]
            train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
            pipe = make_pipeline(model_type).fit(train[feats], train[target])
            y_pred = pipe.predict(test[feats])
        y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
        metrics = base_metrics(test[target].to_numpy(), y_pred, y_prev)
        return ExperimentResult(
            name=f"{model_type}_{feature_set.replace('+', '_')}", category="model_compare",
            params={"model": model_type, "target": target, "feature_set": feature_set},
            metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
            started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
            duration_s=round(time.time() - t0, 2), status="done")
    except Exception as e:
        return ExperimentResult(
            name=f"{model_type}_{feature_set.replace('+', '_')}", category="model_compare",
            params={"model": model_type, "target": target, "feature_set": feature_set},
            metrics={}, started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
            duration_s=round(time.time() - t0, 2), status="error", error=str(e))


def run_cycle(cycle: int):
    logger.info(f"\n{'='*50}\nCycle {cycle} — {datetime.now(timezone.utc).isoformat()}")
    cfg = CONFIG_CYCLES[cycle % len(CONFIG_CYCLES)]
    logger.info(f"  Config: model={cfg['model']} set={cfg['feature_set']} target={cfg['target']}")
    result = _run_model_cfg(**cfg)
    eid = save_result(
        name=result.name, category=result.category,
        params={**result.params, "agent_cycle": cycle},
        metrics=result.metrics, started_at=result.started_at,
        finished_at=result.finished_at, duration_s=result.duration_s,
        status=result.status, error=result.error,
    )
    logger.info(f"  {result.name}: r2={result.metrics.get('r2', 'N/A'):.6f}  "
                f"dur={result.duration_s:.1f}s  id={eid}  {result.status}")
    try:
        path = generate_comparison_report("model_compare")
        logger.info(f"  Report: {path}")
    except Exception as e:
        logger.warning(f"  Report failed: {e}")
    logger.info(f"Cycle {cycle} done. Sleep {CYCLE_SLEEP_S}s")
    time.sleep(CYCLE_SLEEP_S)


def main():
    logger.info("=" * 50)
    logger.info("AGENT MODEL STARTED")
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
