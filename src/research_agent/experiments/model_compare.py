"""So sánh các model architectures (Ridge vs RF vs HGB) trên cùng feature set."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.research_agent.base import Experiment, ExperimentResult, Registry


def _run_model(model_type: str, target: str = "pk_t+1", feature_set: str = "price+news_adv_dual"):
    from src.modeling.baseline import FEATURE_SETS, make_pipeline, compute_metrics as base_metrics
    from src.modeling.dataset import SPLIT_DATE, build_panel, time_split

    panel = build_panel()
    feats = [c for c in FEATURE_SETS[feature_set] if c in panel.columns]
    train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
    pipe = make_pipeline(model_type).fit(train[feats], train[target])
    y_pred = pipe.predict(test[feats])
    y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
    return base_metrics(test[target].to_numpy(), y_pred, y_prev)


@Registry.register
class RidgeExperiment(Experiment):
    name = "ridge_linear"
    category = "model_compare"
    description = "Ridge regression (linear HAR) — baseline model"

    def run(self, **kwargs) -> ExperimentResult:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            metrics = _run_model("ridge", kwargs.get("target", "pk_t+1"), kwargs.get("feature_set", "price+news_adv_dual"))
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"model": "Ridge", "target": kwargs.get("target", "pk_t+1"),
                        "feature_set": kwargs.get("feature_set", "price+news_adv_dual")},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={}, metrics={},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))


@Registry.register
class RFExperiment(Experiment):
    name = "random_forest"
    category = "model_compare"
    description = "Random Forest (200 trees, max_depth=10) — nonlinear baseline"

    def run(self, **kwargs) -> ExperimentResult:
        from sklearn.ensemble import RandomForestRegressor
        from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
        from src.modeling.baseline import FEATURE_SETS, compute_metrics as base_metrics

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            target = kwargs.get("target", "pk_t+1")
            fset = kwargs.get("feature_set", "price+news_adv_dual")
            panel = build_panel()
            feats = [c for c in FEATURE_SETS[fset] if c in panel.columns]
            train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
            rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0).fit(train[feats], train[target])
            y_pred = rf.predict(test[feats])
            y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
            metrics = base_metrics(test[target].to_numpy(), y_pred, y_prev)
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"model": "RF", "target": target, "feature_set": fset},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={}, metrics={},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))


@Registry.register
class HGBExperiment(Experiment):
    name = "hist_gradient_boost"
    category = "model_compare"
    description = "HistGradientBoosting (200 iter, max_depth=4) — GBM baseline"

    def run(self, **kwargs) -> ExperimentResult:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            metrics = _run_model("gbm", kwargs.get("target", "pk_t+1"), kwargs.get("feature_set", "price+news_adv_dual"))
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"model": "HGB", "target": kwargs.get("target", "pk_t+1"),
                        "feature_set": kwargs.get("feature_set", "price+news_adv_dual")},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={}, metrics={},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))
