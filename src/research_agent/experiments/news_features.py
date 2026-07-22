"""So sánh các feature groups — basic news vs dual embeddings vs full set."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.research_agent.base import Experiment, ExperimentResult, Registry


def _run_fset(fset_name: str, target: str = "pk_t+1", model: str = "ridge"):
    from src.modeling.baseline import FEATURE_SETS, make_pipeline, compute_metrics as base_metrics
    from src.modeling.dataset import SPLIT_DATE, build_panel, time_split

    panel = build_panel()
    feats = [c for c in FEATURE_SETS[fset_name] if c in panel.columns]
    train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
    pipe = make_pipeline(model).fit(train[feats], train[target])
    y_pred = pipe.predict(test[feats])
    y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
    return base_metrics(test[target].to_numpy(), y_pred, y_prev), len(feats)


@Registry.register
class NewsBasicExperiment(Experiment):
    name = "news_basic"
    category = "feature_compare"
    description = "Price + basic news features (counts + sentiment_mean)"

    def run(self, **kwargs) -> ExperimentResult:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            metrics, nf = _run_fset("price+news_basic", kwargs.get("target", "pk_t+1"), kwargs.get("model", "ridge"))
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"feature_set": "price+news_basic", "n_features": nf,
                        "target": kwargs.get("target", "pk_t+1"), "model": kwargs.get("model", "ridge")},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={}, metrics={},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))


@Registry.register
class NewsAdvDualExperiment(Experiment):
    name = "news_adv_dual"
    category = "feature_compare"
    description = "Price + basic + dual-group embeddings (kq+th, 80 dims)"

    def run(self, **kwargs) -> ExperimentResult:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            metrics, nf = _run_fset("price+news_adv_dual", kwargs.get("target", "pk_t+1"), kwargs.get("model", "ridge"))
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"feature_set": "price+news_adv_dual", "n_features": nf,
                        "target": kwargs.get("target", "pk_t+1"), "model": kwargs.get("model", "ridge")},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={}, metrics={},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))


@Registry.register
class NewsAdvFullExperiment(Experiment):
    name = "news_adv_full"
    category = "feature_compare"
    description = "Price + basic + dual + EWMA + novelty + dispersion + shock (493 feats)"

    def run(self, **kwargs) -> ExperimentResult:
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            metrics, nf = _run_fset("price+news_adv_full", kwargs.get("target", "pk_t+1"), kwargs.get("model", "ridge"))
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"feature_set": "price+news_adv_full", "n_features": nf,
                        "target": kwargs.get("target", "pk_t+1"), "model": kwargs.get("model", "ridge")},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={}, metrics={},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))
