"""So sánh các phương pháp sentiment analysis trên news data."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from config import EDA_OUTPUT_DIR
from src.research_agent.base import Experiment, ExperimentResult, Registry


@Registry.register
class VaderSentimentExperiment(Experiment):
    name = "vader_sentiment"
    category = "sentiment"
    description = "VADER sentiment on news headlines — tính polarity scores"

    def default_params(self) -> dict:
        return {"model": "VADER", "target": "pk_t+1", "feature_set": "price+sentiment5"}

    def run(self, **kwargs) -> ExperimentResult:
        from src.modeling.baseline import FEATURE_SETS, TARGETS, make_pipeline, compute_metrics as base_metrics
        from src.modeling.dataset import SPLIT_DATE, build_panel, time_split

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            panel = build_panel()
            feats = [c for c in FEATURE_SETS["price+sentiment5"] if c in panel.columns]
            target = kwargs.get("target", "pk_t+1")
            train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
            pipe = make_pipeline("ridge").fit(train[feats], train[target])
            y_pred = pipe.predict(test[feats])
            y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
            metrics = base_metrics(test[target].to_numpy(), y_pred, y_prev)

            return ExperimentResult(
                name=self.name, category=self.category,
                params={"target": target, "n_features": len(feats)},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={},
                metrics={}, started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))


@Registry.register
class TextBlobSentimentExperiment(Experiment):
    name = "textblob_sentiment"
    category = "sentiment"
    description = "TextBlob sentiment — baseline lexicon approach"

    def default_params(self) -> dict:
        return {"model": "TextBlob", "target": "pk_t+1"}

    def run(self, **kwargs) -> ExperimentResult:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        from src.modeling.dataset import SPLIT_DATE, build_panel, time_split
        from src.modeling.baseline import FEATURE_SETS, TARGETS, make_pipeline, compute_metrics as base_metrics

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        try:
            try:
                nltk.data.find("sentiment/vader_lexicon")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()

            source_path = EDA_OUTPUT_DIR / "news" / "sparse_news_features.parquet"
            if not source_path.exists():
                raise FileNotFoundError(f"{source_path} not found")
            df = pd.read_parquet(source_path)
            df["title"] = df.get("title", "")
            df["tb_neg"] = df["title"].apply(lambda t: sia.polarity_scores(str(t))["neg"])
            df["tb_pos"] = df["title"].apply(lambda t: sia.polarity_scores(str(t))["pos"])
            df["tb_compound"] = df["title"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

            panel = build_panel()
            tb_cols = ["tb_neg", "tb_pos", "tb_compound"]
            panel = panel.merge(
                df[["ticker", "trading_date", *tb_cols]],
                left_on=["ticker", "date"], right_on=["ticker", "trading_date"], how="left"
            )
            target = kwargs.get("target", "pk_t+1")
            price_feats = ["har_daily", "har_weekly", "har_monthly", "atr_14", "realized_vol_5d", "realized_vol_20d"]
            feats = price_feats + [c for c in tb_cols if c in panel.columns]
            train, test = time_split(panel.dropna(subset=[target]).copy(), SPLIT_DATE)
            pipe = make_pipeline("ridge").fit(train[feats], train[target])
            y_pred = pipe.predict(test[feats])
            y_prev = test.groupby("ticker")[target].shift(1).to_numpy()
            metrics = base_metrics(test[target].to_numpy(), y_pred, y_prev)
            return ExperimentResult(
                name=self.name, category=self.category,
                params={"target": target, "n_features": len(feats)},
                metrics={k: float(v) if v is not None else float("nan") for k, v in metrics.items()},
                started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="done")
        except Exception as e:
            return ExperimentResult(
                name=self.name, category=self.category, params={},
                metrics={}, started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_s=round(time.time() - t0, 2), status="error", error=str(e))
