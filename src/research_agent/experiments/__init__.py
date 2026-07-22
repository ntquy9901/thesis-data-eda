"""Experiment implementations — mỗi file là một nhóm method được so sánh."""

from src.research_agent.experiments.sentiment_methods import (
    VaderSentimentExperiment,
    TextBlobSentimentExperiment,
)
from src.research_agent.experiments.model_compare import (
    RidgeExperiment,
    RFExperiment,
    HGBExperiment,
)
from src.research_agent.experiments.news_features import (
    NewsBasicExperiment,
    NewsAdvDualExperiment,
    NewsAdvFullExperiment,
)

__all__ = [
    "VaderSentimentExperiment",
    "TextBlobSentimentExperiment",
    "RidgeExperiment",
    "RFExperiment",
    "HGBExperiment",
    "NewsBasicExperiment",
    "NewsAdvDualExperiment",
    "NewsAdvFullExperiment",
]
