"""Tests for src.modeling.features (Story 8-2 / 11-1 — embedding features, no sentiment)."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import features as F
from src.modeling.features import EMB_FEATURES, aggregate_articles, topic_flags


@pytest.fixture(autouse=True)
def _fake_phobert(monkeypatch, tmp_path):
    """Avoid real PhoBERT download/inference in tests (random vectors, real data shape) AND
    redirect the embedding cache to tmp_path — a mocked run must NEVER write fake vectors into
    the real data/features/news_emb_*.parquet consumed by production runs."""
    import src.features.news_embeddings as ne

    def _fake_extract(texts, **kwargs):
        return np.random.default_rng(0).normal(size=(len(texts), 768)).astype(np.float32)

    monkeypatch.setattr(ne, "extract_phobert_embeddings", _fake_extract)
    monkeypatch.setattr(ne, "FEATURES_DIR", tmp_path / "features")


def test_topic_flags_matches_category():
    f = topic_flags("Lợi nhuận quý của VCB tăng mạnh")
    assert f["topic_earnings_count"] == 1
    assert f["topic_dividend_count"] == 0


def test_topic_flags_macro():
    f = topic_flags("Fed tăng lãi suất, lạm phát giảm")
    assert f["topic_macro_count"] == 1


def test_aggregate_articles_basic():
    rows = pd.DataFrame({
        "emb_0": [1.0, 2.0, 3.0],
        "emb_1": [0.5, 0.5, 0.5],
        "topic_earnings_count": [1, 0, 1],
        "topic_dividend_count": [0, 0, 0],
        "topic_ma_count": [0, 0, 0],
        "topic_management_count": [0, 0, 0],
        "topic_regulation_count": [0, 0, 0],
        "topic_macro_count": [0, 1, 0],
        "topic_sector_count": [0, 0, 0],
    })
    a = aggregate_articles(rows)
    assert a["emb_0"] == pytest.approx(2.0)
    assert a["emb_1"] == pytest.approx(0.5)
    assert np.isnan(a["emb_2"])  # rows fixture only has 2 dims (small-PCA case)
    assert a["topic_earnings_count"] == 2 and a["topic_macro_count"] == 1


def test_aggregate_articles_empty_is_nan():
    a = aggregate_articles(pd.DataFrame())
    assert np.isnan(a[EMB_FEATURES[0]])


def test_real_build_advanced_features_smoke():
    df = F.build_advanced_features()
    if df.empty:
        pytest.skip("no news data")
    assert "ticker" in df.columns and "date" in df.columns
    assert set(EMB_FEATURES) <= set(df.columns)
    # no-news rows must be NaN (not 0) for embedding features
    nn = df[df[EMB_FEATURES[0]].isna()]
    assert nn[EMB_FEATURES[1]].isna().all()


def test_real_run_writes_parquet_smoke():
    written = F.run()
    if not written:
        pytest.skip("no news")
    assert written[0].name == "advanced_news_features.parquet"
