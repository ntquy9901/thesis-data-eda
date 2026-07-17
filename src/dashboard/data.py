"""Dashboard data layer — typed, cached loaders for every ``eda_output/`` artifact.

Pure functions (no Streamlit import) so they are unit-testable. The Streamlit
app composes them; tests call them directly on real or temp artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.common import EDA_OUTPUT_DIR

MODELING = "modeling"


def _p(base: Path, *parts: str) -> Path:
    return Path(base).joinpath(*parts)


def load_panel(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, MODELING, "panel.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_metrics(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, MODELING, "metrics.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_significance(base: Path = EDA_OUTPUT_DIR) -> dict:
    p = _p(base, MODELING, "significance.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_price_metrics(ticker: str, base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "price", f"price_metrics_{ticker}.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_sparse_news(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "news", "sparse_news_features.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_advanced_news(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, MODELING, "advanced_news_features.parquet")
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_event_study(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "relationship", "event_study.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_news_embedding_source_stats(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "news_embedding", "source_stats.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_news_embedding_coverage(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "news_embedding", "embedding_coverage.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_embedding_price_corr(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "news_embedding", "embedding_price_corr.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_extended_horizon_corr(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    p = _p(base, "news_embedding", "extended_horizon_corr.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_articles_list(group: str, source: str | None = None, limit: int | None = 200) -> pd.DataFrame:
    """Raw article rows (title/lead/source/pub_date/url) for the dashboard's list-view page —
    reads directly from crawl_data (NOT an eda_output artifact), for manual spot-check reading.
    ``group``: "khach_quan" or "tong_hop" — see ``src.features.news_embeddings.GROUP_SOURCES``
    for the canonical (dynamically-discovered) source classification. If ``source`` is given,
    restricts to that single source; otherwise loads every discovered source in the group."""
    from src.data.discover_news import discover_source_files, load_source
    from src.features.news_embeddings import GROUP_SOURCES

    wanted = {source} if source else GROUP_SOURCES.get(group, set())
    frames = []
    for s, path in discover_source_files().items():
        if s not in wanted:
            continue
        try:
            frames.append(load_source(s, path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    cols = [c for c in ["source", "pub_date", "title", "lead", "url", "category"] if c in df.columns]
    out = df[cols].copy()
    if "pub_date" in out.columns:
        out = out.sort_values("pub_date", ascending=False)
    return out.head(limit) if limit else out


def load_json(name: str, base: Path = EDA_OUTPUT_DIR) -> dict:
    """Load a JSON artifact by relative path, e.g. 'news/sentiment_summary.json'."""
    p = _p(base, *name.split("/"))
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_text(name: str, base: Path = EDA_OUTPUT_DIR) -> str:
    p = _p(base, *name.split("/"))
    return p.read_text(encoding="utf-8") if p.exists() else ""


def available_tickers(base: Path = EDA_OUTPUT_DIR) -> list[str]:
    """Tickers that have a price_metrics parquet."""
    d = _p(base, "price")
    if not d.exists():
        return []
    out = sorted(p.stem.replace("price_metrics_", "") for p in d.glob("price_metrics_*.parquet"))
    return out


def headline_metrics(base: Path = EDA_OUTPUT_DIR) -> dict:
    """One-row-per-target best-model summary for the overview."""
    m = load_metrics(base)
    sig = load_significance(base)
    out = {}
    if m.empty:
        return out
    for target in m["target"].unique():
        sub = m[(m.target == target) & (m.model == "ridge") & (m.feature_set == "price")]
        if sub.empty:
            continue
        row = sub.iloc[0]
        dm = sig.get("per_target", {}).get(target, {}).get("dm", {})
        out[target] = {
            "r2_price": row["r2"],
            "rmse_price": row["rmse"],
            "dir_acc": row["dir_acc"],
            "dm_pvalue": dm.get("dm_pvalue"),
        }
    return out


def load_novelty_price_corr(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    """Phase 13: Novelty-based correlation with price."""
    p = _p(base, "news_embedding", "novelty_price_corr.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_uncertainty_price_corr(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    """Phase 14: Uncertainty index correlation with price."""
    p = _p(base, "uncertainty", "uncertainty_price_corr.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_decay_price_corr(base: Path = EDA_OUTPUT_DIR) -> pd.DataFrame:
    """Phase 15: Temporal decay of embedding signal."""
    p = _p(base, "news_embedding", "decay_price_corr.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
