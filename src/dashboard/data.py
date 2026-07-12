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
