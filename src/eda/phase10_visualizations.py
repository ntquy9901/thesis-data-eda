"""Phase 10 — Visualizations (per EDA Guide).

Produces the 11 required charts. Many are emitted by earlier phases (rolling
volatility, cross-ticker correlation heatmap, ACF/PACF, event-study plot,
cross-correlation); this phase fills the gaps and writes a charts index.

Outputs (under respective ``eda_output/<phase>/`` dirs + ``report/charts_index.md``):
- missing_heatmap.png (quality)
- news_coverage_by_stock.png, news_count_by_day.png, sentiment_distribution.png (news)
- return_distribution.png, volatility_distribution.png (price)
- news_count_vs_future_vol.png (relationship)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import EDA_TICKERS
from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs, phase_output_dir


def _save_bar(series: pd.Series, path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    series.plot.bar(ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_hist(series: pd.Series, path: Path, title: str, bins: int = 50) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    series.dropna().hist(bins=bins, ax=ax, color="darkorange", edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_scatter(x: pd.Series, y: pd.Series, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=6, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_missing_heatmap(path: Path) -> None:
    import matplotlib.pyplot as plt

    miss = EDA_OUTPUT_DIR / "quality" / "missingness_report.csv"
    if not miss.exists():
        return
    df = pd.read_csv(miss)
    pivot = df.pivot_table(index="table", columns="column", values="pct", aggfunc="first")
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(pivot) + 2))
    im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="Reds")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="% missing")
    ax.set_title("Missing value heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _load_returns_vol() -> pd.DataFrame:
    frames = []
    for ticker in EDA_TICKERS:
        pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)[["log_returns", "realized_vol_20d"]].assign(ticker=ticker)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_news_count_by_day() -> pd.Series | None:
    pq = EDA_OUTPUT_DIR / "news" / "sparse_news_features.parquet"
    if not pq.exists():
        return None
    news = pd.read_parquet(pq)
    news["trading_date"] = pd.to_datetime(news["trading_date"])
    return news.groupby("trading_date")["news_count_1d"].sum()


def run_phase() -> list[Path]:
    from src.eda.common import configure_plots

    ensure_output_dirs()
    configure_plots()
    written: list[Path] = []
    chart_index = ["# Charts index (Phase 10)\n"]

    # 1. Missing heatmap
    p = phase_output_dir("quality") / "missing_heatmap.png"
    _save_missing_heatmap(p)
    if p.exists():
        written.append(p)
        chart_index.append("1. missing_heatmap.png (quality/)")

    # 2. News coverage by stock
    nps = EDA_OUTPUT_DIR / "news" / "news_per_stock.csv"
    if nps.exists():
        df = pd.read_csv(nps).set_index("ticker")["news_count"].sort_values(ascending=False)
        p = phase_output_dir("news") / "news_coverage_by_stock.png"
        _save_bar(df, p, "News coverage by stock", "article count")
        written.append(p)
        chart_index.append("2. news_coverage_by_stock.png (news/)")

    # 3. News count by day
    ncd = _load_news_count_by_day()
    if ncd is not None and not ncd.empty:
        p = phase_output_dir("news") / "news_count_by_day.png"
        _save_bar(ncd.tail(120), p, "News count by day (last 120 trading days)", "articles")
        written.append(p)
        chart_index.append("3. news_count_by_day.png (news/)")

    # 4. Sentiment distribution
    sparse = EDA_OUTPUT_DIR / "news" / "sparse_news_features.parquet"
    if sparse.exists():
        sent = pd.read_parquet(sparse)["sentiment_mean"].dropna()
        if not sent.empty:
            p = phase_output_dir("news") / "sentiment_distribution.png"
            _save_hist(sent, p, "Sentiment distribution")
            written.append(p)
            chart_index.append("4. sentiment_distribution.png (news/)")

    # 5 + 6. Return + volatility distribution
    rv = _load_returns_vol()
    if not rv.empty:
        p5 = phase_output_dir("price") / "return_distribution.png"
        _save_hist(rv["log_returns"], p5, "Log-return distribution")
        written.append(p5)
        chart_index.append("5. return_distribution.png (price/)")
        p6 = phase_output_dir("price") / "volatility_distribution.png"
        _save_hist(rv["realized_vol_20d"], p6, "Realized volatility (20d) distribution")
        written.append(p6)
        chart_index.append("6. volatility_distribution.png (price/)")

    # 7 rolling_vol, 8 corr_heatmap, 9 event_study_plot, 8b cross_corr — already in their phase dirs
    chart_index.append("7. rolling_vol.png (price/, phase 3)")
    chart_index.append("8. corr_heatmap.png (price/, phase 3) + cross_corr.png (relationship/, phase 5)")
    chart_index.append("9. event_study_plot.png + acf_pacf_<ticker>.png (phase 3/6)")

    # 10. News count vs future volatility
    if sparse.exists() and not rv.empty:
        news = pd.read_parquet(sparse)
        # join with a target (pk_t+5) per ticker/date
        tgt_frames = []
        for ticker in EDA_TICKERS:
            pq = EDA_OUTPUT_DIR / "price" / f"price_metrics_{ticker}.parquet"
            if pq.exists():
                d = pd.read_parquet(pq)[["date", "pk_t+5"]].assign(ticker=ticker)
                tgt_frames.append(d)
        if tgt_frames:
            tgt = pd.concat(tgt_frames, ignore_index=True)
            tgt["date"] = pd.to_datetime(tgt["date"]).dt.normalize()
            news["trading_date"] = pd.to_datetime(news["trading_date"]).dt.normalize()
            joined = news.merge(tgt, left_on=["ticker", "trading_date"], right_on=["ticker", "date"], how="inner")
            if {"news_count_1d", "pk_t+5"} <= set(joined.columns):
                p = phase_output_dir("relationship") / "news_count_vs_future_vol.png"
                _save_scatter(joined["news_count_1d"], joined["pk_t+5"], p,
                              "News count (1d) vs future Parkinson vol (t+5)",
                              "news_count_1d", "pk_t+5")
                written.append(p)
                chart_index.append("10. news_count_vs_future_vol.png (relationship/)")

    # 11. SHAP/importance — N/A (no model trained in EDA)
    chart_index.append("11. SHAP/feature-importance — N/A (no model; deferred to modeling)")

    idx = phase_output_dir("report") / "charts_index.md"
    idx.write_text("\n".join(chart_index), encoding="utf-8")
    written.append(idx)
    return written


if __name__ == "__main__":  # pragma: no cover
    for pth in run_phase():
        print(f"Wrote {pth}")
