"""Phase 4 — News EDA (per EDA Guide).

Coverage (news/day, news/stock, days without news), publish-time distribution
(before/during/after market, lunch, weekend), ``effective_trading_date`` mapping,
sentiment summaries, topic extraction (mapped to 7 categories), source distribution.

Outputs (under ``eda_output/news/``):
- ``coverage_report.csv``, ``publish_time.json``, ``publish_time.png``
- ``news_per_stock.csv``, ``sentiment_summary.json``, ``topics.json``, ``source_report.json``

Per-source date normalization: ssi + vndirect are DD/MM/YYYY; cafef + the
consolidated news_articles.csv are ISO 8601 (verified against real data).
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

from config import EDA_TICKERS, PRICE_DATA_DIR
from src.eda.common import ensure_output_dirs, phase_output_dir
from src.eda.phase01_profiling import NEWS_FILES

logger = logging.getLogger(__name__)

# DD/MM/YYYY sources (verified on real data); cafef + news_articles are ISO.
SOURCE_DAYFIRST = {"ssi_articles": True, "vndirect_articles": True}
MARKET_CLOSE_HOUR = 15  # VN afternoon close, UTC+7
LUNCH_START_HOUR = 11  # lunch break ~11:30–13:00
LUNCH_END_HOUR = 13
VN_TZ = "Asia/Ho_Chi_Minh"

# Minimal Vietnamese stopword list (function words that pollute NMF topics).
VN_STOPWORDS = {
    "của", "và", "là", "trong", "với", "cho", "một", "các", "được", "có",
    "không", "đã", "sẽ", "về", "từ", "theo", "này", "đó", "cũng", "như",
    "nao", "sau", "trước", "khi", "nếu", "nhưng", "vì", "để", "tới", "being",
    "the", "to", "in", "of", "and", "for", "on", "with", "yoy", "we", "our",
}

# Topic → 7 EDA-Guide categories, by Vietnamese/English keyword.
TOPIC_CATEGORIES = {
    "earnings": ["lợi nhuận", "doanh thu", "kết quả kinh doanh", "quý", "earnings", "ebitda"],
    "dividend": ["cổ tức", "chia thưởng", "dividend"],
    "ma": ["mua sáp nhập", "sáp nhập", "mua lại", "merger", "acquisition"],
    "management": ["ban lãnh đạo", "tổng giám đốc", "ceo", "management", "bổ nhiệm"],
    "regulation": ["quy định", "pháp lý", "kiểm soát", "regulation", "ubck"],
    "macro": ["lạm phát", "lãi suất", "gdp", "cpi", "fed", "macro", "ngân hàng nhà nước"],
    "sector": ["ngành", "sector", "chuỗi cung ứng", "hàng không", "ngân hàng", "bất động sản"],
}


# ---------- pure helpers (unit-tested) ----------
def normalize_pub_date(series: pd.Series, source: str) -> pd.Series:
    """Parse pub_date with per-source dayfirst; result tz-aware UTC (NaT on failure)."""
    if series is None:
        return pd.Series(pd.NaT, index=pd.RangeIndex(0))
    dayfirst = SOURCE_DAYFIRST.get(source, False)
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=True)


def effective_trading_date(news_dt: pd.Series, trading_dates, close_hour: int = MARKET_CLOSE_HOUR) -> pd.Series:
    """Map each news datetime → its effective trading date.

    News before the market close on a trading day → same day; after close or on a
    non-trading day → rolled forward to the next available trading date.
    **NaT inputs stay NaT** (never silently mapped to the calendar tail).
    Returns tz-naive date series.
    """
    td = pd.to_datetime(pd.Series(trading_dates)).dt.tz_localize(None).dt.normalize().sort_values().drop_duplicates()
    td_arr = td.values.astype("datetime64[D]")
    if len(td_arr) == 0:
        return pd.Series(pd.NaT, index=news_dt.index)

    local = pd.to_datetime(news_dt, errors="coerce", utc=True).dt.tz_convert(VN_TZ)
    after_close = local.dt.hour >= close_hour
    eff_day = local.dt.normalize()
    eff_day = eff_day.where(~after_close, eff_day + pd.Timedelta(days=1))
    eff_arr = eff_day.dt.tz_localize(None).values.astype("datetime64[D]")

    idx = np_searchsorted(td_arr, eff_arr)
    idx = np.clip(idx, 0, len(td_arr) - 1)  # tail news → last trading date
    result = pd.Series(pd.to_datetime(td_arr[idx]).normalize(), index=news_dt.index)
    return result.where(local.notna().values, pd.NaT)  # propagate NaT


def np_searchsorted(calendar, values):
    """First index in sorted ``calendar`` >= each value. Pure (testable)."""
    return np.searchsorted(calendar, values, side="left")


def classify_publish_time(news_dt: pd.Series, close_hour: int = MARKET_CLOSE_HOUR) -> pd.DataFrame:
    """Bucket news by market session + weekend. NaT → ``unparsed``. Pure."""
    local = pd.to_datetime(news_dt, errors="coerce", utc=True).dt.tz_convert(VN_TZ)
    hour = local.dt.hour
    session = pd.Series("unparsed", index=news_dt.index)
    valid = local.notna()
    during = valid & (hour >= 9) & (hour < LUNCH_START_HOUR + 0.5)  # 9:00–11:30 morning
    afternoon = valid & (hour >= LUNCH_END_HOUR) & (hour < close_hour)  # 13:00–15:00
    session[during | afternoon] = "during_market"
    session[valid & (hour < 9)] = "before_market"
    session[valid & (hour >= close_hour)] = "after_market"
    lunch = valid & (hour >= LUNCH_START_HOUR + 0.5) & (hour < LUNCH_END_HOUR)
    session[lunch] = "lunch_break"
    weekend = pd.Series(local.dt.dayofweek >= 5, index=news_dt.index).fillna(False)
    return pd.DataFrame({"session": session, "is_weekend": weekend})


# ---------- phase runner ----------
def _load_news_frame() -> pd.DataFrame:
    frames = []
    for name, path in NEWS_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        df["_source_file"] = name
        if "article_url" in df.columns and "url" not in df.columns:
            df = df.rename(columns={"article_url": "url"})
        if "section" in df.columns and "category" not in df.columns:
            df = df.rename(columns={"section": "category"})
        df["pub_date_dt"] = normalize_pub_date(df.get("pub_date"), name)
        title = df.get("title", pd.Series(index=df.index)).fillna("")
        lead = df.get("lead", pd.Series(index=df.index)).fillna("")
        df["_text"] = (title.astype(str) + " " + lead.astype(str))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _trading_calendar() -> pd.Series:
    dates = set()
    for ticker in EDA_TICKERS:
        p = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if p.exists():
            dates.update(pd.to_datetime(pd.read_csv(p, encoding="utf-8")["date"], errors="coerce").dropna())
    return pd.Series(sorted(dates))


def _coverage_report(news: pd.DataFrame, trading_dates) -> pd.DataFrame:
    if news.empty or "pub_date_dt" not in news:
        return pd.DataFrame(columns=["metric", "value"])
    td = pd.to_datetime(pd.Series(trading_dates)).dt.normalize()
    valid = news.dropna(subset=["pub_date_dt"])
    n_news = len(valid)
    n_days_with_news = valid["pub_date_dt"].dt.tz_localize(None).dt.normalize().nunique()
    n_trading_days = len(td)
    news_on_trading = valid["pub_date_dt"].dt.tz_localize(None).dt.normalize().isin(td).sum()
    rows = [
        {"metric": "total_articles", "value": n_news},
        {"metric": "avg_news_per_trading_day", "value": round(n_news / max(n_trading_days, 1), 3)},
        {"metric": "days_with_news", "value": int(n_days_with_news)},
        {"metric": "trading_days", "value": int(n_trading_days)},
        {"metric": "news_on_trading_days", "value": int(news_on_trading)},
        {"metric": "trading_days_without_news", "value": int(max(n_trading_days - news_on_trading, 0))},
    ]
    return pd.DataFrame(rows)


def _news_per_stock(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["ticker", "news_count"])
    try:
        from src.sprint1.task1_3_vietnamese_nlp import VietnameseNLPProcessor

        proc = VietnameseNLPProcessor()
        tickers = proc.extract_stock_tickers(news["_text"].tolist())
    except Exception as e:  # NLP unavailable → regex fallback
        logger.warning("VietnameseNLPProcessor unavailable (%s); using regex", e)
        import re

        from config import VN30_TICKERS

        pat = re.compile(r"\b(" + "|".join(VN30_TICKERS) + r")\b")
        tickers = [pat.findall(t) for t in news["_text"].tolist()]
    counts = Counter(t for row in tickers for t in row)
    return pd.DataFrame([{"ticker": k, "news_count": v} for k, v in counts.most_common()])


def _sentiment_summary(news: pd.DataFrame) -> dict:
    if news.empty:
        return {}
    try:
        from src.sprint1.task1_3_vietnamese_nlp import VietnameseNLPProcessor

        proc = VietnameseNLPProcessor()
        results = proc.calculate_sentiment_vietnamese(news["_text"].tolist())
        scores = pd.Series([r.get("score", 0.0) for r in results], dtype=float)  # key is "score"
    except Exception as e:
        logger.warning("sentiment unavailable: %s", e)
        return {"note": "VietnameseNLPProcessor unavailable; sentiment skipped"}
    pos = int((scores > 0).sum())
    neg = int((scores < 0).sum())
    neu = int((scores == 0).sum())
    return {
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std(ddof=0)), 4),  # population std → finite for n=1
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "positive_ratio": round(pos / max(len(scores), 1), 4),
        "negative_ratio": round(neg / max(len(scores), 1), 4),
    }


def _map_topic_category(top_terms: list[str]) -> str | None:
    """Heuristic-map a topic's top terms to one of the 7 EDA-Guide categories."""
    joined = " ".join(top_terms).lower()
    for cat, keywords in TOPIC_CATEGORIES.items():
        if any(kw in joined for kw in keywords):
            return cat
    return None


def _topics(news: pd.DataFrame, n_topics: int = 7, top_words: int = 8) -> dict:
    if news.empty or "_text" not in news:
        return {}
    try:
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        logger.warning("scikit-learn unavailable: %s", e)
        return {"note": "scikit-learn unavailable; topics skipped"}
    texts = news["_text"].astype(str).replace("", " ").tolist()
    if len(texts) < n_topics * 5:
        return {"note": "insufficient documents for topic modeling"}
    try:
        vec = TfidfVectorizer(max_features=2000, stop_words=list(VN_STOPWORDS))
        tfidf = vec.fit_transform(texts)
        if tfidf.nnz == 0:
            return {"note": "empty vocabulary; topics skipped"}
    except ValueError as e:
        return {"note": f"vectorization failed: {e}"}
    terms = vec.get_feature_names_out()
    nmf = NMF(n_components=n_topics, random_state=0, max_iter=300)
    W = nmf.fit_transform(tfidf)
    topics = {}
    for i, comp in enumerate(nmf.components_):
        top = [terms[j] for j in comp.argsort()[-top_words:][::-1]]
        topics[f"topic_{i+1}"] = {
            "top_terms": top,
            "category": _map_topic_category(top),
            "doc_count": int((W[:, i] > 0.01).sum()),
        }
    return topics


def _source_report(news: pd.DataFrame) -> dict:
    if news.empty or "_source_file" not in news:
        return {}
    counts = news["_source_file"].value_counts().to_dict()
    dup_url = int(news.dropna(subset=["url"]).duplicated(subset=["url"]).sum()) if "url" in news else 0
    return {"source_counts": counts, "duplicate_urls": dup_url, "repost_rate": round(dup_url / max(len(news), 1), 4)}


def _plot_publish_time(pt: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt

    counts = pt["session"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color="steelblue")
    ax.set_title("News publish time by market session")
    ax.set_ylabel("article count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_phase() -> list:
    from src.eda.common import configure_plots

    ensure_output_dirs()
    configure_plots()
    outdir = phase_output_dir("news")
    news = _load_news_frame()
    trading = _trading_calendar()
    written = []

    cov = _coverage_report(news, trading)
    cov.to_csv(outdir / "coverage_report.csv", index=False, encoding="utf-8")
    written.append(outdir / "coverage_report.csv")

    if not news.empty:
        eff = effective_trading_date(news["pub_date_dt"], trading)
        # alignment = fraction of news with a parseable date that maps to a trading day
        # (NaT news — unparseable date — are NOT counted, so this is not vacuously 1.0)
        alignment = round(float(eff.notna().sum() / max(len(news), 1)), 4)
    else:
        alignment = 0.0

    pt = classify_publish_time(news["pub_date_dt"]) if not news.empty else pd.DataFrame()
    pt_stats = {
        "session_counts": pt["session"].value_counts().to_dict() if not pt.empty else {},
        "weekend_count": int(pt["is_weekend"].sum()) if not pt.empty else 0,
        "effective_trading_date_alignment": alignment,
        "note": "alignment = fraction of news with a parseable date mapping to a trading day",
    }
    (outdir / "publish_time.json").write_text(json.dumps(pt_stats, indent=2), encoding="utf-8")
    written.append(outdir / "publish_time.json")
    if not pt.empty:
        _plot_publish_time(pt, outdir / "publish_time.png")
        written.append(outdir / "publish_time.png")

    nps = _news_per_stock(news)
    nps.to_csv(outdir / "news_per_stock.csv", index=False, encoding="utf-8")
    written.append(outdir / "news_per_stock.csv")

    (outdir / "sentiment_summary.json").write_text(json.dumps(_sentiment_summary(news), indent=2), encoding="utf-8")
    written.append(outdir / "sentiment_summary.json")
    (outdir / "topics.json").write_text(json.dumps(_topics(news), indent=2), encoding="utf-8")
    written.append(outdir / "topics.json")
    (outdir / "source_report.json").write_text(json.dumps(_source_report(news), indent=2), encoding="utf-8")
    written.append(outdir / "source_report.json")
    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
