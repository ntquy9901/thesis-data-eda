"""Story 14-1 — Level 1 sentiment/event scoring (per docs/gpt-guide/news_feature_evaluation_guideline.md).

Reintroduces explicit sentiment scoring for statistical screening AND the modeling panel,
alongside (not instead of) the PhoBERT embeddings kept from Story 11-1 — the guideline's
Level-1 candidate-feature list is: Positive/Negative/Fear/Optimism/Uncertainty score, Event type.

Five per-article scores, one shared methodology (fraction of distinct category keywords
matched, in [0, 1]) so magnitudes are directly comparable across categories:
- ``positive_score`` / ``negative_score`` / ``fear_score`` / ``optimism_score``
- ``uncertainty_score`` — a per-article keyword-count score, DISTINCT from Phase 14's BBD
  ``uncertainty_ratio`` (a market-wide daily ratio requiring Economy AND Policy AND
  Uncertainty terms jointly). The two are not meant to agree; they answer different questions
  (per-article/per-ticker signal here vs. market-wide EPU-style index there).

``event type`` is operationalized as the existing 7 ``TOPIC_CATEGORIES`` keyword flags
(``src.eda.phase04_news_eda``), aggregated as per-(ticker, date) counts — the same taxonomy
already used for topic_*_count in ``src.modeling.features``, but counted market-wide here
(all discovered sources, not just the "tổng hợp" group) since event_type is a market-wide
signal for Level-1/Level-2 evaluation purposes.

Aggregation: per-article scores -> explode by mentioned VN30 ticker -> mean (scores) /
sum (event counts) per (ticker, effective_trading_date) -> reindexed onto the full trading
calendar per ticker (NaN on no-news days, consistent with the project's no-news-NaN rule).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from config import EDA_TICKERS, VN30_TICKERS
from src.data.discover_news import discover_source_files, load_source
from src.eda.phase04_news_eda import TOPIC_CATEGORIES, _trading_calendar, effective_trading_date

# Vietnamese financial-domain keyword lists, one per Level-1 sentiment category. Kept short and
# auditable (simple keyword-fraction scoring, not a trained classifier) — same design tradeoff
# as Phase 14's BBD keyword lists.
POSITIVE_KW = [
    "tăng trưởng", "tăng mạnh", "phát triển", "thành công", "lợi nhuận", "khởi sắc",
    "vượt kỳ vọng", "mạnh mẽ", "thu nhập tốt", "kinh doanh tốt", "doanh thu cao",
    "lãi lớn", "cổ tức cao", "mua vào", "đầu tư mạnh", "giá tăng", "hồi phục",
    "tích cực", "bứt phá", "kỷ lục",
]
NEGATIVE_KW = [
    "giảm mạnh", "thu hẹp", "khó khăn", "sụt giảm", "đóng cửa", "thiệt hại",
    "yếu kém", "bất ổn", "khủng hoảng", "thua lỗ", "giá giảm", "bán ra",
    "cắt lỗ", "nghi ngờ", "tiêu cực", "phá sản", "vỡ nợ", "sa thải", "nợ xấu", "lỗ nặng",
]
FEAR_KW = [
    "hoảng loạn", "tháo chạy", "bán tháo", "sụp đổ", "vỡ nợ", "phá sản",
    "mất trắng", "cảnh báo", "nguy cơ", "đe dọa", "rủi ro cao", "mất kiểm soát", "đổ vỡ",
]
OPTIMISM_KW = [
    "kỳ vọng", "lạc quan", "triển vọng", "cơ hội", "tiềm năng", "hứa hẹn",
    "khả quan", "tin tưởng", "tăng tốc", "đột phá", "vượt trội",
]
# Same category as Phase 14's "U" (Uncertainty) keyword list — intentionally reused so the two
# uncertainty signals differ ONLY in aggregation logic (per-article score vs. tri-category
# market-wide ratio), not in vocabulary.
UNCERTAINTY_KW = ["bất định", "không chắc chắn", "rủi ro", "lo ngại", "bất ổn", "khó lường"]

SENTIMENT_LEXICON: dict[str, list[str]] = {
    "positive": POSITIVE_KW,
    "negative": NEGATIVE_KW,
    "fear": FEAR_KW,
    "optimism": OPTIMISM_KW,
    "uncertainty": UNCERTAINTY_KW,
}
SENTIMENT_SCORE_COLS = [f"{cat}_score" for cat in SENTIMENT_LEXICON]
EVENT_TYPE_COLS = [f"event_{cat}" for cat in TOPIC_CATEGORIES]

TICKER_PATTERN = re.compile(r"\b(" + "|".join(VN30_TICKERS) + r")\b", re.IGNORECASE)


# ---------- pure helpers (unit-tested) ----------
def _normalize(text: str) -> str:
    """NFC-normalize + lowercase before substring matching (mixed NFC/NFD source encoding)."""
    return unicodedata.normalize("NFC", str(text)).lower()


def category_score(text: str, keywords: list[str]) -> float:
    """Fraction of DISTINCT keywords from ``keywords`` present in ``text`` (already normalized), in [0, 1]."""
    if not keywords:
        return 0.0
    matched = sum(1 for kw in keywords if kw in text)
    return matched / len(keywords)


def article_sentiment_scores(text: str) -> dict[str, float]:
    """5 Level-1 sentiment scores for one article's raw text."""
    t = _normalize(text)
    return {f"{cat}_score": round(category_score(t, kws), 4) for cat, kws in SENTIMENT_LEXICON.items()}


def article_event_flags(text: str) -> dict[str, int]:
    """1/0 flag per event-type (TOPIC_CATEGORIES) for one article's raw text."""
    t = _normalize(text)
    return {f"event_{cat}": int(any(kw in t for kw in kws)) for cat, kws in TOPIC_CATEGORIES.items()}


# ---------- article-level + aggregation runners ----------
def _load_all_news() -> pd.DataFrame:
    """All discovered sources (market-wide), title+lead concatenated -> ``_text``."""
    frames = []
    for source, path in discover_source_files().items():
        try:
            frames.append(load_source(source, path))
        except Exception:
            continue
    news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if news.empty:
        return news
    title = news.get("title", pd.Series(index=news.index)).fillna("")
    lead = news.get("lead", pd.Series(index=news.index)).fillna("")
    news["_text"] = (title.astype(str) + " " + lead.astype(str)).str.strip()
    return news[news["_text"].str.len() > 0].reset_index(drop=True)


def _normalize_series(text: pd.Series) -> pd.Series:
    """Vectorized NFC-normalize + lowercase (mixed NFC/NFD source encoding)."""
    return text.astype(str).map(lambda t: unicodedata.normalize("NFC", t)).str.lower()


def _vectorized_lexicon_scores(text_norm: pd.Series, lexicon: dict[str, list[str]]) -> pd.DataFrame:
    """Same fraction-of-keywords-matched scoring as ``category_score``, but as vectorized
    (C-level) ``str.contains`` passes instead of a per-row Python loop — the 1M+-row news
    corpus makes a Python-level ``.apply`` prohibitively slow (O(n_rows x n_keywords) Python
    calls); this is the same O(n_keywords) vectorized passes over the whole column instead."""
    out = {}
    for cat, kws in lexicon.items():
        if not kws:
            out[f"{cat}_score"] = pd.Series(0.0, index=text_norm.index)
            continue
        hits = sum(text_norm.str.contains(kw, regex=False) for kw in kws)
        out[f"{cat}_score"] = (hits / len(kws)).round(4)
    return pd.DataFrame(out)


def _vectorized_event_flags(text_norm: pd.Series) -> pd.DataFrame:
    """Vectorized equivalent of ``article_event_flags`` (see ``_vectorized_lexicon_scores``)."""
    out = {}
    for cat, kws in TOPIC_CATEGORIES.items():
        hit = pd.Series(False, index=text_norm.index)
        for kw in kws:
            hit = hit | text_norm.str.contains(kw, regex=False)
        out[f"event_{cat}"] = hit.astype(int)
    return pd.DataFrame(out)


def build_article_sentiment(news: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-article: 5 sentiment scores + 7 event-type flags + effective_trading_date."""
    if news is None:
        news = _load_all_news()
    if news.empty:
        return pd.DataFrame()
    news = news.reset_index(drop=True)
    trading = _trading_calendar()
    eff_date = effective_trading_date(news["pub_date"], trading)
    text_norm = _normalize_series(news["_text"])
    scores = _vectorized_lexicon_scores(text_norm, SENTIMENT_LEXICON)
    flags = _vectorized_event_flags(text_norm)
    out = pd.concat([news[["_text"]], scores, flags], axis=1)
    out["date"] = eff_date.dt.normalize().to_numpy()
    return out.dropna(subset=["date"])


def _explode_tickers(article_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (article x mentioned VN30 ticker), carrying scores/flags/date."""
    if article_df.empty:
        return pd.DataFrame()
    feature_cols = SENTIMENT_SCORE_COLS + EVENT_TYPE_COLS
    tickers = [TICKER_PATTERN.findall(t) for t in article_df["_text"]]
    records = []
    for i, tks in enumerate(tickers):
        if not tks:
            continue
        row = {c: article_df[c].iloc[i] for c in feature_cols}
        row["date"] = article_df["date"].iloc[i]
        for t in {tk.upper() for tk in tks}:
            records.append({"ticker": t, **row})
    return pd.DataFrame(records)


def build_daily_sentiment_features(ticker_universe: list[str] | None = None) -> pd.DataFrame:
    """(ticker, date) x [5 sentiment score means, 7 event-type counts], full trading calendar
    per ticker (NaN on no-news days, consistent with the project's no-news-NaN rule)."""
    universe = ticker_universe or EDA_TICKERS
    exploded = _explode_tickers(build_article_sentiment())
    if exploded.empty:
        return pd.DataFrame()

    agg_map = dict.fromkeys(SENTIMENT_SCORE_COLS, "mean")
    agg_map.update(dict.fromkeys(EVENT_TYPE_COLS, "sum"))
    daily = exploded.groupby(["ticker", "date"]).agg(agg_map).reset_index()

    td = pd.DatetimeIndex(pd.to_datetime(pd.Series(_trading_calendar())).dt.normalize().sort_values().unique())
    feature_cols = SENTIMENT_SCORE_COLS + EVENT_TYPE_COLS
    out_frames = []
    for ticker in universe:
        sub = daily[daily["ticker"] == ticker]
        if sub.empty:
            df = pd.DataFrame(np.nan, index=td, columns=feature_cols)
        else:
            df = sub.set_index("date").reindex(td)[feature_cols]
        df.insert(0, "ticker", ticker)
        df.index.name = "date"
        out_frames.append(df.reset_index())
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def run() -> list:
    """Populate ``eda_output/modeling/sentiment_features.parquet`` (the modeling-panel artifact)."""
    from src.eda.common import EDA_OUTPUT_DIR, ensure_output_dirs

    ensure_output_dirs()
    outdir = EDA_OUTPUT_DIR / "modeling"
    outdir.mkdir(parents=True, exist_ok=True)
    df = build_daily_sentiment_features()
    if df.empty:
        return []
    out = outdir / "sentiment_features.parquet"
    df.to_parquet(out, index=False)
    return [out]


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
