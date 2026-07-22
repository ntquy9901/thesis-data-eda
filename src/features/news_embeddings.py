"""Story 11-1 — PhoBERT embedding features for news, replacing sentiment.

Two MUTUALLY EXCLUSIVE groups, split by source identity (redefined after user feedback —
the original file-based split had "tổng hợp" as the literal union of all 5 sources, so every
"khách quan" article trivially also appeared in "tổng hợp"; this partition has no overlap):
- "khach_quan" (objective/factual reporting): cafef, hsc, vnexpress, thanhnien, tuoitre, nld,
  vietnamplus — general-press financial/mainstream news portals.
- "tong_hop" (aggregated/analyst commentary): ssi, vndirect, vnstock, vietstock, vsdc —
  securities firms' own research/market-wrap content, which synthesizes/comments rather than
  reports directly.

Sources are DISCOVERED dynamically (``src.data.discover_news.discover_source_files``) rather
than hardcoded by filename — new crawl files dropped into ``crawl_data/data`` (including its
``objective/`` subdirectory) are picked up automatically. A source not in either list above is
"unclassified": excluded from both groups but surfaced in Phase-11's ``source_stats.csv`` so a
human notices and can add it to the classification.

Design: the EXPENSIVE step (PhoBERT [CLS] encoding) is cached INCREMENTALLY at the
article level, keyed by ``url`` (data/features/news_emb_articles_{group}.parquet).
On each run, only articles whose ``url`` is not already cached get encoded — a
daily crawl adding N new articles costs O(N) encode calls, not a full re-encode
of the whole history. PCA reduction is cheap and applied fresh on every read —
this also lets the two groups share ONE PCA basis (fit on pooled train-period
rows) for a valid cross-group comparison in the Phase-11 EDA, instead of two
incomparable per-group spaces.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, VN30_TICKERS
from src.data.discover_news import discover_source_files, load_source
from src.eda.phase04_news_eda import TOPIC_CATEGORIES, _trading_calendar, effective_trading_date
from src.nlp.embeddings import extract_phobert_embeddings


def topic_flags(text: str) -> dict[str, int]:
    """1/0 flag per EDA-Guide category for one article (keyword match)."""
    t = str(text).lower()
    return {f"topic_{cat}_count": int(any(kw in t for kw in kws)) for cat, kws in TOPIC_CATEGORIES.items()}


FEATURES_DIR = PROJECT_ROOT / "data" / "features"
TRAIN_CUTOFF = "2020-01-01"  # kept independent of src.modeling.dataset.SPLIT_DATE (2025-01-01);
# must stay <= SPLIT_DATE to avoid leaking test-period rows into the PCA fit.
PCA_DIM = 32
RAW_DIM = 768
KHACH_QUAN_SOURCES = {
    "cafef", "hsc", "vnexpress", "thanhnien", "tuoitre", "nld", "vietnamplus",
    # 2026-07-18: thanhnien/tuoitre/vietnamplus each got a 2nd, non-overlapping crawl file
    # (a new top-level historical backfill alongside the existing objective/ tier-classified
    # file) -> discover_news.discover_source_files() now disambiguates the name collision by
    # parent directory ("_root"/"_objective" suffix) instead of silently dropping one file.
    # Both variants are still mainstream press -> both belong in khach_quan.
    "thanhnien_root", "thanhnien_objective",
    "tuoitre_root", "tuoitre_objective",
    "vietnamplus_root", "vietnamplus_objective",
}
TONG_HOP_SOURCES = {"ssi", "vndirect", "vnstock", "vietstock", "vsdc"}
GROUP_SOURCES = {"khach_quan": KHACH_QUAN_SOURCES, "tong_hop": TONG_HOP_SOURCES}
TICKER_PATTERN = re.compile(r"\b(" + "|".join(VN30_TICKERS) + r")\b", re.IGNORECASE)
TOPIC_COLS = [f"topic_{c}_count" for c in TOPIC_CATEGORIES]
# Strict match (raw_0, raw_1, ...) — NOT a loose prefix. Some crawl sources have a genuine
# 'raw_text' data column (objective/ tier schema); a loose "startswith('raw_')" check would
# wrongly sweep that text column up as if it were an embedding dimension.
_EMB_RAW_COL_RE = re.compile(r"^raw_\d+$")


def _raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _EMB_RAW_COL_RE.match(c)]


def unclassified_sources() -> set[str]:
    """Discovered sources that aren't in either group's classification — surfaced so a human
    notices a new crawl file was added and can classify it, rather than silently guessing."""
    return set(discover_source_files()) - KHACH_QUAN_SOURCES - TONG_HOP_SOURCES


def _load_group(group: str) -> pd.DataFrame:
    """Raw article rows for one group, with content + source columns."""
    if group not in GROUP_SOURCES:
        raise ValueError(f"unknown group: {group}")
    wanted = GROUP_SOURCES[group]
    files = discover_source_files()
    frames = []
    for source, path in files.items():
        if source not in wanted:
            continue
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
    news = news[news["_text"].str.len() > 0].reset_index(drop=True)
    if "url" not in news.columns:
        raise ValueError(f"group={group!r} news frame has no 'url' column; cannot cache incrementally")
    news = news.dropna(subset=["url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)
    # PhoBERT encoding is the expensive step (Docstring: "the EXPENSIVE step"), and
    # `_explode_tickers` downstream discards every article that never mentions a VN30 ticker
    # anyway (verified 2026-07-18: only ~0.3% of the ~1.45M-row post-backfill corpus mentions
    # one) — filtering BEFORE encoding changes zero output rows but cuts the CPU-only encode
    # job from an estimated ~37h to ~8min. Safe because every consumer of this pipeline
    # (`_build_raw` -> `_explode_tickers`) already only keeps ticker-matched rows.
    return news[news["_text"].str.contains(TICKER_PATTERN, regex=True, na=False)].reset_index(drop=True)


def _article_cache_path(source: str):
    """PER-SOURCE cache — keyed by url, one file per news source. Isolates sources from each
    other (a corrupted/incompatible cache for one source never affects another) and from group
    reclassification (changing which sources belong to which group — as happened once already —
    never requires re-encoding, since group membership is just a filter over ``source`` applied
    at read time, see ``_load_group``/``GROUP_SOURCES``)."""
    return FEATURES_DIR / f"news_emb_articles_{source}.parquet"


def _get_article_embeddings(source: str, news: pd.DataFrame) -> pd.DataFrame:
    """Incremental cache for ONE source, keyed by ``url``: encode only articles not already
    cached, then persist.

    A corrupted/truncated cache file is treated as empty (full re-encode of just this source)
    rather than crashing every future run — self-heals a bad file left by a killed process."""
    cache_path = _article_cache_path(source)
    cached = pd.DataFrame({"url": []})
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
        except Exception:
            cached = pd.DataFrame({"url": []})

    cached_raw_cols = _raw_cols(cached)
    if not cached.empty and cached_raw_cols and len(cached_raw_cols) != RAW_DIM:
        # embedding dimensionality changed (e.g. different model) -> cache is incompatible with
        # the current model; rebuild this source's cache rather than silently NaN-pad mismatched
        # columns (other sources' caches are untouched).
        cached = pd.DataFrame({"url": []})

    known = set(cached["url"]) if not cached.empty else set()
    new_rows = news[~news["url"].isin(known)]
    if new_rows.empty:
        return cached
    embs = extract_phobert_embeddings(new_rows["_text"].tolist())
    raw_cols = {f"raw_{i}": embs[:, i] for i in range(embs.shape[1])}
    new_df = pd.DataFrame({"url": new_rows["url"].values, **raw_cols})
    merged = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, index=False)
    return merged


def _explode_tickers(news: pd.DataFrame) -> pd.DataFrame:
    """One row per (article x mentioned ticker), with eff_date + source + raw embedding cols."""
    trading = _trading_calendar()
    eff_date = effective_trading_date(news["pub_date"], trading)
    tickers = [TICKER_PATTERN.findall(t) for t in news["_text"]]
    raw_cols = _raw_cols(news)
    records = []
    for i, tks in enumerate(tickers):
        if not tks or pd.isna(eff_date.iloc[i]):
            continue
        flags = topic_flags(news["_text"].iloc[i])
        row_raw = {c: news[c].iloc[i] for c in raw_cols}
        for t in {tk.upper() for tk in tks}:
            records.append({
                "ticker": t, "date": eff_date.iloc[i], "source": news["source"].iloc[i],
                **row_raw, **flags,
            })
    return pd.DataFrame(records)


def _build_raw(group: str) -> pd.DataFrame:
    """(date, ticker, source, raw_0..767, topic_*) — embeddings from each source's own
    incremental cache (merged per source, then combined for the group)."""
    news = _load_group(group)
    if news.empty:
        return pd.DataFrame()
    frames = []
    for source, sub_news in news.groupby("source"):
        article_embs = _get_article_embeddings(source, sub_news)
        if article_embs.empty:
            continue
        m = sub_news.merge(article_embs, on="url", how="inner")
        if not m.empty:
            frames.append(m)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        return pd.DataFrame()
    exploded = _explode_tickers(merged)
    if exploded.empty:
        return pd.DataFrame()
    exploded["date"] = exploded["date"].dt.normalize()
    return exploded


def load_or_build_raw(group: str) -> pd.DataFrame:
    """(date, ticker, source, raw_*, topic_*) for one group; PhoBERT runs only on new articles."""
    return _build_raw(group)


def _reduce(df: pd.DataFrame, dim: int = PCA_DIM) -> pd.DataFrame:
    """Apply PCA (fit on train-period rows) to a raw-embedding frame; honest fallback if too few
    train rows (keeps the full RAW embedding instead of a mislabeled 1-dim 'PCA').

    Output columns are ALWAYS ``emb_*`` (so every downstream consumer — modeling features,
    Phase-11/12 EDA — can rely on one naming convention); ``pca_applied`` records whether PCA
    actually ran, so the fallback is still visible/inspectable rather than silently mislabeled."""
    raw_cols = _raw_cols(df)
    embs = df[raw_cols].to_numpy()
    train_mask = (df["date"] < pd.Timestamp(TRAIN_CUTOFF)).to_numpy()
    n_train = int(train_mask.sum())
    if n_train < 2:
        reduced, out_dim, pca_applied = embs, embs.shape[1], False
    else:
        from sklearn.decomposition import PCA

        d = min(dim, embs.shape[1], max(1, n_train - 1))
        pca = PCA(n_components=d, svd_solver="randomized").fit(embs[train_mask])
        reduced, out_dim, pca_applied = pca.transform(embs).astype(np.float32), d, True
    emb_cols = {f"emb_{i}": reduced[:, i] for i in range(out_dim)}
    other = df.drop(columns=raw_cols)
    out = pd.concat([other.reset_index(drop=True), pd.DataFrame(emb_cols)], axis=1)
    out["pca_applied"] = pca_applied
    return out


def build_group_embeddings(group: str) -> pd.DataFrame:
    """(date, ticker, source, emb_0..emb_{dim-1}, topic_*) for one group — own PCA basis."""
    raw = load_or_build_raw(group)
    if raw.empty:
        return pd.DataFrame()
    return _reduce(raw)


def build_comparable_group_embeddings() -> dict[str, pd.DataFrame]:
    """Both groups reduced with a SHARED PCA (fit on pooled train-period rows), so they live in
    the same subspace and are validly comparable (Phase-11 EDA cross-group scatter/similarity)."""
    raws = {g: load_or_build_raw(g) for g in ("khach_quan", "tong_hop")}
    pooled_train = []
    for df in raws.values():
        if df.empty:
            continue
        raw_cols = _raw_cols(df)
        mask = (df["date"] < pd.Timestamp(TRAIN_CUTOFF)).to_numpy()
        pooled_train.append(df.loc[mask, raw_cols].to_numpy())
    pooled = np.concatenate(pooled_train, axis=0) if pooled_train else np.zeros((0, RAW_DIM))

    out = {}
    if len(pooled) >= 2:
        from sklearn.decomposition import PCA

        dim = min(PCA_DIM, pooled.shape[1], max(1, len(pooled) - 1))
        pca = PCA(n_components=dim, svd_solver="randomized").fit(pooled)
        for g, df in raws.items():
            if df.empty:
                out[g] = pd.DataFrame()
                continue
            raw_cols = _raw_cols(df)
            reduced = pca.transform(df[raw_cols].to_numpy()).astype(np.float32)
            emb_cols = {f"emb_{i}": reduced[:, i] for i in range(dim)}
            g_out = pd.concat([df.drop(columns=raw_cols).reset_index(drop=True), pd.DataFrame(emb_cols)], axis=1)
            g_out["pca_applied"] = True
            out[g] = g_out
    else:
        for g, df in raws.items():
            out[g] = _reduce(df) if not df.empty else pd.DataFrame()
    return out


def compute_novelty_scores(group: str = "tong_hop", window_days: int = 5) -> pd.DataFrame:
    """Story 12-1 — per (ticker, article): novelty = 1 - max cosine similarity (raw 768-dim
    space) to that ticker's articles published in the preceding ``window_days`` (INCLUDING
    same-day articles other than itself — a same-day rehash is exactly the kind of duplicate
    this feature should catch). No prior/same-day articles in the window -> novelty = 1.0
    (maximally novel by convention: nothing to compare against, so it cannot be flagged as a
    rehash).

    CAVEAT: consecutive trading days share most of their trailing window's articles, so
    ``novelty_mean`` is autocorrelated across days the same way Story 12-3's decayed embedding
    is (see ``phase15_temporal_decay_correlation.py``'s docstring) — naive Pearson/Spearman
    p-values on this feature should be read as exploratory, not as i.i.d.-sample-valid."""
    raw = load_or_build_raw(group)
    if raw.empty:
        return pd.DataFrame()
    raw_cols = _raw_cols(raw)
    out_frames = []
    for _ticker, sub in raw.groupby("ticker"):
        sub = sub.sort_values("date").reset_index(drop=True)
        embs = sub[raw_cols].to_numpy()
        norms = np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9, None)
        unit = embs / norms
        dates = sub["date"].to_numpy()
        novelty = np.ones(len(sub), dtype=float)
        for i in range(len(sub)):
            cutoff = dates[i] - np.timedelta64(window_days, "D")
            mask = (dates <= dates[i]) & (dates >= cutoff)
            mask[i] = False  # exclude the article itself
            if mask.any():
                sim = float(np.clip((unit[mask] @ unit[i]).max(), -1.0, 1.0))
                novelty[i] = 1.0 - sim
        rec = sub[["date", "ticker", "source"]].copy()
        rec["novelty"] = novelty
        out_frames.append(rec)
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()


def novelty_daily(group: str = "tong_hop", window_days: int = 5) -> pd.DataFrame:
    """Per (ticker, date): mean novelty of that day's articles."""
    scores = compute_novelty_scores(group, window_days)
    if scores.empty:
        return pd.DataFrame()
    return (
        scores.groupby(["ticker", "date"])["novelty"]
        .mean()
        .reset_index()
        .rename(columns={"novelty": "novelty_mean"})
    )


def decayed_embedding_features(
    group: str = "tong_hop", halflife_days: int = 5, lookback_days: int = 20, dim: int = PCA_DIM
) -> pd.DataFrame:
    """Story 12-3 — per (ticker, TRADING DAY): exponentially-decayed weighted mean embedding over
    the preceding ``lookback_days`` (weight ``0.5 ** (age_days / halflife_days)``, normalized to
    sum to 1 across contributing articles), then PCA-reduced (own basis, train-period fit).

    Evaluated over every trading day in the ticker's active coverage range (not just days the
    ticker had its own same-day article) — this is what makes it a genuine multi-day signal
    rather than same-day aggregation with extra steps. A trading day with zero contributing
    articles in the lookback window is not emitted (caller reindexes against the trading
    calendar -> NaN, consistent with the no-news-NaN rule elsewhere).

    CAVEAT: lookback windows overlap across consecutive trading days, so this feature is
    autocorrelated over time — naive Pearson/Spearman/FDR p-values assume i.i.d. samples and
    will look more "significant" than a rigorous (e.g. block-bootstrap-corrected) test would
    support. Treat correlations against this feature as exploratory only."""
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be > 0, got {halflife_days}")
    raw = load_or_build_raw(group)
    if raw.empty:
        return pd.DataFrame()
    raw_cols = _raw_cols(raw)
    # np.asarray forces raw numpy datetime64 scalars (not pd.Timestamp) when iterated below —
    # `pd.Timestamp - numpy.datetime64[us]-array` raises a UFuncTypeError on this numpy/pandas
    # version combo, while `numpy.datetime64 - numpy.datetime64-array` works correctly.
    trading_days = np.asarray(pd.to_datetime(pd.Series(_trading_calendar())).dt.normalize().unique())
    records = []
    for ticker, sub in raw.groupby("ticker"):
        sub = sub.sort_values("date").reset_index(drop=True)
        embs = sub[raw_cols].to_numpy()
        dates = sub["date"].to_numpy()
        lo, hi = dates.min(), dates.max()
        candidate_days = trading_days[(trading_days >= lo) & (trading_days <= hi)]
        for d in candidate_days:
            cutoff = d - np.timedelta64(lookback_days, "D")
            mask = (dates <= d) & (dates >= cutoff)
            if not mask.any():
                continue
            age_days = (d - dates[mask]) / np.timedelta64(1, "D")
            w = 0.5 ** (age_days / halflife_days)
            wsum = w.sum()
            if wsum <= 0:
                continue
            w = w / wsum
            weighted = (embs[mask] * w[:, None]).sum(axis=0)
            records.append({"ticker": ticker, "date": d, **{f"raw_{i}": weighted[i] for i in range(len(weighted))}})
    if not records:
        return pd.DataFrame()
    decayed = pd.DataFrame(records)
    reduced = _reduce(decayed, dim=dim)
    return reduced.rename(columns={c: c.replace("emb_", "emb_decay_") for c in reduced.columns if c.startswith("emb_")})


def run() -> list:
    """Populate each source's PER-SOURCE incremental article-embedding cache (PhoBERT runs only
    on articles not already cached). Logs which source files were discovered/processed this run
    to ``reports/news_processing_log.md`` for traceability."""
    from src.data.discover_news import discover_source_files, load_source, log_processing

    processed = []
    for source, path in discover_source_files().items():
        try:
            df = load_source(source, path)
        except Exception:
            continue
        group = "khach_quan" if source in KHACH_QUAN_SOURCES else "tong_hop" if source in TONG_HOP_SOURCES else "unclassified"
        processed.append({"source": f"{source} ({group})", "path": str(path), "n_rows": len(df)})
    if processed:
        log_processing(processed)

    for group in ("khach_quan", "tong_hop"):
        load_or_build_raw(group)
    return sorted(p for p in FEATURES_DIR.glob("news_emb_articles_*.parquet"))


if __name__ == "__main__":  # pragma: no cover
    for p in run():
        print(f"Wrote {p}")
