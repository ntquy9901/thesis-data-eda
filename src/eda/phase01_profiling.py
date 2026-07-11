"""Phase 1 — Dataset Profiling (per EDA Guide).

Produces a profiling table: one row per input table with row_count,
col_count, dtype summary, primary/candidate key, date range, and memory.

Output: ``eda_output/profiling/profiling_table.csv``

Per EDA Guide rule: never modify raw data — this reads sources read-only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CRAWL_DATA_ROOT, EDA_TICKERS, PRICE_DATA_DIR
from src.eda.common import ensure_output_dirs, phase_output_dir

# Tables to profile (news consolidated + per-source + macro). Schema-agnostic:
# we detect columns by name, not position, because sources differ (cafef uses
# ``section``/``article_url``; others use ``category``/``url``).
NEWS_FILES = {
    "news_articles": CRAWL_DATA_ROOT / "news_articles.csv",
    "ssi_articles": CRAWL_DATA_ROOT / "ssi_articles.csv",
    "cafef_articles": CRAWL_DATA_ROOT / "cafef_articles.csv",
    "vndirect_articles": CRAWL_DATA_ROOT / "vndirect_articles.csv",
}
MACRO_FILES = {
    "dxy": CRAWL_DATA_ROOT / "macro" / "raw" / "dxy.csv",
    "sbv_policy_rates": CRAWL_DATA_ROOT / "macro" / "raw" / "sbv_policy_rates.csv",
}
DATE_COLS = ("pub_date", "date", "effective_date", "collected_at")


def dtype_summary(df: pd.DataFrame) -> str:
    """Compact dtype histogram, e.g. ``object(5), int64(2), float64(4)``."""
    counts = df.dtypes.astype(str).value_counts()
    return ", ".join(f"{k}({v})" for k, v in counts.items())


def detect_date_range(df: pd.DataFrame) -> tuple[str | None, pd.Timestamp, pd.Timestamp]:
    """Return (matched_date_col, min, max) for the first present date column.

    Uses ``format="mixed", utc=True`` (pandas >= 2.0) so a column that interleaves
    ISO and DD/MM strings AND mixes tz-aware/naive values (the consolidated
    news_articles.csv does both) does not collapse to mass-NaT nor raise a
    mixed-timezone error. Residual DD/MM-vs-MM/DD ambiguity for purely-numeric
    dates remains; authoritative per-source normalization is Phase 4.
    """
    for col in DATE_COLS:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", format="mixed", utc=True)
            return col, s.min(), s.max()
    return None, pd.Timestamp(None), pd.Timestamp(None)


def detect_keys(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Primary key = ``id`` if unique-ish; candidate key = url-like column."""
    primary = "id" if "id" in df.columns else ("date" if "date" in df.columns else None)
    for cand in ("url", "article_url", "title"):
        if cand in df.columns:
            return primary, cand
    return primary, None


def profile_dataframe(name: str, df: pd.DataFrame) -> dict:
    """Build one profiling row from an in-memory DataFrame (pure, testable)."""
    date_col, dmin, dmax = detect_date_range(df)
    primary, candidate = detect_keys(df)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    return {
        "table": name,
        "row_count": len(df),
        "col_count": df.shape[1],
        "dtypes": dtype_summary(df),
        "primary_key": primary,
        "candidate_key": candidate,
        "date_col": date_col,
        "date_min": dmin,
        "date_max": dmax,
        "memory_mb": round(mem_mb, 2),
    }


def profile_table() -> pd.DataFrame:
    """Profile all news/macro files + each EDA ticker's OHLCV file."""
    rows: list[dict] = []
    for name, path in {**NEWS_FILES, **MACRO_FILES}.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        rows.append(profile_dataframe(name, df))
    for ticker in EDA_TICKERS:
        path = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8")
        rows.append(profile_dataframe(f"{ticker}_price", df))
    return pd.DataFrame(rows)


def run_phase() -> Path:
    ensure_output_dirs()
    df = profile_table()
    out = phase_output_dir("profiling") / "profiling_table.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    return out


if __name__ == "__main__":  # pragma: no cover
    out = run_phase()
    print(f"Wrote {out} ({len(pd.read_csv(out))} tables profiled)")
