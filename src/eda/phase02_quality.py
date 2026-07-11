"""Phase 2 — Data Quality (per EDA Guide).

Checks missing values (by column, by stock, by date), duplicates (news by
url/title, prices by date), and invalid values (negative volume, high<low,
future timestamps). Outputs land under ``eda_output/quality/``.

Per EDA Guide rule: never modify raw data — reads sources read-only.
"""

from __future__ import annotations

import json

import pandas as pd

from config import EDA_TICKERS, PRICE_DATA_DIR
from src.eda.common import ensure_output_dirs, phase_output_dir
from src.eda.phase01_profiling import NEWS_FILES

DATE_COLS = ("pub_date", "date", "effective_date", "collected_at")


# ---------- pure helpers (unit-tested) ----------
def missingness_for(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-column missingness for one table: rows = (table, column, n_missing, pct)."""
    if df.empty:
        return pd.DataFrame(columns=["table", "column", "n_missing", "pct"])
    n = len(df)
    records = []
    for col in df.columns:
        nm = int(df[col].isna().sum())
        records.append(
            {"table": name, "column": col, "n_missing": nm, "pct": round(nm / n * 100, 2)}
        )
    return pd.DataFrame(records)


def missingness_by_stock(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Overall missingness % per stock ticker (one row per ticker).

    ``frames`` maps ticker -> its OHLCV DataFrame. Cell-level missingness
    captures nulls; date gaps are reported separately by :func:`date_gap_report`.
    """
    rows = []
    for ticker, df in frames.items():
        n_cells = int(df.shape[0]) * int(df.shape[1])
        nm = int(df.isna().sum().sum())
        rows.append(
            {
                "ticker": ticker,
                "n_rows": int(len(df)),
                "n_missing_cells": nm,
                "pct_missing": round(nm / n_cells * 100, 2) if n_cells else 0.0,
            }
        )
    return pd.DataFrame(rows)


def date_gap_count(df: pd.DataFrame, date_col: str = "date") -> int:
    """Count missing trading dates (gaps) between min and max date.

    Uses Mon–Fri business days as a proxy for the VN trading calendar (does not
    subtract Tet/holidays, so the count is an upper bound on real gaps).
    """
    if date_col not in df.columns or df.empty:
        return 0
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return 0
    expected = pd.bdate_range(dates.min(), dates.max())
    return int(len(expected.difference(dates)))


def duplicates_for(df: pd.DataFrame, key_cols: list[str]) -> int:
    """Count duplicate rows on ``key_cols`` (NaN keys excluded, not counted as dups)."""
    present = [c for c in key_cols if c in df.columns]
    if not present:
        return 0
    return int(df.dropna(subset=present).duplicated(subset=present).sum())


def price_invalid(df: pd.DataFrame) -> dict:
    """Detect invalid OHLCV rows: negative/unparseable volume, high<low, future dates.

    Numeric columns are coerced so string-stored prices/volumes are handled; the
    future-date check is tz-safe (normalizes tz-aware vs tz-naive before compare).
    """
    out = {"negative_volume": 0, "unparseable_volume": 0, "high_lt_low": 0, "future_dates": 0}
    if df.empty:
        return out
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        out["negative_volume"] = int((vol < 0).sum())
        out["unparseable_volume"] = int(vol.isna().sum() - df["volume"].isna().sum())
    if {"high", "low"}.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        out["high_lt_low"] = int((high < low).sum())
    for col in DATE_COLS:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.dt.tz is not None:  # tz-aware → strip tz for naive compare
                ts = ts.dt.tz_localize(None)
            out["future_dates"] = int((ts > pd.Timestamp.now()).sum())
            break
    return out


# ---------- phase runner ----------
def _read_news(name: str, path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def missingness_report() -> pd.DataFrame:
    """Per-column missingness across all news files + EDA tickers."""
    frames = [missingness_for(_read_news(n, p), n) for n, p in NEWS_FILES.items() if p.exists()]
    for ticker in EDA_TICKERS:
        p = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if p.exists():
            frames.append(missingness_for(pd.read_csv(p, encoding="utf-8"), f"{ticker}_price"))
    if not frames:
        return pd.DataFrame(columns=["table", "column", "n_missing", "pct"])
    return pd.concat(frames, ignore_index=True)


def duplicate_report() -> dict:
    rep: dict[str, int] = {}
    for name, p in NEWS_FILES.items():
        if not p.exists():
            continue
        df = _read_news(name, p)
        # cafef exposes ``article_url`` instead of ``url``; count whichever exists.
        url_col = "url" if "url" in df.columns else ("article_url" if "article_url" in df.columns else None)
        rep[f"{name}_by_url"] = duplicates_for(df, [url_col]) if url_col else 0
        rep[f"{name}_by_title"] = duplicates_for(df, ["title"])
    for ticker in EDA_TICKERS:
        p = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if p.exists():
            rep[f"{ticker}_price_by_date"] = duplicates_for(pd.read_csv(p, encoding="utf-8"), ["date"])
    return rep


def invalid_values_report() -> dict:
    return {
        ticker: price_invalid(pd.read_csv(p, encoding="utf-8"))
        for ticker in EDA_TICKERS
        if (p := PRICE_DATA_DIR / f"{ticker}_ohlcv.csv").exists()
    }


def by_stock_and_date() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (missingness_by_stock, date_gaps) for the EDA ticker price panel."""
    frames: dict[str, pd.DataFrame] = {}
    gaps: list[dict] = []
    for ticker in EDA_TICKERS:
        p = PRICE_DATA_DIR / f"{ticker}_ohlcv.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8")
        frames[ticker] = df
        gaps.append({"ticker": ticker, "date_gaps": date_gap_count(df)})
    return missingness_by_stock(frames), pd.DataFrame(gaps)


def run_phase() -> list:
    """Run all Phase-2 checks; return list of written artifact paths."""
    ensure_output_dirs()
    outdir = phase_output_dir("quality")

    by_col = missingness_report()
    by_col_path = outdir / "missingness_report.csv"
    by_col.to_csv(by_col_path, index=False, encoding="utf-8")

    by_stock, by_date = by_stock_and_date()
    by_stock_path = outdir / "missingness_by_stock.csv"
    by_stock.to_csv(by_stock_path, index=False, encoding="utf-8")
    by_date_path = outdir / "missingness_by_date.csv"
    by_date.to_csv(by_date_path, index=False, encoding="utf-8")

    dup = duplicate_report()
    dup_path = outdir / "duplicate_report.json"
    dup_path.write_text(json.dumps(dup, indent=2), encoding="utf-8")

    inv = invalid_values_report()
    inv_path = outdir / "invalid_values.json"
    inv_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")

    return [by_col_path, by_stock_path, by_date_path, dup_path, inv_path]


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
