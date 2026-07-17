"""Dynamic discovery + schema-normalizing loader for news CSVs under ``CRAWL_DATA_ROOT``.

Per CLAUDE.md: raw news data comes from ALL files under crawl_data/data — the user adds new
crawl files continuously, so sources are discovered by scanning + schema-validating the
directory tree, not by hardcoding filenames. A small denylist skips known backup/archive/
duplicate snapshots of the SAME underlying crawl (verified by content inspection: identical
brokerage-name ``source``-column values to ``vnstock_articles.csv``, overlapping row counts) —
this is NOT a source allowlist; any new schema-valid file is picked up automatically.

Two schemas are recognized:
- OLD (cafef/ssi/vndirect/vnstock/hsc-style): ``title``, ``pub_date``, ``lead`` (DD/MM or ISO
  mixed dates, per the existing project convention).
- NEW "tier" schema (``objective/`` subdirectory, added 2026-07-17): ``title``, ``publish_time``
  (ISO 8601 UTC), ``source_tier``, ``raw_text`` (full body, richer than the old ``lead``).

Every processing run appends a dated entry to ``reports/news_processing_log.md`` recording which
files were read (name + full path + row count), for traceability.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config import CRAWL_DATA_ROOT, PROJECT_ROOT

# Backup/archive/duplicate snapshots of the SAME underlying vnstock PDF crawl (verified: their
# 'source' column holds brokerage names like "Vietstock"/"MBS"/"KBSV" identical to
# vnstock_articles.csv, with overlapping/subset row counts) — not distinct sources.
_DENYLIST = {
    "data.csv",
    "data_2021_2025.csv",
    "data_archive.csv",
    "vnstock_pdf_raw.csv",
    "vnstock_pdfs_extracted.csv",
    # news_articles.csv is the literal union of cafef+ssi+vndirect+hsc (row counts verified to
    # match exactly) — reading it AND its constituents would double-count every article.
    "news_articles.csv",
}
# Rolling consolidated snapshot (explicitly a partial union — see CLAUDE.md), not a distinct
# source; the underlying tier files (vietstock_records.csv, vsdc_records.csv, ...) are read
# directly instead.
_SNAPSHOT_PREFIX = "objective_v"

OLD_SCHEMA_COLS = {"title", "pub_date"}
NEW_SCHEMA_COLS = {"title", "publish_time", "source_tier"}

PROCESSING_LOG_PATH = PROJECT_ROOT / "reports" / "news_processing_log.md"


def _infer_source_name(path: Path) -> str:
    """Derive a short source name from a filename, e.g. 'news_unenriched_vnexpress_records.csv'
    -> 'vnexpress', 'cafef_articles.csv' -> 'cafef'."""
    name = path.stem
    if name.startswith("news_unenriched_"):
        name = name[len("news_unenriched_"):]
    for suffix in ("_articles", "_records"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.lower()


def discover_source_files() -> dict[str, Path]:
    """{source_name: path} for every schema-valid news CSV under ``CRAWL_DATA_ROOT``
    (recursive), excluding known backups/duplicates/snapshots."""
    found: dict[str, Path] = {}
    if not CRAWL_DATA_ROOT.exists():
        return found
    for p in sorted(CRAWL_DATA_ROOT.rglob("*.csv")):
        if p.name in _DENYLIST or p.name.startswith(_SNAPSHOT_PREFIX):
            continue
        try:
            cols = set(pd.read_csv(p, nrows=0, encoding="utf-8").columns)
        except Exception:
            continue
        if not (OLD_SCHEMA_COLS <= cols or NEW_SCHEMA_COLS <= cols):
            continue
        found[_infer_source_name(p)] = p
    return found


def load_source(source: str, path: Path) -> pd.DataFrame:
    """Load one discovered file, normalized to include ``source``/``pub_date``/``lead`` columns
    regardless of which schema the file uses."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = set(df.columns)
    if NEW_SCHEMA_COLS <= cols:
        df["pub_date"] = pd.to_datetime(df["publish_time"], errors="coerce", utc=True).dt.tz_localize(None)
        df["lead"] = df.get("raw_text", pd.Series(index=df.index)).fillna("")
    else:
        df["pub_date"] = pd.to_datetime(
            df.get("pub_date"), format="mixed", dayfirst=True, errors="coerce", utc=True
        ).dt.tz_localize(None)
        df["lead"] = df.get("lead", pd.Series(index=df.index)).fillna("")
    df["source"] = source
    return df


def load_all_sources(log: bool = True) -> dict[str, pd.DataFrame]:
    """Discover + load every news source. Appends a dated trace-log entry to
    ``reports/news_processing_log.md`` (file name + full path + row count) unless ``log=False``
    (tests pass ``log=False`` to avoid polluting the real log)."""
    files = discover_source_files()
    out: dict[str, pd.DataFrame] = {}
    processed = []
    for source, path in files.items():
        try:
            df = load_source(source, path)
        except Exception:
            continue
        out[source] = df
        processed.append({"source": source, "path": str(path), "n_rows": len(df)})
    if log and processed:
        log_processing(processed)
    return out


def log_processing(processed: list[dict]) -> None:
    """Append a dated section to ``reports/news_processing_log.md`` for traceability."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n## {ts}\n"]
    for rec in sorted(processed, key=lambda r: r["source"]):
        lines.append(f"- `{rec['source']}` — `{rec['path']}` ({rec['n_rows']} rows)")
    PROCESSING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
