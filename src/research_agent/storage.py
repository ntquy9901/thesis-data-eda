"""SQLite storage for experiment results — cho phép so sánh cross-method."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "results" / "research_agent" / "experiments.db"


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            params TEXT,
            metrics TEXT,
            started_at TEXT,
            finished_at TEXT,
            duration_s REAL,
            status TEXT DEFAULT 'done',
            error TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            path TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)
    conn.commit()
    return conn


def save_result(
    name: str,
    category: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    started_at: str,
    finished_at: str,
    duration_s: float,
    status: str = "done",
    error: str | None = None,
    artifacts: list[str] | None = None,
) -> int:
    conn = _ensure_db()
    cur = conn.execute(
        "INSERT INTO experiments (name, category, params, metrics, started_at, finished_at, duration_s, status, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, category, json.dumps(params), json.dumps(metrics), started_at, finished_at, duration_s, status, error),
    )
    eid = cur.lastrowid
    if artifacts:
        for p in artifacts:
            conn.execute("INSERT INTO experiment_artifacts (experiment_id, path) VALUES (?, ?)", (eid, p))
    conn.commit()
    conn.close()
    return eid


def get_all_results(category: str | None = None) -> list[dict]:
    conn = _ensure_db()
    if category:
        rows = conn.execute(
            "SELECT name, category, params, metrics, duration_s, status, error, created_at "
            "FROM experiments WHERE category = ? ORDER BY created_at DESC", (category,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, category, params, metrics, duration_s, status, error, created_at "
            "FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "name": r[0], "category": r[1],
            "params": json.loads(r[2]) if r[2] else {},
            "metrics": json.loads(r[3]) if r[3] else {},
            "duration_s": r[4], "status": r[5], "error": r[6], "created_at": r[7],
        })
    return results


def get_comparison_df(category: str) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    rows = get_all_results(category)
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        rec = {"name": r["name"], "category": r["category"],
               "duration_s": r["duration_s"], "status": r["status"], "created_at": r["created_at"]}
        rec.update(r["metrics"])
        rec.update(r["params"])
        records.append(rec)
    return pd.DataFrame(records)
