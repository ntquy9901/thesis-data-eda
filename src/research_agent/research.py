"""Web research module — tự động search papers và methods mới về financial NLP."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import PROJECT_ROOT

RESEARCH_LOG = PROJECT_ROOT / "results" / "research_agent" / "research_log.jsonl"


def _log(entry: dict):
    RESEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


TOPICS = [
    "financial news sentiment analysis stock prediction",
    "NLP event detection earnings calls transcripts",
    "news-driven volatility forecasting GARCH",
    "large language models finance text classification",
    "alternative data news sentiment trading strategy",
    "Vietnamese stock news NLP sentiment analysis",
    "news embedding cross-sectional return prediction",
]


def search_arxiv(topic: str, max_results: int = 3) -> list[dict]:
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{topic.replace(' ', '+')}",
        "start": 0, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending",
    }
    try:
        r = httpx.get(url, params=params, timeout=15.0, headers={"User-Agent": "ResearchAgent/1.0"})
        r.raise_for_status()
        papers = []
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns)
            summary = entry.find("a:summary", ns)
            papers.append({
                "title": title.text.strip().replace("\n", " ") if title is not None else "",
                "summary": summary.text.strip().replace("\n", " ") if summary is not None else "",
                "source": "arxiv",
            })
        return papers
    except Exception as e:
        _log({"event": "search_error", "topic": topic, "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()})
        return []


def search_web(topic: str) -> list[dict]:
    try:
        from web_search import search as ws_search
        results = ws_search(topic, max_results=5)
        return [{"title": r.get("title", ""), "snippet": r.get("snippet", ""), "source": "web"} for r in results]
    except ImportError:
        pass
    try:
        r = httpx.get(f"https://api.duckduckgo.com/?q={topic}&format=json", timeout=10.0)
        data = r.json()
        return [{"title": data.get("Heading", ""), "snippet": data.get("AbstractText", ""), "source": "duckduckgo"}]
    except Exception:
        return []


def run_research_cycle() -> list[dict]:
    findings = []
    for topic in TOPICS:
        papers = search_arxiv(topic)
        for p in papers:
            _log({"event": "paper_found", "topic": topic, **p, "timestamp": datetime.now(timezone.utc).isoformat()})
            findings.append(p)
    _log({"event": "research_cycle_complete", "n_findings": len(findings),
           "timestamp": datetime.now(timezone.utc).isoformat()})
    return findings
