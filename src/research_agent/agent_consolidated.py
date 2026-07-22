"""Consolidated reporting — generates master summary from all agents' results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT
from src.research_agent.storage import get_all_results

logger = logging.getLogger("agent_consolidated")
handler = logging.FileHandler("results/research_agent/logs/agent_consolidated.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

RESULTS_DIR = PROJECT_ROOT / "results" / "research_agent"
REPORTS_DIR = RESULTS_DIR / "reports"
CONSOLIDATED_PATH = RESULTS_DIR / "consolidated_report.md"


def _compute_delta_r2(results: list[dict], category: str) -> dict:
    by_name = {}
    for r in results:
        if r["category"] != category:
            continue
        name = r["name"]
        if name not in by_name:
            by_name[name] = []
        if "r2" in r["metrics"] and r["metrics"]["r2"] is not None:
            by_name[name].append(r["metrics"]["r2"])
    stats = {}
    for name, vals in by_name.items():
        if vals:
            stats[name] = {
                "mean_r2": round(sum(vals) / len(vals), 6),
                "min_r2": round(min(vals), 6),
                "max_r2": round(max(vals), 6),
                "n_runs": len(vals),
            }
    return stats


def _find_best(results: list[dict], metric: str = "r2") -> dict | None:
    best = None
    best_val = -float("inf")
    for r in results:
        v = r["metrics"].get(metric)
        if v is not None and v > best_val:
            best_val = v
            best = r
    return best


def generate_consolidated():
    results = get_all_results()
    if not results:
        return

    lines = [
        f"# Research Agent — Consolidated Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total experiments: {len(results)}",
        "",
    ]

    n_categories = {}
    for r in results:
        n_categories[r["category"]] = n_categories.get(r["category"], 0) + 1
    lines.append("## Experiments by Category\n")
    for cat, n in sorted(n_categories.items()):
        lines.append(f"- **{cat}**: {n} runs")
    lines.append("")

    for category in sorted(set(r["category"] for r in results)):
        cat_results = [r for r in results if r["category"] == category]
        lines.append(f"\n## Category: {category} ({len(cat_results)} runs)\n")
        lines.append("| Method | Mean R² | Min R² | Max R² | Runs |")
        lines.append("|--------|---------|--------|--------|------|")
        stats = _compute_delta_r2(results, category)
        for name, s in sorted(stats.items()):
            lines.append(f"| {name} | {s['mean_r2']:.6f} | {s['min_r2']:.6f} | {s['max_r2']:.6f} | {s['n_runs']} |")
        best = _find_best(cat_results)
        if best:
            lines.append(f"\n**Best in {category}:** {best['name']} (r2={best['metrics'].get('r2', 'N/A')})")

    lines.append("\n## Top-5 Best Runs Overall\n")
    sorted_by_r2 = sorted(results, key=lambda r: r["metrics"].get("r2", -float("inf")), reverse=True)[:5]
    lines.append("| Rank | Name | Category | R² | RMSE |")
    lines.append("|------|------|----------|-----|------|")
    for i, r in enumerate(sorted_by_r2, 1):
        r2 = r["metrics"].get("r2", "N/A")
        rmse = r["metrics"].get("rmse", "N/A")
        lines.append(f"| {i} | {r['name']} | {r['category']} | {r2} | {rmse} |")

    lines.append("\n## Agent Status\n")
    lines.append("| Agent | Log | Status |")
    lines.append("|-------|-----|--------|")
    agent_logs = [
        ("Sentiment", RESULTS_DIR / "logs" / "agent_sentiment.log"),
        ("Model", RESULTS_DIR / "logs" / "agent_model.log"),
        ("Feature", RESULTS_DIR / "logs" / "agent_feature.log"),
        ("Horizon", RESULTS_DIR / "logs" / "agent_horizon.log"),
        ("Ticker", RESULTS_DIR / "logs" / "agent_ticker.log"),
    ]
    for name, log_path in agent_logs:
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                content = f.read()
            last_line = content.strip().split("\n")[-1] if content.strip() else "(empty)"
            lines.append(f"| {name} | {log_path.name} | `{last_line}` |")
        else:
            lines.append(f"| {name} | {log_path.name} | Not started |")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLIDATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSOLIDATED_PATH.write_text("\n".join(lines), encoding="utf-8")

    json_path = RESULTS_DIR / "consolidated_data.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_experiments": len(results),
        "categories": n_categories,
        "stats": {cat: _compute_delta_r2(results, cat) for cat in set(r["category"] for r in results)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Consolidated report: {CONSOLIDATED_PATH}")
    return CONSOLIDATED_PATH


if __name__ == "__main__":
    generate_consolidated()
