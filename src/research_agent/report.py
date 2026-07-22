"""Comparison reporting — so sánh kết quả cross-experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "results" / "research_agent" / "reports"


def generate_comparison_report(category: str) -> str:
    from src.research_agent.storage import get_comparison_df

    df = get_comparison_df(category)
    if df.empty:
        return f"No experiments found for category '{category}'"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Research Agent — {category} Comparison",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
    ]

    metric_cols = [c for c in ["r2", "rmse", "mae", "qlike", "dir_acc"] if c in df.columns]
    group_cols = [c for c in ["name", "model", "feature_set", "target"] if c in df.columns]

    if "name" in df.columns and metric_cols:
        for _, row in df.iterrows():
            label = " | ".join(str(row.get(c, "")) for c in group_cols if c in df.columns)
            lines.append(f"\n### {label}")
            for mc in metric_cols:
                val = row.get(mc)
                lines.append(f"  {mc}: {val}" if val is not None else f"  {mc}: N/A")
            dur = row.get("duration_s")
            if dur is not None:
                lines.append(f"  duration: {dur:.1f}s")

    if metric_cols and "name" in df.columns:
        lines.append("\n## Summary Stats per Method\n")
        for name, grp in df.groupby("name"):
            lines.append(f"\n### {name} ({len(grp)} runs)")
            for mc in metric_cols:
                vals = grp[mc].dropna()
                if len(vals) > 0:
                    lines.append(f"  {mc}: mean={vals.mean():.6f}  std={vals.std():.6f}  "
                                 f"min={vals.min():.6f}  max={vals.max():.6f}")

    report = "\n".join(lines)
    path = REPORTS_DIR / f"{category}_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    path.write_text(report, encoding="utf-8")
    return str(path)
