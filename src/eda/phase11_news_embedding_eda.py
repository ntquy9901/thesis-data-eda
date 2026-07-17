"""Phase 11 — News embedding EDA (Story 11-1).

Compares PhoBERT embedding vectors across two MUTUALLY EXCLUSIVE groups, split by source
identity (redefined after user feedback — see ``news_embeddings.py``'s module docstring):
- "khach_quan" (objective/factual reporting): cafef, hsc, vnexpress, thanhnien, tuoitre, nld,
  vietnamplus
- "tong_hop" (aggregated/analyst commentary): ssi, vndirect, vnstock, vietstock, vsdc

Sources are discovered dynamically (``src.data.discover_news``) — new crawl files are picked up
automatically; any source not in either list is tagged "unclassified" here so it's visible
rather than silently dropped.

Outputs -> eda_output/news_embedding/:
- ``source_stats.csv`` — per-source article count, embedded (ticker-matched) count,
  missing title/lead rate, mean embedding norm (quality proxy)
- ``group_scatter.png`` — PCA 2D scatter, colored by group
- ``group_similarity.json`` — mean cosine similarity within vs across groups
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.discover_news import discover_source_files, load_source
from src.eda.common import ensure_output_dirs, phase_output_dir
from src.features.news_embeddings import GROUP_SOURCES, build_comparable_group_embeddings


def _source_stats() -> pd.DataFrame:
    """Per-source: article count, missing title/lead rate (data quality proxy), tagged with
    which group (khach_quan/tong_hop/unclassified) each source belongs to."""
    source_to_group = {s: g for g, srcs in GROUP_SOURCES.items() for s in srcs}
    rows = []
    for s, path in discover_source_files().items():
        try:
            df = load_source(s, path)
        except Exception:
            continue
        title_missing = df.get("title", pd.Series(dtype=object)).isna().mean() if not df.empty else np.nan
        lead_missing = df.get("lead", pd.Series(dtype=object)).isna().mean() if not df.empty else np.nan
        rows.append({
            "source": s, "group": source_to_group.get(s, "unclassified"), "n_articles": len(df),
            "title_missing_rate": round(float(title_missing), 4) if pd.notna(title_missing) else None,
            "lead_missing_rate": round(float(lead_missing), 4) if pd.notna(lead_missing) else None,
        })
    return pd.DataFrame(rows)


def _embedding_coverage(emb_by_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (group, source): rows embedded + mean embedding L2 norm (quality proxy)."""
    rows = []
    for group, df in emb_by_group.items():
        if df.empty:
            continue
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        for src, sub in df.groupby("source"):
            norms = np.linalg.norm(sub[emb_cols].to_numpy(), axis=1)
            rows.append({
                "group": group, "source": src, "n_embedded_rows": len(sub),
                "mean_emb_norm": round(float(norms.mean()), 4),
                "std_emb_norm": round(float(norms.std()), 4),
            })
    return pd.DataFrame(rows)


def _group_similarity(emb_by_group: dict[str, pd.DataFrame], sample: int = 500, seed: int = 0) -> dict:
    """Mean cosine similarity within each group vs across the two groups (sampled)."""
    rng = np.random.default_rng(seed)
    samples = {}
    for group, df in emb_by_group.items():
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if df.empty or not emb_cols:
            continue
        arr = df[emb_cols].to_numpy()
        idx = rng.choice(len(arr), size=min(sample, len(arr)), replace=False)
        v = arr[idx]
        samples[group] = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)

    def _mean_cos(a, b) -> float | None:
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            return None
        sim = a @ b.T
        return round(float(sim.mean()), 4)

    groups = list(samples.keys())
    out = {"n_groups": len(groups)}
    for g in groups:
        out[f"within_{g}"] = _mean_cos(samples[g], samples[g])
    if len(groups) == 2:
        out["across_groups"] = _mean_cos(samples[groups[0]], samples[groups[1]])
    return out


def _plot_group_scatter(emb_by_group: dict[str, pd.DataFrame], out_path) -> bool:
    """Returns True iff a plot was written (both groups share ONE PCA basis, fit upstream by
    build_comparable_group_embeddings — a joint 2D projection is only valid because of that)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    frames = []
    for group, df in emb_by_group.items():
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if df.empty or not emb_cols:
            continue
        sub = df[emb_cols].copy()
        sub["group"] = group
        frames.append(sub)
    if not frames:
        return False
    pooled = pd.concat(frames, ignore_index=True)
    emb_cols = [c for c in pooled.columns if c.startswith("emb_")]
    xy = PCA(n_components=2).fit_transform(pooled[emb_cols].to_numpy())

    fig, ax = plt.subplots(figsize=(7, 6))
    for group, color in zip(pooled["group"].unique(), ["#1f77b4", "#d62728"], strict=False):
        mask = (pooled["group"] == group).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.4, label=group, color=color)
    ax.set_title("News embedding PCA (khách quan vs tổng hợp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def run_phase() -> list:
    ensure_output_dirs()
    outdir = phase_output_dir("news_embedding")

    written = []
    stats = _source_stats()
    stats.to_csv(outdir / "source_stats.csv", index=False, encoding="utf-8")
    written.append(outdir / "source_stats.csv")

    emb_by_group = build_comparable_group_embeddings()

    coverage = _embedding_coverage(emb_by_group)
    coverage.to_csv(outdir / "embedding_coverage.csv", index=False, encoding="utf-8")
    written.append(outdir / "embedding_coverage.csv")

    sim = _group_similarity(emb_by_group)
    (outdir / "group_similarity.json").write_text(json.dumps(sim, indent=2), encoding="utf-8")
    written.append(outdir / "group_similarity.json")

    scatter_path = outdir / "group_scatter.png"
    if _plot_group_scatter(emb_by_group, scatter_path):
        written.append(scatter_path)

    return written


if __name__ == "__main__":  # pragma: no cover
    for p in run_phase():
        print(f"Wrote {p}")
