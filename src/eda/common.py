"""Shared utilities for the EDA pipeline (Phases 1-10).

Provides the VN30 subset scope, ``eda_output/`` path helpers, and shared plot
styling. This module never reads or mutates raw source data — phase modules
reuse the loaders in :mod:`src.data.load_prices` / :mod:`src.data.load_news`.

Per EDA Guide rules: never modify raw data; every finding must be backed by
statistics or visualization; all artifacts go under ``eda_output/``.
"""

from pathlib import Path

from config import EDA_OUTPUT_DIR, EDA_TICKERS, PROJECT_ROOT

__all__ = [
    "EDA_TICKERS",
    "EDA_OUTPUT_DIR",
    "EDA_SUBDIRS",
    "ensure_output_dirs",
    "ticker_output_path",
    "phase_output_dir",
    "configure_plots",
]

# Phase subdirectories (matches the EDA Guide output structure + Arch §16.3).
# Phase 6 (event study) and Phase 5 (relationship) share "relationship/".
EDA_SUBDIRS: list[str] = [
    "profiling",  # Phase 1
    "quality",  # Phase 2
    "price",  # Phase 3
    "news",  # Phase 4 + Phase 7 (sparse news)
    "relationship",  # Phase 5 + Phase 6 (event study)
    "feature_engineering",  # Phase 8
    "leakage",  # Phase 9
    "report",  # Final report
]


def ensure_output_dirs() -> list[Path]:
    """Create ``eda_output/`` and every phase subdirectory.

    Idempotent: existing directories are left untouched. Returns the list of
    directory paths (root first, then subdirs in EDA_SUBDIRS order).
    """
    dirs = [EDA_OUTPUT_DIR] + [EDA_OUTPUT_DIR / s for s in EDA_SUBDIRS]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def phase_output_dir(phase: str) -> Path:
    """Return ``eda_output/<phase>/``, validating the phase name."""
    if phase not in EDA_SUBDIRS:
        raise ValueError(f"phase must be one of {EDA_SUBDIRS}, got {phase!r}")
    return EDA_OUTPUT_DIR / phase


def ticker_output_path(ticker: str, phase: str, name: str, ext: str = "png") -> Path:
    """Return ``eda_output/<phase>/<ticker>_<name>.<ext>`` for a per-ticker artifact.

    Use :func:`phase_output_dir` directly for aggregate (cross-ticker) artifacts
    such as correlation heatmaps.

    Args:
        ticker: stock ticker, e.g. ``"VCB"``.
        phase: one of :data:`EDA_SUBDIRS`.
        name: artifact name without the ticker prefix or extension, e.g. ``"returns"``.
        ext: file extension without the dot (default ``"png"``).
    """
    if phase not in EDA_SUBDIRS:
        raise ValueError(f"phase must be one of {EDA_SUBDIRS}, got {phase!r}")
    if not ext.isalpha():
        raise ValueError(f"ext must be alphabetic (no dot), got {ext!r}")
    return EDA_OUTPUT_DIR / phase / f"{ticker}_{name}.{ext}"


def configure_plots() -> None:
    """Apply shared matplotlib style used by all phase visualization modules.

    DejaVu Sans (matplotlib default) renders Vietnamese diacritics, so no extra
    font dependency is required. Call once at the top of each plotting script.
    """
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


if __name__ == "__main__":  # pragma: no cover
    created = ensure_output_dirs()
    print(f"Ensured {len(created)} EDA output dirs under {EDA_OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
