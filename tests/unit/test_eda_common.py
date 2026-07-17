"""Unit tests for src.eda.common (Story 1.1)."""

import pytest

from src.eda import common
from src.eda.common import (
    EDA_OUTPUT_DIR,
    EDA_SUBDIRS,
    EDA_TICKERS,
    ensure_output_dirs,
    phase_output_dir,
    ticker_output_path,
)


def test_eda_tickers_is_vn30_subset():
    """EDA_TICKERS must be VN30 tickers (scaled to full VN30 in Epic 8)."""
    from config import VN30_TICKERS

    assert EDA_TICKERS == VN30_TICKERS
    assert len(EDA_TICKERS) == 30
    assert "VCB" in EDA_TICKERS
    assert all(isinstance(t, str) and t.isupper() for t in EDA_TICKERS)


def test_eda_subdirs_match_guide():
    """Subdirs must cover the 8 EDA Guide output areas + news_embedding (Story 11-1)."""
    expected = {
        "profiling",
        "quality",
        "price",
        "news",
        "relationship",
        "feature_engineering",
        "leakage",
        "report",
        "news_embedding",
        "uncertainty",
    }
    assert set(EDA_SUBDIRS) == expected


def test_ensure_output_dirs_creates_all(tmp_path, monkeypatch):
    """ensure_output_dirs creates the root + all 8 subdirs (idempotent)."""
    # Redirect EDA_OUTPUT_DIR to a temp location to avoid touching the real tree.
    fake_root = tmp_path / "eda_output"
    monkeypatch.setattr(common, "EDA_OUTPUT_DIR", fake_root)

    created = ensure_output_dirs()
    assert len(created) == 1 + len(EDA_SUBDIRS)
    assert fake_root.is_dir()
    for sub in EDA_SUBDIRS:
        assert (fake_root / sub).is_dir()

    # Idempotent: running again does not raise and returns the same set.
    created_again = ensure_output_dirs()
    assert len(created_again) == len(created)


def test_phase_output_dir_validates():
    """Unknown phase names raise ValueError; known names return the dir."""
    assert phase_output_dir("price") == EDA_OUTPUT_DIR / "price"
    with pytest.raises(ValueError):
        phase_output_dir("bogus")


def test_ticker_output_path_constructs_correctly(tmp_path, monkeypatch):
    """ticker_output_path builds <ticker>_<name>.<ext> under the right phase dir."""
    monkeypatch.setattr(common, "EDA_OUTPUT_DIR", tmp_path / "eda_output")
    p = ticker_output_path("VCB", "price", "returns")
    assert p == tmp_path / "eda_output" / "price" / "VCB_returns.png"

    p2 = ticker_output_path("FPT", "price", "acf_pacf", ext="svg")
    assert p2 == tmp_path / "eda_output" / "price" / "FPT_acf_pacf.svg"

    with pytest.raises(ValueError):
        ticker_output_path("VCB", "nope", "x")
    with pytest.raises(ValueError):
        ticker_output_path("VCB", "price", "x", ext=".png")  # ext must not contain dot


def test_configure_plots_is_callable():
    """configure_plots runs without error (imports matplotlib lazily)."""
    common.configure_plots()  # should not raise
