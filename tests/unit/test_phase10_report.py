"""Tests for src.eda.phase10_visualizations and src.eda.report."""

import pytest

from src.eda import phase10_visualizations as p10
from src.eda import report


def test_candidate_features_frame_has_recommendation():
    df = report.candidate_features_frame()
    assert "feature" in df.columns and "recommendation" in df.columns
    assert set(df["recommendation"]).issubset({"keep", "drop", "review"})
    # parkinson_vol is a baseline-aligned keep (unless dropped, which is rare)
    assert "parkinson_vol" in set(df["feature"])


def test_real_phase10_run_smoke():
    written = p10.run_phase()
    if not written:
        pytest.skip("no upstream artifacts (run phases 1-9 first)")
    names = {p.name for p in written}
    assert "charts_index.md" in names


def test_real_report_run_smoke():
    written = report.run()
    if not written:
        pytest.skip("no artifacts")
    names = {p.name for p in written}
    assert {"eda_final_report.md", "candidate_features.csv"} <= names
    text = next(p for p in written if p.name == "eda_final_report.md").read_text(encoding="utf-8")
    assert "Executive Summary" in text and "Candidate Features" in text and "Risks" in text
