"""Smoke tests for Vietnam stock analysis project.

These tests verify basic functionality and package health.
Run with: pytest -m smoke
"""

from pathlib import Path

import pytest


@pytest.mark.smoke
def test_project_structure():
    """Verify basic project structure exists."""
    project_root = Path(__file__).parent.parent

    # Check key directories exist
    assert (project_root / "src").exists()
    assert (project_root / "data").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "reports").exists()

    # Check src subdirectories
    assert (project_root / "src" / "data").exists()
    assert (project_root / "src" / "features").exists()
    assert (project_root / "src" / "analysis").exists()


@pytest.mark.smoke
def test_package_imports():
    """Verify core packages can be imported."""
    import numpy as np
    import pandas as pd

    # Basic sanity checks
    assert np.__version__ is not None
    assert pd.__version__ is not None

    # Test basic operations
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(df) == 3
    assert df["a"].sum() == 6


@pytest.mark.smoke
def test_claude_md_exists():
    """Verify CLAUDE.md exists and contains key sections."""
    project_root = Path(__file__).parent.parent
    claude_md = project_root / "CLAUDE.md"

    assert claude_md.exists()
    content = claude_md.read_text(encoding="utf-8")

    # Check for key sections from the template
    assert "Think Before Coding" in content
    assert "Simplicity First" in content
    assert "Definition of Done" in content
    assert "Per-project setup" in content


@pytest.mark.smoke
def test_time_zone_configuration():
    """Verify Vietnam timezone is properly configured."""
    from datetime import datetime

    import pytz

    # Check Vietnam timezone exists
    vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    assert vn_tz is not None

    # Test timezone conversion
    utc_now = datetime.now(pytz.UTC)
    vn_time = utc_now.astimezone(vn_tz)
    assert vn_time is not None
