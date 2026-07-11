"""Shared pytest configuration: force a headless matplotlib backend so plotting
modules run reliably inside the test suite (no GUI / display required)."""

import matplotlib

matplotlib.use("Agg")
