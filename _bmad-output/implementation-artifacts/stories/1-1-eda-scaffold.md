# Story 1.1 — EDA Scaffold + Config

**Epic:** 1 (EDA Foundation)
**Story key:** `1-1-eda-scaffold`
**Phase:** setup (enables Phases 1–10)
**FR:** FR-017
**Status:** ready-for-dev

## Context
This is the first story of the EDA pipeline (see `_bmad-output/planning-artifacts/epics.md`, `docs/PRD.md` §14, `docs/Technical_Architecture.md` §16). It creates the `src/eda/` package skeleton, configures the VN30 subset scope, and sets up the `eda_output/` directory structure so all subsequent phase modules (1.2 → 6.2) can write artifacts consistently.

Reuse existing loaders — do NOT duplicate data access:
- `src/data/load_prices.py` → `load_stock_ohlcv(ticker, start_date, end_date)`
- `src/data/load_news.py` → `load_news_articles(source, start_date, end_date)`
- `config/__init__.py` → paths, `VN30_TICKERS`

## Requirements (Acceptance Criteria)
- [ ] `src/eda/__init__.py` created
- [ ] `src/eda/common.py` providing:
  - `EDA_TICKERS = ["VCB", "FPT", "HPG", "SSI", "MWG"]`
  - `EDA_OUTPUT_DIR` (resolved from config, = `{project-root}/eda_output`)
  - `EDA_SUBDIRS = ["profiling", "quality", "price", "news", "relationship", "feature_engineering", "leakage", "report"]`
  - `ensure_output_dirs()` → creates `eda_output/` + all subdirs (idempotent)
  - `ticker_output_path(ticker, phase, ext)` → helper returning `eda_output/<phase>/<ticker>_<...>.<ext>`
  - shared matplotlib style (Vietnamese font-safe, figsize defaults)
- [ ] `config/__init__.py`: add `EDA_TICKERS`, `EDA_OUTPUT_DIR = PROJECT_ROOT / "eda_output"`
- [ ] `eda_output/` + 8 subdirectories created on run
- [ ] Unit test `tests/unit/test_eda_common.py`: `ensure_output_dirs()` creates all 8 dirs; `import src.eda.common` succeeds
- [ ] Smoke: existing smoke tests still pass

## Technical Notes
- Use `pathlib.Path`, no hardcoded absolute paths (CLAUDE.md hygiene).
- Do NOT read/modify raw source data in this story — only set up structure.
- Match existing code style in `src/data/`.

## Out of Scope
- Actual profiling/quality logic (Stories 1.2, 1.3)
- Any phase computation

## Verify
```bash
.venv/Scripts/python.exe -m pytest -m smoke
.venv/Scripts/python.exe -m pytest tests/unit/test_eda_common.py -v
.venv/Scripts/python.exe -c "from src.eda.common import ensure_output_dirs; ensure_output_dirs(); print('ok')"
```
Definition of Done: all acceptance boxes checked, smoke + new unit test pass, changed lines ≥80% diff-coverage.
