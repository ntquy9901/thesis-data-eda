# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

> **How to use:** Place this file as `CLAUDE.md` in your repo root. Fill the **Per-project setup** block at the bottom (the only place stack specifics go). Claude Code (and other AI coding tools that read CLAUDE.md) will follow these rules automatically. Version 1.0 — 2026-07-09.

---

## ⚙️ BMAD Workflow Rules (BẮT BUỘC — áp dụng cho mọi công việc phát triển)

> Quy tắc ưu tiên cao nhất: **mọi công việc code/dev phải chạy theo quy trình BMAD**, không làm task lẻ tẻ. Vi phạm = dừng và thiết lập lại quy trình trước khi code.

**Quy trình bắt buộc (theo thứ tự):**
1. **Sprint** — Mọi công việc thuộc một sprint được theo dõi trong `_bmad-output/sprint-status.yaml` (tạo bằng `/bmad-sprint-planning`). Chưa có sprint → chưa code.
2. **Story** — Mỗi công việc phải thuộc một Story rõ ràng (tạo bằng `/bmad-create-story`). Không làm task mồ côi không có story.
3. **Task** — Mỗi Story tách thành các Task có tiêu chí hoàn thành verify được (test/smoke pass).
4. **Dev/Review loop** — Dùng `bmad-loop run` (orchestrator) để drive: dev pass (`bmad-dev-auto`, tự review + commit) → review pass → commit theo story.

**Quy tắc vận hành:**
- **Trước khi code:** đọc `sprint-status.yaml`, xác định sprint + story + task hiện tại. Nếu thiếu → tạo bằng skill bmad tương ứng trước.
- **Khi gặp blocker/xung đột:** dùng `bmad-loop-resolve` (escalation), không tự ý bỏ qua.
- **Khi gom công việc trì hoãn:** dùng `bmad-loop-sweep` (deferred-work ledger).
- **Sau khi code:** cập nhật sprint status, verify tests pass, commit gắn với story ID.

**Cấu hình & tooling:**
- Config chia sẻ: `_bmad/config.yaml` | Config cá nhân: `_bmad/config.user.yaml`
- Orchestrator policy: `.bmad-loop/policy.toml` | CLI adapters: `claude`, `codex`
- **Điều kiện tiên quyết:** orchestrator `bmad-loop` phải được cài (`uv tool install bmad-loop`). Nếu chưa cài, **ưu tiên cài trước** khi bắt đầu dev — không có tool, skills không tự chạy.

---

## 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

## 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan with per-step verify checks.

---

# Project Quality Rules

> These rules are **project-agnostic** — they apply to any language/stack. Fill the **Per-project setup** block at the bottom once per repo; do not hardcode stack details into the rules.

## Definition of Done
A task is done only when ALL are true:
- Code directly satisfies the requested change; no unrelated refactor.
- **Tests:** when behavior changes, write/run unit tests and ensure **>= 80% of the CHANGED lines are covered**, measured by **diff-coverage** (NOT total): produce a coverage report for the change, then run a diff-coverage gate (e.g. `diff-cover <coverage-report> --fail-under=80`). Ensure the change is committed or staged so the diff is measurable.
- **Checks run:** run the project's test + lint commands — or mark `Not run` with a reason. Never claim a command passed unless it actually ran.
- **Lint scope:** exclude vendored / generated / third-party tooling directories from lint (they are not the project's own code).
- **Code review (always):** must run a code review (e.g. `/bmad-code-review` in Claude Code, or a PR-based adversarial peer review) and address findings before marking done. **Required for every change — including non-production (docs/config/scripts) — no exception.** Summarize the result + actions in the report.
- **Summary report:** generate a context-appropriate `reports/<YYYY-MM-DD_HHMM>_summaryOfUpdate_report.md` (not a rigid template).
- **Smoke (gate):** at least one smoke test (tagged `smoke` — register the tag/marker in your test runner config) that boots the app/service and runs one happy-path (e.g. a health endpoint returns 200). The smoke command **must pass before done**. If a smoke test needs live infra / external services, mark `Not run` locally with a reason and run it in CI.
- **Impact analysis:** before a non-trivial change, identify its blast radius — find all callers/dependents/consumers (grep the symbol; check the project's registration & integration points; note cross-repo consumers). Summarize what's affected + what was verified. Flag risk if blast radius is high and not fully test-covered.
- **Similar check:** after a fix/pattern change, grep the same idiom/duplicate across the repo and any sibling / shared-scaffolding repos. Apply the same change where applicable, or list remaining instances as a follow-up. Don't fix one of N copies silently.

## Testing quality rules (ENFORCED — learned from Epic 1 review)
Line coverage alone does NOT prove quality. The Epic-1 adversarial review found 5 real bugs (date mass-NaT, tz-aware crash, NaN-counted-as-duplicate, dead `known_tickers` code, missing acceptance criteria) while helper line-coverage was 100%. Therefore, for every story:
- **Run the diff-coverage GATE, not just `pytest --cov`.** Exact command: `uv run pytest tests/unit --cov=src --cov-report=xml -q && diff-cover coverage.xml --fail-under=80 --compare-branch origin/main`. A green `pytest --cov 100%` is NOT sufficient — `diff-cover` must pass on the changed lines. If it reports <80%, add tests for the uncovered (usually runner/I/O) lines before done.
- **Cover I/O runners, not only pure helpers.** Unit tests of pure functions miss the bugs in data-loading/report-writing code. Every `run_phase()` / report-builder function must have an integration test (monkeypatch file paths → tmp fixtures) before its story is done.
- **Data-pipeline tests must include a real-data-sample smoke.** Synthetic fixtures miss encoding (UTF-8 vs cp1252), mixed date formats (ISO vs DD/MM), mixed timezones, and schema drift across sources. At least one test per phase reads a SMALL slice of the real source data and asserts it runs without exception and emits sane output.
- **Code review is `/bmad-code-review` (3-layer: Blind Hunter + Edge Case Hunter + Acceptance Auditor), run BEFORE marking done.** Self-review is NOT a substitute. Address all confirmed findings (critical/major) before done; document minors as follow-ups.

## Summary report (generated per change)
When a change is done, **generate** a concise, context-appropriate markdown summary — do not fill a rigid template. Save it as `reports/<YYYY-MM-DD_HHMM>_summaryOfUpdate_report.md`.
- Write it to fit THIS change: include what's relevant and omit what's not — **except code review, which is always required and always summarized.**
- Cover, as applicable: what changed, files changed (path → purpose), tests + coverage %, code-review result + actions, commands actually run, risks/follow-ups, a Definition-of-Done checklist.
- Be honest: state only what truly happened; write `Not run` (with reason) for anything skipped.

## Code hygiene (all languages)
- No hidden global state / unbounded in-process caches (use bounded TTL/size caches; externalize shared state to a managed store).
- No secrets in code (use a secrets manager / env).
- No hardcoded absolute local paths (use `pathlib.Path` relative to project root or config).
- No production logic that lives only in a notebook.

## Data source rules (Vietnam stock analysis project)
- **PRIMARY DATA SOURCES:** All raw data MUST come from these directories only:
  1. **News data:** `D:\bmad-projects\crawl_data\data` (mandatory)
  2. **Price data:** `D:\bmad-projects\stock_vol_prediction01\data\raw` (mandatory)
- **No alternative data sources:** Do not fetch or use data from other locations unless explicitly requested
- **News data files available:**
  - `news_articles.csv` - Consolidated news articles from multiple sources
  - `cafef_articles.csv` - CafeF news articles
  - `ssi_articles.csv` - SSI news articles  
  - `vndirect_articles.csv` - VNDirect news articles
  - `vnstock_articles.csv` - VietStock news articles
  - `hsc_articles.csv` - HSC news articles
  - `cafef_candidates.jsonl` - Candidate articles for crawling
  - PDF extraction data in `pdf/`, `pdf_ssi/`, `macro/` subdirectories
- **Price data files available:**
  - `prices/` - Individual stock OHLCV files (format: `{TICKER}_ohlcv.csv`)
  - Available tickers: ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, MBB, MSN, MWG, NVL, PDR, PLX, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM (VN30 constituents)
  - Additional directories: `vn30/`, `vn100/`, `hnx/` with enhanced versions
  - Price data format: `date,open,high,low,close,volume` (date format: YYYY-MM-DD)
- **Data access pattern:** Read from source directories → process → save to `data/processed/`
- **No modification of source:** Never write to or modify files in the source directories

## Data refresh rules (daily updates)
- **DAILY UPDATES:** User crawls new data daily - analysis MUST detect and use new data automatically
- **Refresh mechanism:** Use `src.data.refresh_data` module to check for and load new data
- **Refresh detection:** Check file modification timestamps and data date ranges to identify new data
- **Incremental processing:** Only process new/changed data, not entire dataset (unless requested)
- **Refresh frequency:** Can be configured in `config.REFRESH_SETTINGS`
- **Manual refresh:** Call `refresh_all_data()` to force reload from sources
- **Auto-refresh on analysis:** When running analysis, always check if data is current (within last 24 hours unless otherwise specified)
- **Data versioning:** Track data versions in `data/processed/.data_versions.json` to enable rollback if needed

---

## Per-project setup (fill in once per repo — this is the ONLY place stack specifics go)
- **Language / toolchain:** Python 3.10+ + uv (package manager)
- **Test command:** `uv run pytest tests/`
- **Coverage source + diff-coverage command:** source `src/`; `uv run pytest --cov=src --cov-report=xml && diff-cover coverage.xml --fail-under=80`
- **Lint command:** `uv run ruff check . && uv run mypy src/`
- **Lint excludes (vendored/generated dirs):** `.agents .claude _bmad data/raw data/processed`
- **Smoke command:** `uv run pytest -m smoke`
- **Code-review tool:** `/code-review`
- **Language-specific extras:** 
  - Python: avoid bare `except` and mutable default args; use type hints + `pathlib`
  - Use `polars` for large datasets (>1M rows), `pandas` otherwise
  - All notebooks must have companion Python module with production logic extracted
  - Vietnamese text: use `underthesea` for tokenization, `sentence-transformers` for embeddings

---

# Vietnamese Stock Market Analysis Project

## Tổng quan (Project Overview)

Dự án phân tích chất lượng dữ liệu từ dữ liệu thô liên quan đến thị trường chứng khoán Việt Nam.

**Mục tiêu chính:**
1. Phân tích mối tương quan giữa tin tức cổ phiếu và giá cổ phiếu
2. Phân tích mối tương quan giữa tin tức và độ biến động cổ phiếu

## Nguồn dữ liệu (Data Sources)

### 1. Dữ liệu Tin tức (News Data)
**PRIMARY SOURCE:** `D:\bmad-projects\crawl_data\data` (mandatory per CLAUDE.md)

**Available files:**
- `news_articles.csv` - Consolidated news from all sources
- `cafef_articles.csv` - CafeF articles
- `ssi_articles.csv` - SSI articles
- `vndirect_articles.csv` - VNDirect articles
- `vnstock_articles.csv` - VietStock articles
- `hsc_articles.csv` - HSC articles

**News data structure (from CSV):**
- `source`: Nguồn tin (cafef, ssi, vndirect, vnstock, hsc)
- `title`: Tiêu đề bài viết
- `category`: Chuyên mục (thi-truong-chung-khoan, etc.)
- `pub_date`: Thời gian xuất bản (ISO 8601, UTC+7)
- `url`: Link bài viết gốc
- `author`: Tác giả (optional)
- `lead`: Tóm tắt/Lead bài viết
- `pdf_url`: Link PDF (nếu có)
- `collected_at`: Thời gian cào dữ liệu

**Additional data:**
- PDF files in `pdf/`, `pdf_ssi/`, `macro/` subdirectories
- `cafef_candidates.jsonl` - Candidate URLs for crawling

### 2. Dữ liệu Giá cổ phiếu (Stock Price Data)
**PRIMARY SOURCE:** `D:\bmad-projects\stock_vol_prediction01\data\raw` (mandatory per CLAUDE.md)

**Available data:**
- **Individual stock OHLCV:** `prices/{TICKER}_ohlcv.csv` files
  - Available tickers: 30 VN30 constituents (ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, MBB, MSN, MWG, NVL, PDR, PLX, POW, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM)
  - Format: `date,open,high,low,close,volume`
  - Date format: YYYY-MM-DD
  - Timezone: UTC+7 (Asia/Ho_Chi_Minh)
- **Index data:** Available in `vn30/`, `vn100/`, `hnx/` directories
- **Enhanced versions:** `vn30_enhanced/`, `vn100_enhanced/`, `hnx_enhanced/` directories

**Sample data structure (VCB as example):**
```
date,open,high,low,close,volume
2009-06-30,9.13,9.13,9.13,9.13,294070
2009-07-01,9.13,9.20,9.13,9.20,564520
...
```

**Data coverage:**
- Historical data from 2009 onwards (varies by ticker)
- Daily frequency
- No adjusted prices (raw close prices)

## Cấu trúc dự án (Project Structure)

```
data_eda/
├── data/
│   ├── raw/                # Dữ liệu gốc (không commit vào git)
│   │   ├── news/
│   │   └── prices/
│   ├── processed/          # Dữ liệu đã xử lý
│   └── features/           # Feature engineering
├── notebooks/              # Jupyter EDA (không commit vào git)
├── src/                    # Production code
│   ├── data/              # Data loading/processing
│   ├── features/          # Feature engineering
│   ├── analysis/          # Correlation/statistical analysis
│   └── visualization/      # Plotting/reporting
├── tests/                  # Tests
├── reports/               # Generated reports
└── config/                # Configuration files
```

## Công cụ (Tech Stack)

### Core Data
- `pandas` - Tabular data (<1M rows)
- `polars` - Large datasets (>=1M rows)
- `numpy` - Numerical computing

### Financial Analysis
- `yfinance` - Market data fetching (if needed)
- `arch` - Volatility models (GARCH)
- `statsmodels` - Statistical testing

### NLP (Vietnamese)
- `underthesea` - Vietnamese tokenization/POS tagging
- `transformers` - Multilingual models
- `sentence-transformers` - Text embeddings

### Visualization
- `matplotlib` - Static plots
- `seaborn` - Statistical visualization
- `plotly` - Interactive charts

## Quy trình phân tích (Analysis Pipeline)

### 1. Data Quality Assessment
- Missing values, outliers detection
- Data consistency checks
- Basic EDA per data source

### 2. Feature Engineering

**News features:**
- Sentiment scores (VADER, PhoBERT-based)
- Topic modeling (LDA, NMF)
- Text embeddings (multilingual sentence-transformers)
- Publication timing features (hour-of-day, day-of-week)
- News volume/aggregation

**Price features:**
- Returns (log returns, % returns)
- Volatility measures (realized vol, ATR, Parkinson)
- Volume indicators
- Technical indicators (optional: RSI, MACD, etc.)

### 3. Correlation Analysis

**News-Price:**
- Event study (cumulative abnormal returns)
- Lead-lag analysis (cross-correlation)
- Granger causality testing
- Regression analysis (sentiment ~ returns)

**News-Volatility:**
- Volatility clustering (GARCH models)
- News impact on volatility
- High-volatility event detection
- Conditional volatility analysis

### 4. Statistical Testing
- Hypothesis testing (t-test, Mann-Whitney)
- Significance tests (p-values, confidence intervals)
- Multiple testing correction (Bonferroni, FDR)

## Đặc thù thị trường Việt Nam (Vietnam Market Specifics)

- **Trading hours**: Mon-Fri 9:00-11:30, 13:00-15:00 (UTC+7)
- **Holidays**: Tết Nguyên Đán, Giỗ Tổ Hùng Vương, Liberation Day, Labor Day, National Day
- **Price limits**: ±7% daily band (exceptions: IPO, new listings)
- **Settlement**: T+2 cycle
- **News timing**: News published outside trading hours (evening/weekend)
- **Currency**: VND

## Quy convention (Conventions)

- **Code language**: English (variables, functions, comments)
- **Documentation language**: Vietnamese (context-appropriate)
- **Date format**: ISO 8601 (YYYY-MM-DD)
- **Timezone**: UTC+7 (Asia/Ho_Chi_Minh)
- **Encoding**: UTF-8 (Vietnamese support)
- **Data location**: All data under `data/` (gitignored)
- **Notebooks**: Exploratory only; production logic in `src/`
