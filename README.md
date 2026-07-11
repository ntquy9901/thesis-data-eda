# Vietnamese Stock Market Data Analysis - News & Price Correlation

Phân tích mối tương quan giữa tin tức và giá cổ phiếu/độ biến động trên thị trường chứng khoán Việt Nam.

## 🚀 Quick Start

```bash
# Install uv (fast Python package manager)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run smoke tests to verify setup
pytest -m smoke

# Run linting
ruff check .
mypy src/
```

## 📁 Project Structure

```
data_eda/
├── src/                    # Production code (Python modules)
│   ├── data/              # Data loading/processing
│   ├── features/          # Feature engineering
│   ├── analysis/          # Correlation/statistical analysis
│   └── visualization/      # Plotting/reporting
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── test_smoke.py      # Smoke tests
├── data/                   # Data storage (gitignored)
│   ├── raw/               # Raw data from sources
│   ├── processed/         # Cleaned/processed data
│   └── features/          # Engineered features
├── notebooks/              # Jupyter notebooks (exploratory, gitignored)
├── reports/               # Generated analysis reports
├── config/                # Configuration files
├── CLAUDE.md              # Quality standards and guidelines ⭐
├── pyproject.toml         # Project configuration
└── requirements.txt       # Dependencies

## 🔧 Development Workflow

Per CLAUDE.md quality standards:

1. **Think before coding** - State assumptions, ask when unclear
2. **Simplicity first** - Minimum code, no speculative features
3. **Surgical changes** - Touch only what's necessary
4. **Goal-driven** - Verify with tests and code review

### Definition of Done

Every change must satisfy:
- ✅ Code directly satisfies request
- ✅ Tests written (≥80% diff-coverage)
- ✅ All checks pass (pytest, ruff, mypy)
- ✅ Code reviewed and findings addressed
- ✅ Summary report generated in `reports/`
- ✅ Smoke tests passing

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Smoke tests (verify basic functionality)
pytest -m smoke

# With coverage
pytest --cov=src --cov-report=xml
diff-cover coverage.xml --fail-under=80
```

### Linting

```bash
# Run all linters
ruff check .
mypy src/

# Auto-fix issues
ruff check . --fix
```

## 📊 Current Status

🚧 **Đang chờ dữ liệu** - Dự án đang ở giai đoạn setup, chờ user cung cấp:
- Dữ liệu tin tức (news data)
- Dữ liệu giá cổ phiếu (price data)

## 📖 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Quality standards, behavioral guidelines, and tech stack details ⭐ **READ THIS**
- **[MEMORY.md](MEMORY.md)** - Project memory and context index
- **[reports/README.md](reports/README.md)** - Report generation conventions

## 🎯 Tech Stack

- **Language**: Python 3.10+
- **Package Manager**: uv (recommended) or pip
- **Testing**: pytest + pytest-cov + diff-cover
- **Linting**: ruff + mypy
- **Data**: pandas, polars, numpy
- **Finance**: yfinance, arch (GARCH volatility)
- **NLP**: underthesea (Vietnamese), transformers
- **Visualization**: matplotlib, seaborn, plotly

---

*Updated: 2026-07-11*
