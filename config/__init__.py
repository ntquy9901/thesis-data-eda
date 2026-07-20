"""Configuration module for Vietnam stock analysis."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"

# PRIMARY DATA SOURCES (per CLAUDE.md rule)
# All raw data MUST come from these directories only

# News data source
CRAWL_DATA_ROOT = Path("C:/luanvan/crawl_data/data")
CRAWL_NEWS_ARTICLES = CRAWL_DATA_ROOT / "news_articles.csv"
CRAWL_CAFEF_ARTICLES = CRAWL_DATA_ROOT / "cafef_articles.csv"
CRAWL_SSI_ARTICLES = CRAWL_DATA_ROOT / "ssi_articles.csv"
CRAWL_VNDIRECT_ARTICLES = CRAWL_DATA_ROOT / "vndirect_articles.csv"
CRAWL_VNSTOCK_ARTICLES = CRAWL_DATA_ROOT / "vnstock_articles.csv"
CRAWL_HSC_ARTICLES = CRAWL_DATA_ROOT / "hsc_articles.csv"
CRAWL_PDF_DIR = CRAWL_DATA_ROOT / "pdf"
CRAWL_PDF_SSI_DIR = CRAWL_DATA_ROOT / "pdf_ssi"
CRAWL_MACRO_DIR = CRAWL_DATA_ROOT / "macro"

# Price data source
PRICE_DATA_ROOT = Path("C:/luanvan/stock_vol_prediction01/data/raw")
PRICE_DATA_DIR = PRICE_DATA_ROOT / "prices"
VN30_DIR = PRICE_DATA_ROOT / "vn30"
VN100_DIR = PRICE_DATA_ROOT / "vn100"
HNX_DIR = PRICE_DATA_ROOT / "hnx"
VN30_ENHANCED_DIR = PRICE_DATA_ROOT / "vn30_enhanced"
VN100_ENHANCED_DIR = PRICE_DATA_ROOT / "vn100_enhanced"
HNX_ENHANCED_DIR = PRICE_DATA_ROOT / "hnx_enhanced"

# Available tickers (VN30 constituents)
VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SHB", "SSB",
    "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM"
]

# EDA scope (per EDA Guide / PRD §14). Scaled to the FULL VN30 in Epic 8 (Story 8-1)
# — broader evidence base for the nonlinear modeling comparison.
EDA_TICKERS = list(VN30_TICKERS)
EDA_OUTPUT_DIR = PROJECT_ROOT / "eda_output"

# Vietnam market configuration
TIMEZONE = "Asia/Ho_Chi_Minh"
TRADING_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
TRADING_HOURS_MORNING = (9, 0)  # 9:00 AM
TRADING_HOURS_AFTERNOON = (13, 0)  # 1:00 PM
TRADING_HOURS_END = (15, 0)  # 3:00 PM

# Price limits
DAILY_PRICE_LIMIT_PCT = 0.07  # ±7%

# Data quality thresholds
MISSING_DATA_THRESHOLD = 0.2  # 20% max missing
OUTLIER_STD_THRESHOLD = 3  # 3 standard deviations

# NLP configuration
VIETNAMESE_TOKENIZER = "underthesea"
SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Cache configuration
ENABLE_CACHE = True
CACHE_TTL_SECONDS = 3600  # 1 hour

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
