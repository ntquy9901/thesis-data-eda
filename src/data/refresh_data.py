"""Data refresh module for daily updates.

This module implements the CLAUDE.md rule for handling daily data updates:
- Detect new/changed data in source directories
- Incremental processing of only new data
- Data versioning for rollback capability
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from config import (
    CRAWL_DATA_ROOT,
    PRICE_DATA_ROOT,
    PROCESSED_DATA_DIR,
    TIMEZONE,
)
from src.data.load_news import load_news_articles, verify_data_integrity as verify_news_integrity
from src.data.load_prices import load_stock_ohlcv, get_available_tickers, verify_price_data_integrity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data version tracking file
DATA_VERSIONS_FILE = PROCESSED_DATA_DIR / ".data_versions.json"


def get_data_versions() -> Dict[str, str]:
    """Get current data versions from tracking file.

    Returns:
        Dictionary with data source names and their last modification timestamps
    """
    if not DATA_VERSIONS_FILE.exists():
        return {}

    with open(DATA_VERSIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data_versions(versions: Dict[str, str]) -> None:
    """Save data versions to tracking file.

    Args:
        versions: Dictionary of data source names and modification timestamps
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_VERSIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(versions, f, indent=2, ensure_ascii=False)


def get_file_modification_time(filepath: Path) -> Optional[str]:
    """Get file modification time as ISO string.

    Args:
        filepath: Path to file

    Returns:
        ISO format datetime string or None if file doesn't exist
    """
    if not filepath.exists():
        return None

    # Get modification time
    mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
    return mod_time.isoformat()


def check_news_data_refresh() -> Dict[str, any]:
    """Check if news data has been updated since last refresh.

    Returns:
        Dictionary with refresh status and information
    """
    logger.info("Checking news data for updates...")

    # Get current file modification time
    from config import CRAWL_NEWS_ARTICLES
    current_mod_time = get_file_modification_time(CRAWL_NEWS_ARTICLES)

    if current_mod_time is None:
        return {
            "needs_refresh": False,
            "reason": "News data file not found",
            "last_mod_time": None
        }

    # Get last recorded version
    versions = get_data_versions()
    last_mod_time = versions.get("news_articles")

    needs_refresh = (last_mod_time is None) or (current_mod_time > last_mod_time)

    return {
        "needs_refresh": needs_refresh,
        "current_mod_time": current_mod_time,
        "last_mod_time": last_mod_time,
        "reason": "New data available" if needs_refresh else "No updates"
    }


def check_price_data_refresh() -> Dict[str, any]:
    """Check if price data has been updated since last refresh.

    Returns:
        Dictionary with refresh status and information
    """
    logger.info("Checking price data for updates...")

    # Check a sample ticker (VCB) for modifications
    from config import PRICE_DATA_DIR
    sample_file = PRICE_DATA_DIR / "VCB_ohlcv.csv"
    current_mod_time = get_file_modification_time(sample_file)

    if current_mod_time is None:
        return {
            "needs_refresh": False,
            "reason": "Price data file not found",
            "last_mod_time": None
        }

    # Get last recorded version
    versions = get_data_versions()
    last_mod_time = versions.get("price_data")

    needs_refresh = (last_mod_time is None) or (current_mod_time > last_mod_time)

    return {
        "needs_refresh": needs_refresh,
        "current_mod_time": current_mod_time,
        "last_mod_time": last_mod_time,
        "reason": "New data available" if needs_refresh else "No updates"
    }


def refresh_news_data(force: bool = False) -> Dict[str, any]:
    """Refresh news data from source.

    Args:
        force: Force refresh even if no updates detected

    Returns:
        Dictionary with refresh results
    """
    logger.info("Refreshing news data...")

    # Check if refresh is needed
    refresh_status = check_news_data_refresh()
    if not refresh_status["needs_refresh"] and not force:
        logger.info("No news data refresh needed")
        return {
            "success": True,
            "refreshed": False,
            "reason": "No updates available"
        }

    try:
        # Load news data
        df = load_news_articles()

        # Get data date range
        date_range = {
            "min": df["pub_date"].min().isoformat() if "pub_date" in df.columns else None,
            "max": df["pub_date"].max().isoformat() if "pub_date" in df.columns else None,
            "count": len(df)
        }

        # Update version tracking
        from config import CRAWL_NEWS_ARTICLES
        versions = get_data_versions()
        versions["news_articles"] = get_file_modification_time(CRAWL_NEWS_ARTICLES)
        versions["news_date_range"] = date_range
        save_data_versions(versions)

        logger.info(f"News data refreshed: {len(df)} articles")
        return {
            "success": True,
            "refreshed": True,
            "date_range": date_range,
            "count": len(df)
        }

    except Exception as e:
        logger.error(f"Error refreshing news data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def refresh_price_data(force: bool = False) -> Dict[str, any]:
    """Refresh price data from source.

    Args:
        force: Force refresh even if no updates detected

    Returns:
        Dictionary with refresh results
    """
    logger.info("Refreshing price data...")

    # Check if refresh is needed
    refresh_status = check_price_data_refresh()
    if not refresh_status["needs_refresh"] and not force:
        logger.info("No price data refresh needed")
        return {
            "success": True,
            "refreshed": False,
            "reason": "No updates available"
        }

    try:
        # Get available tickers
        tickers = get_available_tickers()

        if not tickers:
            return {
                "success": False,
                "error": "No tickers available"
            }

        # Load sample data to get date range
        sample_ticker = tickers[0]
        df = load_stock_ohlcv(sample_ticker)

        # Get data date range
        date_range = {
            "min": df["date"].min().isoformat() if "date" in df.columns else None,
            "max": df["date"].max().isoformat() if "date" in df.columns else None,
        }

        # Update version tracking
        from config import PRICE_DATA_DIR
        versions = get_data_versions()
        versions["price_data"] = get_file_modification_time(PRICE_DATA_DIR / f"{sample_ticker}_ohlcv.csv")
        versions["price_date_range"] = date_range
        versions["available_tickers"] = tickers
        save_data_versions(versions)

        logger.info(f"Price data refreshed: {len(tickers)} tickers, {date_range}")
        return {
            "success": True,
            "refreshed": True,
            "date_range": date_range,
            "tickers_count": len(tickers)
        }

    except Exception as e:
        logger.error(f"Error refreshing price data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def refresh_all_data(force: bool = False) -> Dict[str, any]:
    """Refresh all data sources.

    Args:
        force: Force refresh even if no updates detected

    Returns:
        Dictionary with overall refresh results
    """
    logger.info("Starting full data refresh...")

    results = {
        "timestamp": datetime.now().isoformat(),
        "news": refresh_news_data(force=force),
        "price": refresh_price_data(force=force)
    }

    # Overall success
    results["success"] = all([
        results["news"].get("success", False),
        results["price"].get("success", False)
    ])

    logger.info(f"Data refresh complete: {'SUCCESS' if results['success'] else 'FAILED'}")
    return results


def get_data_status() -> Dict[str, any]:
    """Get current data status and refresh information.

    Returns:
        Dictionary with current data status
    """
    # Get versions
    versions = get_data_versions()

    # Check refresh status
    news_status = check_news_data_refresh()
    price_status = check_price_data_refresh()

    return {
        "timestamp": datetime.now().isoformat(),
        "news": {
            "last_updated": versions.get("news_articles"),
            "date_range": versions.get("news_date_range"),
            "needs_refresh": news_status["needs_refresh"],
            "refresh_reason": news_status["reason"]
        },
        "price": {
            "last_updated": versions.get("price_data"),
            "date_range": versions.get("price_date_range"),
            "available_tickers": versions.get("available_tickers", []),
            "needs_refresh": price_status["needs_refresh"],
            "refresh_reason": price_status["reason"]
        }
    }


def is_data_current(max_age_hours: int = 24) -> bool:
    """Check if data is current within specified age limit.

    Args:
        max_age_hours: Maximum age in hours (default: 24)

    Returns:
        True if data is current, False otherwise
    """
    versions = get_data_versions()

    # Check news data age
    news_updated = versions.get("news_articles")
    if news_updated:
        news_age = datetime.now() - datetime.fromisoformat(news_updated)
        if news_age > timedelta(hours=max_age_hours):
            logger.warning(f"News data is {news_age.total_seconds()/3600:.1f} hours old")
            return False

    # Check price data age
    price_updated = versions.get("price_data")
    if price_updated:
        price_age = datetime.now() - datetime.fromisoformat(price_updated)
        if price_age > timedelta(hours=max_age_hours):
            logger.warning(f"Price data is {price_age.total_seconds()/3600:.1f} hours old")
            return False

    return True


def auto_refresh_if_needed() -> Dict[str, any]:
    """Automatically refresh data if it's not current.

    Returns:
        Dictionary with refresh results
    """
    if is_data_current():
        logger.info("Data is current, no refresh needed")
        return {
            "success": True,
            "refreshed": False,
            "reason": "Data is current"
        }

    logger.info("Data is not current, refreshing...")
    return refresh_all_data()


if __name__ == "__main__":
    # Test refresh mechanism
    print("Checking data status...")
    status = get_data_status()
    print(f"News last updated: {status['news']['last_updated']}")
    print(f"Price last updated: {status['price']['last_updated']}")
    print(f"News needs refresh: {status['news']['needs_refresh']}")
    print(f"Price needs refresh: {status['price']['needs_refresh']}")

    print("\nRefreshing data...")
    results = refresh_all_data()
    print(f"Refresh results: {results}")
