"""News data loading module.

This module enforces the CLAUDE.md rule: ALL raw data MUST come from
D:\bmad-projects\crawl_data\data directory.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from config import (
    CRAWL_DATA_ROOT,
    CRAWL_NEWS_ARTICLES,
    CRAWL_CAFEF_ARTICLES,
    CRAWL_SSI_ARTICLES,
    CRAWL_VNDIRECT_ARTICLES,
    CRAWL_VNSTOCK_ARTICLES,
    CRAWL_HSC_ARTICLES,
)


def load_news_articles(
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load news articles from the primary crawl data directory.

    Per CLAUDE.md rule: Data is ONLY loaded from D:\bmad-projects\crawl_data\data

    Args:
        source: Filter by source ('cafef', 'ssi', 'vndirect', 'vnstock', 'hsc', or None for all)
        start_date: Filter articles from this date (YYYY-MM-DD)
        end_date: Filter articles until this date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: source, title, category, pub_date, url, author, lead, pdf_url, collected_at

    Raises:
        FileNotFoundError: If the crawl data directory or file doesn't exist
    """
    if not CRAWL_DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Primary data source not found: {CRAWL_DATA_ROOT}. "
            "Ensure the crawl_data project is accessible at this path."
        )

    # Load consolidated news or specific source
    if source is None:
        filepath = CRAWL_NEWS_ARTICLES
    else:
        source_map = {
            "cafef": CRAWL_CAFEF_ARTICLES,
            "ssi": CRAWL_SSI_ARTICLES,
            "vndirect": CRAWL_VNDIRECT_ARTICLES,
            "vnstock": CRAWL_VNSTOCK_ARTICLES,
            "hsc": CRAWL_HSC_ARTICLES,
        }
        filepath = source_map.get(source.lower())
        if filepath is None:
            raise ValueError(
                f"Unknown source: {source}. "
                f"Available: {list(source_map.keys())}"
            )

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            "Ensure the crawl project has generated this file."
        )

    # Load data
    df = pd.read_csv(filepath)

    # Parse dates (handle multiple formats and timezones)
    if "pub_date" in df.columns:
        df["pub_date"] = pd.to_datetime(df["pub_date"], format='mixed', dayfirst=True, utc=True)
        df["pub_date"] = df["pub_date"].dt.tz_convert(None)  # Convert to naive UTC
    if "collected_at" in df.columns:
        df["collected_at"] = pd.to_datetime(df["collected_at"], format='mixed', dayfirst=True, utc=True)
        df["collected_at"] = df["collected_at"].dt.tz_convert(None)  # Convert to naive UTC

    # Filter by date range if specified
    if start_date and "pub_date" in df.columns:
        df = df[df["pub_date"] >= pd.to_datetime(start_date)]
    if end_date and "pub_date" in df.columns:
        df = df[df["pub_date"] <= pd.to_datetime(end_date)]

    return df


def get_available_sources() -> list[str]:
    """Get list of available news sources in the crawl data directory.

    Returns:
        List of source names that have data files available
    """
    sources = []
    source_files = {
        "cafef": CRAWL_CAFEF_ARTICLES,
        "ssi": CRAWL_SSI_ARTICLES,
        "vndirect": CRAWL_VNDIRECT_ARTICLES,
        "vnstock": CRAWL_VNSTOCK_ARTICLES,
        "hsc": CRAWL_HSC_ARTICLES,
    }

    for source, filepath in source_files.items():
        if filepath.exists():
            sources.append(source)

    return sources


def verify_data_integrity() -> dict[str, bool]:
    """Verify that the primary crawl data directory and files are accessible.

    Returns:
        Dictionary with verification status for each data file
    """
    checks = {
        "crawl_data_root": CRAWL_DATA_ROOT.exists(),
        "news_articles_consolidated": CRAWL_NEWS_ARTICLES.exists(),
        "cafef_articles": CRAWL_CAFEF_ARTICLES.exists(),
        "ssi_articles": CRAWL_SSI_ARTICLES.exists(),
        "vndirect_articles": CRAWL_VNDIRECT_ARTICLES.exists(),
        "vnstock_articles": CRAWL_VNSTOCK_ARTICLES.exists(),
        "hsc_articles": CRAWL_HSC_ARTICLES.exists(),
    }
    return checks


if __name__ == "__main__":
    # Test data loading
    print("Verifying data integrity...")
    integrity = verify_data_integrity()
    for check, status in integrity.items():
        print(f"{check}: {'✓' if status else '✗'}")

    print("\nAvailable sources:", get_available_sources())

    # Load sample data
    print("\nLoading sample news data...")
    df = load_news_articles()
    print(f"Total articles: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['pub_date'].min()} to {df['pub_date'].max()}")
    print(f"\nSample article:\n{df.head(1).to_dict('records')[0]}")
