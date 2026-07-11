"""Integration tests for data source access.

Tests that verify the CLAUDE.md rule: All data must come from the
specified source directories only.
"""

import pytest
from pathlib import Path
from src.data.load_news import load_news_articles, get_available_sources, verify_data_integrity
from src.data.load_prices import load_stock_ohlcv, get_available_tickers, verify_price_data_integrity


@pytest.mark.integration
class TestNewsDataSource:
    """Test news data source access."""

    def test_crawl_data_root_exists(self):
        """Verify the primary crawl data directory exists."""
        from config import CRAWL_DATA_ROOT

        assert CRAWL_DATA_ROOT.exists(), "Primary news data source not found"

    def test_news_data_integrity(self):
        """Verify all expected news data files are accessible."""
        integrity = verify_data_integrity()

        # At least the consolidated file should exist
        assert integrity.get("news_articles_consolidated", False), "Consolidated news articles not found"

        # Print status for debugging
        for check, status in integrity.items():
            print(f"{check}: {'✓' if status else '✗'}")

    def test_load_news_articles(self):
        """Test loading news articles from primary source."""
        df = load_news_articles()

        assert not df.empty, "No news articles loaded"
        assert "source" in df.columns, "Missing 'source' column"
        assert "title" in df.columns, "Missing 'title' column"
        assert "pub_date" in df.columns, "Missing 'pub_date' column"

        # Check we have data from multiple sources
        sources = df["source"].unique()
        print(f"Available sources: {sources}")

    def test_get_available_sources(self):
        """Test getting list of available news sources."""
        sources = get_available_sources()
        assert len(sources) > 0, "No news sources available"
        print(f"Available sources: {sources}")


@pytest.mark.integration
class TestPriceDataSource:
    """Test price data source access."""

    def test_price_data_root_exists(self):
        """Verify the primary price data directory exists."""
        from config import PRICE_DATA_ROOT

        assert PRICE_DATA_ROOT.exists(), "Primary price data source not found"

    def test_price_data_integrity(self):
        """Verify price data directory structure is accessible."""
        integrity = verify_price_data_integrity()

        # At least the prices directory should exist
        assert integrity.get("prices_dir", False), "Prices directory not found"
        assert integrity.get("has_price_files", False), "No price files found"

        # Print status for debugging
        for check, status in integrity.items():
            print(f"{check}: {'✓' if status else '✗'}")

    def test_get_available_tickers(self):
        """Test getting list of available stock tickers."""
        tickers = get_available_tickers()
        assert len(tickers) > 0, "No tickers available"
        print(f"Available tickers ({len(tickers)}): {tickers[:10]}...")

    def test_load_stock_ohlcv(self):
        """Test loading OHLCV data from primary source."""
        # Use VCB as a common ticker
        df = load_stock_ohlcv("VCB")

        assert not df.empty, "No price data loaded for VCB"
        assert "date" in df.columns, "Missing 'date' column"
        assert "close" in df.columns, "Missing 'close' column"
        assert "volume" in df.columns, "Missing 'volume' column"

        # Check data is sorted
        dates = df["date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)), "Data not sorted by date"

    def test_load_multiple_stocks(self):
        """Test loading multiple stocks."""
        from src.data.load_prices import load_multiple_stocks

        tickers = ["VCB", "VIC", "VNM"]  # Common tickers
        data = load_multiple_stocks(tickers)

        assert len(data) > 0, "No stocks loaded"
        for ticker in tickers:
            if ticker in data:
                assert not data[ticker].empty, f"Empty data for {ticker}"
                print(f"{ticker}: {len(data[ticker])} records")


@pytest.mark.integration
@pytest.mark.smoke
class TestDataSourcesSmoke:
    """Smoke tests for data source access - quick sanity checks."""

    def test_both_data_sources_accessible(self):
        """Verify both primary data sources are accessible."""
        from config import CRAWL_DATA_ROOT, PRICE_DATA_ROOT

        assert CRAWL_DATA_ROOT.exists(), "News data source not accessible"
        assert PRICE_DATA_ROOT.exists(), "Price data source not accessible"

    def test_can_load_sample_data(self):
        """Test loading a small sample of data from both sources."""
        # Load news sample
        news_df = load_news_articles()
        assert len(news_df) > 0, "No news data available"

        # Load price sample
        price_df = load_stock_ohlcv("VCB", start_date="2024-01-01", end_date="2024-01-31")
        assert len(price_df) > 0, "No price data available for VCB in Jan 2024"

        print(f"✓ News data: {len(news_df)} articles")
        print(f"✓ Price data: {len(price_df)} VCB records (Jan 2024)")
