"""Stock price data loading module.

This module enforces the CLAUDE.md rule: ALL price data MUST come from
D:\bmad-projects\stock_vol_prediction01\data\raw directory.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
from config import (
    PRICE_DATA_ROOT,
    PRICE_DATA_DIR,
    VN30_DIR,
    VN100_DIR,
    HNX_DIR,
    VN30_TICKERS,
)


def load_stock_ohlcv(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load OHLCV data for a specific stock from the primary data directory.

    Per CLAUDE.md rule: Price data is ONLY loaded from
    D:\bmad-projects\stock_vol_prediction01\data\raw

    Args:
        ticker: Stock ticker symbol (e.g., 'VCB', 'VIC')
        start_date: Filter from this date (YYYY-MM-DD)
        end_date: Filter until this date (YYYY-MM-DD)
        data_dir: Override default price data directory (for testing only)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume

    Raises:
        FileNotFoundError: If the price data directory or file doesn't exist
        ValueError: If ticker is not recognized
    """
    # Use default directory if not specified
    if data_dir is None:
        data_dir = PRICE_DATA_DIR

    if not PRICE_DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Primary price data source not found: {PRICE_DATA_ROOT}. "
            "Ensure the stock_vol_prediction01 project is accessible."
        )

    # Construct file path
    ticker = ticker.upper()
    filepath = data_dir / f"{ticker}_ohlcv.csv"

    if not filepath.exists():
        # Check if ticker is valid
        if ticker in VN30_TICKERS:
            raise FileNotFoundError(
                f"Price data file not found: {filepath}. "
                f"Ensure {ticker}_ohlcv.csv exists in {data_dir}"
            )
        else:
            raise ValueError(
                f"Unknown ticker: {ticker}. "
                f"Available VN30 tickers: {VN30_TICKERS}"
            )

    # Load data
    df = pd.read_csv(filepath)

    # Parse dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=False)
        df = df.sort_values("date")

    # Filter by date range if specified
    if start_date and "date" in df.columns:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date and "date" in df.columns:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df


def load_multiple_stocks(
    tickers: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple stocks.

    Args:
        tickers: List of ticker symbols
        start_date: Filter from this date (YYYY-MM-DD)
        end_date: Filter until this date (YYYY-MM-DD)

    Returns:
        Dictionary mapping ticker symbol to DataFrame
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = load_stock_ohlcv(ticker, start_date, end_date)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {ticker}: {e}")
        except ValueError as e:
            print(f"Warning: {e}")

    return results


def get_available_tickers() -> list[str]:
    """Get list of available stock tickers in the price data directory.

    Returns:
        List of ticker symbols that have OHLCV files
    """
    if not PRICE_DATA_DIR.exists():
        return []

    tickers = []
    for filepath in PRICE_DATA_DIR.glob("*_ohlcv.csv"):
        # Extract ticker from filename
        ticker = filepath.stem.replace("_ohlcv", "")
        tickers.append(ticker)

    return sorted(tickers)


def verify_price_data_integrity() -> dict[str, bool]:
    """Verify that the primary price data directory and files are accessible.

    Returns:
        Dictionary with verification status
    """
    checks = {
        "price_data_root": PRICE_DATA_ROOT.exists(),
        "prices_dir": PRICE_DATA_DIR.exists(),
        "vn30_dir": VN30_DIR.exists() if VN30_DIR else False,
        "vn100_dir": VN100_DIR.exists() if VN100_DIR else False,
        "hnx_dir": HNX_DIR.exists() if HNX_DIR else False,
    }

    # Check a sample ticker
    available_tickers = get_available_tickers()
    checks["has_price_files"] = len(available_tickers) > 0

    return checks


if __name__ == "__main__":
    # Test data loading
    print("Verifying price data integrity...")
    integrity = verify_price_data_integrity()
    for check, status in integrity.items():
        print(f"{check}: {'✓' if status else '✗'}")

    print("\nAvailable tickers:", get_available_tickers())

    # Load sample data
    print("\nLoading sample price data (VCB)...")
    try:
        df = load_stock_ohlcv("VCB")
        print(f"Total records: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nSample data:\n{df.head()}")
    except Exception as e:
        print(f"Error loading VCB data: {e}")
