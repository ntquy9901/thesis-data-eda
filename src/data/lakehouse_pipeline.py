"""Data Lakehouse Pipeline for Vietnam Stock Analysis.

Implements Bronze/Silver/Gold data processing pipeline with:
- Data import from sources to Bronze
- Data cleaning and validation to Silver
- Feature engineering to Gold
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LakehousePipeline:
    """Main pipeline for processing data through Bronze/Silver/Gold layers."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the lakehouse pipeline.

        Args:
            project_root: Project root directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = project_root
        self.lakehouse_root = project_root / "data_lakehouse"

        # Layer directories
        self.bronze_dir = self.lakehouse_root / "bronze"
        self.silver_dir = self.lakehouse_root / "silver"
        self.gold_dir = self.lakehouse_root / "gold"

        # Metadata directories
        self.metadata_dir = self.lakehouse_root / "_metadata"
        self.logs_dir = self.lakehouse_root / "_logs"
        self.checkpoint_dir = self.lakehouse_root / "_checkpoint"

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.bronze_dir / "news",
            self.bronze_dir / "prices",
            self.silver_dir / "news",
            self.silver_dir / "prices",
            self.gold_dir / "news",
            self.gold_dir / "prices",
            self.gold_dir / "correlation",
            self.metadata_dir,
            self.logs_dir,
            self.checkpoint_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Lakehouse directories created/verified")

    def log_pipeline_run(self, stage: str, status: str, metadata: Dict = None):
        """Log pipeline execution for tracking and monitoring.

        Args:
            stage: Pipeline stage name
            status: Execution status (success, failed, partial)
            metadata: Additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "metadata": metadata or {}
        }

        log_file = self.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"Pipeline run logged: {stage} - {status}")


class BronzeImporter(LakehousePipeline):
    """Import raw data from sources to Bronze layer."""

    def import_news_to_bronze(self, force: bool = False) -> Dict[str, any]:
        """Import news articles from source to Bronze layer.

        Args:
            force: Force re-import even if data exists

        Returns:
            Import results dictionary
        """
        logger.info("Importing news to Bronze layer...")

        try:
            from src.data.load_news import load_news_articles
            from config import CRAWL_NEWS_ARTICLES

            # Check if already imported
            bronze_file = self.bronze_dir / "news" / "news_articles.csv"
            if bronze_file.exists() and not force:
                logger.info("Bronze news already exists, skipping import")
                return {
                    "success": True,
                    "imported": False,
                    "reason": "Already exists"
                }

            # Load from source
            df = load_news_articles()

            # Save to Bronze
            df.to_csv(bronze_file, index=False)

            # Log import
            self.log_pipeline_run("bronze_news_import", "success", {
                "rows": len(df),
                "columns": len(df.columns),
                "source_file": str(CRAWL_NEWS_ARTICLES)
            })

            logger.info(f"Imported {len(df)} news articles to Bronze")
            return {
                "success": True,
                "imported": True,
                "rows": len(df),
                "columns": len(df.columns)
            }

        except Exception as e:
            logger.error(f"Error importing news to Bronze: {e}")
            self.log_pipeline_run("bronze_news_import", "failed", {"error": str(e)})
            return {"success": False, "error": str(e)}

    def import_prices_to_bronze(self, force: bool = False) -> Dict[str, any]:
        """Import price data from source to Bronze layer.

        Args:
            force: Force re-import even if data exists

        Returns:
            Import results dictionary
        """
        logger.info("Importing prices to Bronze layer...")

        try:
            from src.data.load_prices import get_available_tickers, load_stock_ohlcv
            from config import PRICE_DATA_DIR

            # Get available tickers
            tickers = get_available_tickers()

            if not tickers:
                return {"success": False, "error": "No tickers available"}

            imported_count = 0
            total_rows = 0

            for ticker in tickers:
                bronze_file = self.bronze_dir / "prices" / f"{ticker}_ohlcv.csv"

                # Check if already imported
                if bronze_file.exists() and not force:
                    logger.info(f"Bronze price data for {ticker} already exists, skipping")
                    continue

                # Load from source
                df = load_stock_ohlcv(ticker)

                # Add ticker column if not present
                if 'ticker' not in df.columns:
                    df['ticker'] = ticker

                # Save to Bronze
                df.to_csv(bronze_file, index=False)

                imported_count += 1
                total_rows += len(df)
                logger.info(f"Imported {len(df)} rows for {ticker}")

            # Log import
            self.log_pipeline_run("bronze_prices_import", "success", {
                "tickers_imported": imported_count,
                "total_rows": total_rows,
                "total_tickers": len(tickers)
            })

            logger.info(f"Imported {imported_count} tickers ({total_rows} rows) to Bronze")
            return {
                "success": True,
                "imported": True,
                "tickers_imported": imported_count,
                "total_rows": total_rows,
                "total_tickers": len(tickers)
            }

        except Exception as e:
            logger.error(f"Error importing prices to Bronze: {e}")
            self.log_pipeline_run("bronze_prices_import", "failed", {"error": str(e)})
            return {"success": False, "error": str(e)}


class SilverProcessor(LakehousePipeline):
    """Process Bronze data to Silver layer (cleaned & validated)."""

    def process_news_to_silver(self) -> Dict[str, any]:
        """Process news from Bronze to Silver layer.

        Returns:
            Processing results dictionary
        """
        logger.info("Processing news to Silver layer...")

        try:
            bronze_file = self.bronze_dir / "news" / "news_articles.csv"

            if not bronze_file.exists():
                return {"success": False, "error": "Bronze news data not found"}

            # Load from Bronze
            df = pd.read_csv(bronze_file)

            # Clean and process
            df_cleaned = self._clean_news_data(df)

            # Save to Silver (Parquet format)
            silver_file = self.silver_dir / "news" / "news_cleaned.parquet"
            df_cleaned.to_parquet(silver_file, index=False)

            # Log processing
            self.log_pipeline_run("silver_news_processing", "success", {
                "input_rows": len(df),
                "output_rows": len(df_cleaned),
                "quality_score": self._calculate_quality_score(df_cleaned)
            })

            logger.info(f"Processed {len(df_cleaned)} news articles to Silver")
            return {
                "success": True,
                "processed": True,
                "input_rows": len(df),
                "output_rows": len(df_cleaned)
            }

        except Exception as e:
            logger.error(f"Error processing news to Silver: {e}")
            self.log_pipeline_run("silver_news_processing", "failed", {"error": str(e)})
            return {"success": False, "error": str(e)}

    def process_prices_to_silver(self) -> Dict[str, any]:
        """Process prices from Bronze to Silver layer.

        Returns:
            Processing results dictionary
        """
        logger.info("Processing prices to Silver layer...")

        try:
            bronze_dir = self.bronze_dir / "prices"

            if not bronze_dir.exists():
                return {"success": False, "error": "Bronze prices directory not found"}

            # Process each ticker
            all_data = []

            for bronze_file in bronze_dir.glob("*_ohlcv.csv"):
                ticker = bronze_file.stem.replace("_ohlcv", "")

                # Load from Bronze
                df = pd.read_csv(bronze_file)

                # Clean and process
                df_cleaned = self._clean_price_data(df, ticker)

                all_data.append(df_cleaned)
                logger.info(f"Processed {len(df_cleaned)} rows for {ticker}")

            # Combine all data
            df_combined = pd.concat(all_data, ignore_index=True)

            # Save to Silver
            silver_file = self.silver_dir / "prices" / "prices_cleaned.parquet"
            df_combined.to_parquet(silver_file, index=False)

            # Log processing
            self.log_pipeline_run("silver_prices_processing", "success", {
                "input_rows": sum(len(d) for d in all_data),
                "output_rows": len(df_combined),
                "tickers_processed": len(all_data)
            })

            logger.info(f"Processed {len(df_combined)} price rows to Silver")
            return {
                "success": True,
                "processed": True,
                "output_rows": len(df_combined),
                "tickers_processed": len(all_data)
            }

        except Exception as e:
            logger.error(f"Error processing prices to Silver: {e}")
            self.log_pipeline_run("silver_prices_processing", "failed", {"error": str(e)})
            return {"success": False, "error": str(e)}

    def _clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process news data.

        Args:
            df: Raw news DataFrame

        Returns:
            Cleaned news DataFrame
        """
        df = df.copy()

        # Generate unique article ID
        df['article_id'] = df['source'] + '_' + df['pub_date'].astype(str) + '_' + df.index.astype(str)

        # Normalize dates
        df['pub_date'] = pd.to_datetime(df['pub_date'], utc=False)

        # Combine title and lead for content
        df['content'] = df['title'].fillna('') + ' ' + df['lead'].fillna('')
        df['content'] = df['content'].str.strip()

        # Calculate word count
        df['word_count'] = df['content'].str.split().str.len()

        # Extract tickers mentioned (basic implementation)
        df['tickers_mentioned'] = df['content'].apply(self._extract_tickers)

        # Add processing timestamp
        df['processed_at'] = datetime.now()

        # Calculate quality score (basic implementation)
        df['quality_score'] = df.apply(self._calculate_news_quality_score, axis=1)

        return df

    def _clean_price_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and process price data.

        Args:
            df: Raw price DataFrame
            ticker: Stock ticker symbol

        Returns:
            Cleaned price DataFrame
        """
        df = df.copy()

        # Add ticker column
        df['ticker'] = ticker

        # Normalize dates
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Calculate volatility indicators
        df['volatility_20d'] = df['returns'].rolling(window=20).std()
        df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()

        # Add processing timestamp
        df['processed_at'] = datetime.now()

        # Calculate quality score
        df['quality_score'] = df.apply(self._calculate_price_quality_score, axis=1)

        return df

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text (basic implementation).

        Args:
            text: Text content

        Returns:
            List of ticker symbols found
        """
        from config import VN30_TICKERS

        tickers_found = []
        text_upper = text.upper()

        for ticker in VN30_TICKERS:
            if ticker in text_upper:
                tickers_found.append(ticker)

        return tickers_found

    def _calculate_news_quality_score(self, row: pd.Series) -> float:
        """Calculate quality score for news article.

        Args:
            row: News article row

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Deduct for missing fields
        if pd.isna(row['title']) or row['title'] == '':
            score -= 0.3
        if pd.isna(row['pub_date']):
            score -= 0.3
        if pd.isna(row['url']) or row['url'] == '':
            score -= 0.2
        if row.get('word_count', 0) < 10:
            score -= 0.2

        return max(0.0, score)

    def _calculate_price_quality_score(self, row: pd.Series) -> float:
        """Calculate quality score for price data.

        Args:
            row: Price data row

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Deduct for missing or invalid values
        if pd.isna(row['close']) or row['close'] <= 0:
            score -= 0.4
        if pd.isna(row['volume']) or row['volume'] < 0:
            score -= 0.3
        if row['high'] < row['low']:
            score -= 0.3

        return max(0.0, score)

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall quality score for dataset.

        Args:
            df: DataFrame to calculate quality for

        Returns:
            Average quality score
        """
        if 'quality_score' in df.columns:
            return df['quality_score'].mean()
        return 0.0


if __name__ == "__main__":
    # Test the pipeline
    pipeline = LakehousePipeline()

    print("Testing Bronze Import...")
    bronze_importer = BronzeImporter()

    news_result = bronze_importer.import_news_to_bronze()
    print(f"News import: {news_result}")

    prices_result = bronze_importer.import_prices_to_bronze()
    print(f"Prices import: {prices_result}")

    print("\nTesting Silver Processing...")
    silver_processor = SilverProcessor()

    news_silver_result = silver_processor.process_news_to_silver()
    print(f"News processing: {news_silver_result}")

    prices_silver_result = silver_processor.process_prices_to_silver()
    print(f"Prices processing: {prices_silver_result}")
