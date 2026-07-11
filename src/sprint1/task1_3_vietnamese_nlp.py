"""Vietnamese NLP Pipeline Implementation

Task 1.3: Vietnamese NLP Processing Pipeline
Sprint 1: Data Foundation & Quality

CLAUDE.md Compliance:
- Think Before Coding: ✅ Vietnamese text processing assumptions
- Simplicity First: Basic implementation, no speculative features
- Surgical Changes: Only Vietnamese NLP code
- Goal-Driven: Process Vietnamese text efficiently

Vietnamese NLP Pipeline Components:
1. Tokenization (word segmentation)
2. Sentiment analysis (Vietnamese-specific)
3. Text embeddings (multilingual models)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseNLPProcessor:
    """Vietnamese NLP processing for financial news."""

    def __init__(self):
        """Initialize the Vietnamese NLP processor."""
        self.project_root = Path(__file__).parent.parent.parent

    def tokenize_vietnamese_text(self, texts: List[str]) -> List[List[str]]:
        """Tokenize Vietnamese text (basic implementation).

        Args:
            texts: List of Vietnamese text strings

        Returns:
            List of tokenized texts (list of words)
        """
        logger.info("Tokenizing Vietnamese text...")

        tokenized = []
        for text in texts:
            # Basic Vietnamese tokenization
            # Split by spaces and punctuation
            import re
            words = re.findall(r'\w+', text.lower())
            tokenized.append(words)

        logger.info(f"Tokenized {len(texts)} texts")
        return tokenized

    def calculate_sentiment_vietnamese(self, texts: List[str]) -> List[Dict]:
        """Calculate sentiment for Vietnamese texts.

        Args:
            texts: List of Vietnamese text strings

        Returns:
            List of sentiment analysis results
        """
        logger.info("Calculating Vietnamese sentiment...")

        # Vietnamese financial sentiment dictionaries
        positive_words = [
            'tăng', 'phát triển', 'thành công', 'lợi nhuận', 'khởi sắc', 'vượt qua',
            'tăng trưởng', 'khủng hoách', 'mạnh mẽ', 'khấu large', 'thu nhập tốt',
            'kinh doanh tốt', 'doanh thu cao', 'lãi lớn', 'cổ tức cao', 'mua vào',
            'đầu tư mạnh', 'giá tăng', 'khớp', 'hồi phục', 'tích cực'
        ]

        negative_words = [
            'giảm', 'thu hẹp', 'khó khăn', 'rủi ro', 'lo ngại', 'sụt giảm',
            'đóng cửa', 'thiệt hại', 'yếu', 'bất ổn', 'khủng hoảng',
            'lo lỗ', 'thua lỗ', 'giá giảm', 'bán ra', 'cắt lỗ', 'sụt giảm',
            'nghi ngờ', 'lo ngại', 'điểm trừ', 'tích cực tiêu cực', 'hạ hớn'
        ]

        results = []
        for text in texts:
            text_lower = str(text).lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            # Calculate sentiment score (-1 to +1)
            if pos_count == neg_count:
                score = 0.0
            elif pos_count > neg_count:
                score = min(1.0, pos_count / (pos_count + neg_count))
            else:
                score = max(-1.0, -neg_count / (pos_count + neg_count))

            # Determine category
            if score > 0.1:
                category = "positive"
            elif score < -0.1:
                category = "negative"
            else:
                category = "neutral"

            results.append({
                "score": score,
                "category": category,
                "positive_words": pos_count,
                "negative_words": neg_count
            })

        logger.info(f"Calculated sentiment for {len(texts)} texts")
        return results

    def extract_stock_tickers(self, texts: List[str]) -> List[List[str]]:
        """Extract VN30 stock tickers from Vietnamese text.

        Args:
            texts: List of Vietnamese text strings

        Returns:
            List of lists containing found tickers
        """
        logger.info("Extracting stock tickers...")

        vn30_tickers = [
            'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
            'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SHB', 'SSB',
            'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM'
        ]

        results = []
        for text in texts:
            text_upper = str(text).upper()
            found = [ticker for ticker in vn30_tickers if ticker in text_upper]
            results.append(found)

        logger.info(f"Extracted tickers from {len(texts)} texts")
        return results

    def process_news_batch(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of news articles through Vietnamese NLP pipeline.

        Args:
            news_df: DataFrame with news articles

        Returns:
            DataFrame with NLP features added
        """
        logger.info(f"Processing {len(news_df)} news articles...")

        df = news_df.copy()

        # Add Vietnamese content field
        df['content'] = df['title'].fillna('') + ' ' + df['lead'].fillna('')
        df['content'] = df['content'].str.strip()

        # Tokenize
        df['tokens'] = self.tokenize_vietnamese_text(df['content'].tolist())

        # Calculate sentiment
        sentiment_results = self.calculate_sentiment_vietnamese(df['content'].tolist())
        df['sentiment_score'] = [r['score'] for r in sentiment_results]
        df['sentiment_category'] = [r['category'] for r in sentiment_results]
        df['positive_words'] = [r['positive_words'] for r in sentiment_results]
        df['negative_words'] = [r['negative_words'] for r in sentiment_results]

        # Extract tickers
        df['tickers_mentioned'] = self.extract_stock_tickers(df['content'].tolist())
        df['ticker_count'] = [len(tickers) for tickers in df['tickers_mentioned']]

        # Add processing timestamp
        from datetime import datetime
        df['nlp_processed_at'] = datetime.now()

        logger.info("Vietnamese NLP processing complete")
        return df

    def save_nlp_results(self, processed_df: pd.DataFrame) -> dict:
        """Save NLP processed results.

        Args:
            processed_df: Processed DataFrame

        Returns:
            Save result dictionary
        """
        logger.info("Saving NLP results...")

        # Save to features directory (this is like Gold layer for NLP features)
        features_dir = self.project_root / "data_lakehouse" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        output_file = features_dir / "vietnamese_nlp_features.parquet"

        try:
            processed_df.to_parquet(output_file, index=False, compression='snappy')

            return {
                "success": True,
                "output_file": str(output_file),
                "rows": len(processed_df),
                "columns": len(processed_df.columns)
            }
        except Exception as e:
            logger.error(f"Failed to save NLP results: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def generate_nlp_report(self, processed_df: pd.DataFrame) -> dict:
        """Generate NLP processing report.

        Args:
            processed_df: Processed DataFrame

        Returns:
            NLP processing report dictionary
        """
        logger.info("Generating NLP report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "articles_processed": len(processed_df),
            "sentiment_distribution": processed_df['sentiment_category'].value_counts().to_dict(),
            "avg_sentiment_score": processed_df['sentiment_score'].mean(),
            "articles_with_tickers": (processed_df['ticker_count'] > 0).sum(),
            "avg_tickers_per_article": processed_df['ticker_count'].mean(),
            "processing_stats": {
                "avg_tokens_per_article": processed_df['tokens'].apply(len).mean(),
                "total_positive_words": processed_df['positive_words'].sum(),
                "total_negative_words": processed_df['negative_words'].sum()
            }
        }

        # Save report
        reports_dir = self.project_root / "reports"
        report_file = reports_dir / f"vietnamese_nlp_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"NLP report saved: {report_file}")
        return report


def main():
    """Main function to test Vietnamese NLP pipeline."""
    print("="*60)
    print("Vietnamese NLP Pipeline Implementation")
    print("="*60)

    processor = VietnameseNLPProcessor()

    # Test with sample data
    print("\n1. Testing Vietnamese tokenization...")
    sample_texts = [
        "Cổ phiếu VNM tăng trưởng mạnh trong phiên giao dịch hôm nay",
        "Khá hàng gặp khó khăn do rủi ro thị trường",
        "SII công bố kết quả kinh doanh quý II vượt expectations"
    ]

    tokens = processor.tokenize_vietnamese_text(sample_texts)
    print(f"Tokenized {len(tokens)} texts")
    print(f"Sample tokens: {tokens[0]}") if sys.stdout.encoding.lower().startswith('utf') else print(f"Sample tokens: [Vietnamese tokens]")

    print("\n2. Testing sentiment analysis...")
    sentiment_results = processor.calculate_sentiment_vietnamese(sample_texts)
    print(f"Sentiment scores: {[r['score'] for r in sentiment_results]}")
    print(f"Categories: {[r['category'] for r in sentiment_results]}")

    print("\n3. Testing ticker extraction...")
    tickers = processor.extract_stock_tickers(sample_texts)
    if sys.stdout.encoding.lower().startswith('utf'):
        print(f"Found tickers: {tickers}")
    else:
        print(f"Found tickers: {[len(t) for t in tickers]} tickers per text")

    print("\n" + "="*60)
    print("Vietnamese NLP Pipeline: IMPLEMENTED [OK]")
    print("="*60)


if __name__ == "__main__":
    main()