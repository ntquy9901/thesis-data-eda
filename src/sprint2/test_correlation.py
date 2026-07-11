"""Test Correlation Analysis with Sample Data

This script creates sample data to test the correlation analysis functionality
when actual price data files are not available.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# Import the correlation analyzer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from sprint2.task2_1_correlation_analysis import NewsPriceCorrelationAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_price_data(days: int = 100) -> pd.DataFrame:
    """Create sample price data for testing.

    Args:
        days: Number of days of data to create

    Returns:
        DataFrame with sample price data
    """
    logger.info(f"Creating {days} days of sample price data...")

    # Create date range
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days, 0, -1)]

    data = []
    for date in dates:
        # Simulate price movements with some randomness
        base_price = 10000
        daily_return = np.random.normal(0.001, 0.02)  # Daily returns with mean 0.1% and 2% volatility
        volume = np.random.normal(1000000, 200000)  # Random trading volume

        data.append({
            'Date': date,
            'Open': base_price * (1 + np.random.normal(0, 0.005)),
            'High': base_price * (1 + abs(np.random.normal(0, 0.01))),
            'Low': base_price * (1 - abs(np.random.normal(0, 0.01))),
            'Close': base_price * (1 + daily_return),
            'Volume': max(0, volume),
            'Ticker': 'VN30'  # Use a generic ticker
        })

    df = pd.DataFrame(data)
    logger.info(f"Created sample price data: {len(df)} records")
    return df


def create_sample_news_data(days: int = 100) -> pd.DataFrame:
    """Create sample news data for testing.

    Args:
        days: Number of days of data to create

    Returns:
        DataFrame with sample news data
    """
    logger.info(f"Creating {days} days of sample news data...")

    # Create date range
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days, 0, -1)]

    data = []
    for date in dates:
        # Simulate news articles with varying sentiment
        num_articles = np.random.poisson(3)  # Average 3 articles per day

        for i in range(num_articles):
            # Generate random sentiment
            sentiment = np.random.normal(0, 0.5)  # Sentiment centered around 0
            sentiment = max(-1, min(1, sentiment))  # Clamp between -1 and 1

            # Create title based on sentiment
            if sentiment > 0.3:
                title = "Cổ phiếu tăng trưởng mạnh trong phiên hôm nay"
            elif sentiment < -0.3:
                title = "Thị trường gặp khó khăn do rủi ro kinh tế"
            else:
                title = "Giao dịch diễn ra với khối lượng trung bình"

            data.append({
                'pub_date': date,
                'title': title,
                'lead': f"Bài viết số {i+1} với sentiment {sentiment:.2f}",
                'sentiment_score': sentiment,
                'source': 'test_source',
                'url': f'http://test.example.com/article/{date.strftime("%Y%m%d")}_{i}'
            })

    df = pd.DataFrame(data)
    logger.info(f"Created sample news data: {len(df)} articles")
    return df


def test_correlation_analysis():
    """Test the correlation analysis with sample data."""
    print("="*60)
    print("Testing Correlation Analysis with Sample Data")
    print("="*60)

    # Create sample data
    print("\n1. Creating sample data...")
    news_df = create_sample_news_data(days=100)
    price_df = create_sample_price_data(days=100)

    # Initialize analyzer
    print("\n2. Initializing correlation analyzer...")
    analyzer = NewsPriceCorrelationAnalyzer()

    # Prepare data
    print("\n3. Preparing data for analysis...")
    news_data = analyzer.prepare_news_data(news_df)
    price_data = analyzer.prepare_price_data(price_df)

    print(f"   - News data: {len(news_data)} days")
    print(f"   - Price data: {len(price_data)} days")

    # Calculate correlations
    print("\n4. Calculating correlations...")
    correlation_results = analyzer.calculate_correlation(news_data, price_data)

    # Analyze sentiment days
    print("\n5. Analyzing high/low sentiment days...")
    sentiment_analysis = analyzer.analyze_high_sentiment_days(news_data, price_data)

    # Generate report
    print("\n6. Generating analysis report...")
    report = analyzer.generate_correlation_report(correlation_results, sentiment_analysis)

    # Display results
    print("\n7. Analysis Results:")
    print("="*60)

    if correlation_results.get('same_day_correlation'):
        same_day = correlation_results['same_day_correlation']
        print(f"Same-day correlation: {same_day['correlation']:.3f}")
        print(f"  P-value: {same_day['p_value']:.3f}")
        print(f"  Significant: {same_day['significant']}")

    if correlation_results.get('next_day_correlation'):
        next_day = correlation_results['next_day_correlation']
        print(f"Next-day correlation: {next_day['correlation']:.3f}")
        print(f"  P-value: {next_day['p_value']:.3f}")
        print(f"  Significant: {next_day['significant']}")

    print("\n8. Key Findings:")
    if 'summary' in report:
        for finding in report['summary'].get('key_findings', ['No significant findings']):
            print(f"   - {finding}")

    print("\n" + "="*60)
    print("Correlation Analysis Test: COMPLETE [OK]")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = test_correlation_analysis()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)