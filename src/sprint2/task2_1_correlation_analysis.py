"""News-Price Correlation Analysis Implementation

Task 2.1: News-Price Correlation Analysis
Sprint 2: Core Analysis Pipeline

CLAUDE.md Compliance:
- Think Before Coding: ✅ Correlation analysis assumptions stated
- Simplicity First: Statistical correlation, no complex ML models
- Surgical Changes: Only correlation analysis code
- Goal-Driven: Identify relationships between news sentiment and price movements

Correlation Analysis Features:
1. News sentiment vs price movement correlation
2. Volume vs news volume correlation
3. Temporal analysis (immediate vs delayed effects)
4. Statistical significance testing
5. Ticker-specific analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import sys
from typing import Dict, List, Tuple
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsPriceCorrelationAnalyzer:
    """News-price correlation analysis for Vietnam stock market."""

    def __init__(self):
        """Initialize the correlation analyzer."""
        self.project_root = Path(__file__).parent.parent.parent
        self.silver_dir = self.project_root / "data_lakehouse" / "silver" / "news"
        self.features_dir = self.project_root / "data_lakehouse" / "features"
        self.price_data_dir = self.project_root / "D:"

        logger.info("News-Price Correlation Analyzer initialized")

    def load_processed_news(self) -> pd.DataFrame:
        """Load processed news data with sentiment.

        Returns:
            DataFrame with processed news
        """
        logger.info("Loading processed news data...")

        # Try to load from Silver layer
        silver_file = self.silver_dir / "news_cleaned.parquet"

        if silver_file.exists():
            df = pd.read_parquet(silver_file)
            logger.info(f"Loaded {len(df)} news articles from Silver layer")
            return df

        # Fallback: load from Bronze and process
        logger.warning("Silver data not found, loading from Bronze")
        bronze_file = self.project_root / "data_lakehouse" / "bronze" / "news" / "news_articles.csv"

        if bronze_file.exists():
            df = pd.read_csv(bronze_file)
            logger.info(f"Loaded {len(df)} news articles from Bronze layer")

            # Basic processing
            df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
            df['content'] = df['title'].fillna('') + ' ' + df['lead'].fillna('')

            return df

        raise FileNotFoundError("No processed news data found")

    def load_price_data(self) -> pd.DataFrame:
        """Load stock price data.

        Returns:
            DataFrame with price data
        """
        logger.info("Loading price data...")

        price_data = []

        # Look for price data in the primary directory
        price_dir = Path("D:/")

        # Try to find CSV files with stock data
        csv_files = list(price_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Check if this looks like stock price data
                if all(col in df.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
                    # Extract ticker from filename
                    ticker = csv_file.stem
                    df['Ticker'] = ticker
                    price_data.append(df)
                    logger.info(f"Loaded price data for {ticker}")
            except Exception as e:
                logger.debug(f"Skipping {csv_file}: {e}")

        if not price_data:
            raise FileNotFoundError("No price data files found")

        combined_df = pd.concat(price_data, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')

        logger.info(f"Loaded {len(combined_df)} price records")
        return combined_df

    def prepare_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare news data for correlation analysis.

        Args:
            news_df: Raw news DataFrame

        Returns:
            Prepared news DataFrame
        """
        logger.info("Preparing news data for analysis...")

        df = news_df.copy()

        # Ensure we have the required columns
        if 'pub_date' not in df.columns:
            df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

        # Calculate basic sentiment if not present
        if 'sentiment_score' not in df.columns:
            logger.info("Calculating basic sentiment scores...")

            positive_words = ['tăng', 'phát triển', 'thành công', 'lợi nhuận', 'khởi sắc', 'vượt qua', 'tăng trưởng']
            negative_words = ['giảm', 'thu hẹp', 'khó khăn', 'rủi ro', 'lo ngại', 'sụt giảm', 'đóng cửa']

            def calculate_sentiment(text):
                text_lower = str(text).lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)

                if pos_count == neg_count:
                    return 0.0
                elif pos_count > neg_count:
                    return min(1.0, pos_count / (pos_count + neg_count))
                else:
                    return max(-1.0, -neg_count / (pos_count + neg_count))

            df['content'] = df['title'].fillna('') + ' ' + df['lead'].fillna('')
            df['sentiment_score'] = df['content'].apply(calculate_sentiment)

        # Extract date only for grouping
        df['date_only'] = df['pub_date'].dt.date

        # Group by date and calculate daily sentiment
        daily_sentiment = df.groupby('date_only').agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'title': 'count'
        }).reset_index()

        daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_count', 'sentiment_std', 'news_count']

        logger.info(f"Prepared {len(daily_sentiment)} days of news data")
        return daily_sentiment

    def prepare_price_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare price data for correlation analysis.

        Args:
            price_df: Raw price DataFrame

        Returns:
            Prepared price DataFrame
        """
        logger.info("Preparing price data for analysis...")

        df = price_df.copy()

        # Ensure we have the required columns
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Calculate daily returns
        df['date_only'] = df['Date'].dt.date

        # Group by date and calculate daily metrics
        daily_prices = df.groupby(['date_only', 'Ticker']).agg({
            'Close': 'last',
            'Open': 'first',
            'Volume': 'sum'
        }).reset_index()

        # Calculate returns
        daily_prices['daily_return'] = daily_prices.groupby('Ticker')['Close'].pct_change()

        # Aggregate across all tickers
        overall_daily = daily_prices.groupby('date_only').agg({
            'daily_return': 'mean',
            'Volume': 'sum'
        }).reset_index()

        overall_daily.columns = ['date', 'avg_return', 'total_volume']

        logger.info(f"Prepared {len(overall_daily)} days of price data")
        return overall_daily

    def calculate_correlation(self, news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Calculate correlation between news sentiment and price movements.

        Args:
            news_data: Daily news sentiment data
            price_data: Daily price data

        Returns:
            Dictionary with correlation results
        """
        logger.info("Calculating news-price correlation...")

        # Merge datasets on date
        merged = pd.merge(news_data, price_data, on='date', how='inner')

        if len(merged) < 3:
            logger.warning(f"Not enough data points for correlation: {len(merged)}")
            return {
                "success": False,
                "error": "Insufficient data for correlation analysis",
                "data_points": len(merged)
            }

        results = {}

        # 1. Same-day correlation
        if 'avg_sentiment' in merged.columns and 'avg_return' in merged.columns:
            # Create clean dataset for correlation
            clean_data = merged[['avg_sentiment', 'avg_return']].dropna()
            if len(clean_data) >= 3:
                corr, p_value = stats.pearsonr(
                    clean_data['avg_sentiment'],
                    clean_data['avg_return']
                )
            else:
                corr, p_value = np.nan, np.nan
            results['same_day_correlation'] = {
                'correlation': float(corr) if not np.isnan(corr) else None,
                'p_value': float(p_value) if not np.isnan(p_value) else None,
                'significant': bool(p_value < 0.05) if not np.isnan(p_value) else None,
                'data_points': int(len(merged))
            }

        # 2. Lagged correlation (news -> next day price)
        if len(merged) > 1:
            merged_lagged = merged.copy()
            merged_lagged['next_day_return'] = merged_lagged['avg_return'].shift(-1)

            clean_lagged = merged_lagged[['avg_sentiment', 'next_day_return']].dropna()
            if len(clean_lagged) >= 3:
                corr_lag, p_value_lag = stats.pearsonr(
                    clean_lagged['avg_sentiment'],
                    clean_lagged['next_day_return']
                )
            else:
                corr_lag, p_value_lag = np.nan, np.nan
            results['next_day_correlation'] = {
                'correlation': float(corr_lag) if not np.isnan(corr_lag) else None,
                'p_value': float(p_value_lag) if not np.isnan(p_value_lag) else None,
                'significant': bool(p_value_lag < 0.05) if not np.isnan(p_value_lag) else None,
                'data_points': int(len(clean_lagged)) if 'clean_lagged' in locals() else int(len(merged_lagged.dropna()))
            }

        # 3. News volume vs trading volume correlation
        if 'news_count' in merged.columns and 'total_volume' in merged.columns:
            clean_volume = merged[['news_count', 'total_volume']].dropna()
            if len(clean_volume) >= 3:
                corr_vol, p_value_vol = stats.pearsonr(
                    clean_volume['news_count'],
                    clean_volume['total_volume']
                )
            else:
                corr_vol, p_value_vol = np.nan, np.nan
            results['volume_correlation'] = {
                'correlation': float(corr_vol) if not np.isnan(corr_vol) else None,
                'p_value': float(p_value_vol) if not np.isnan(p_value_vol) else None,
                'significant': bool(p_value_vol < 0.05) if not np.isnan(p_value_vol) else None,
                'data_points': int(len(clean_volume)) if 'clean_volume' in locals() else int(len(merged))
            }

        # 4. Sentiment volatility vs return volatility
        if 'sentiment_std' in merged.columns:
            merged['sentiment_volatility'] = merged['sentiment_std'].fillna(0)
            merged['return_volatility'] = merged['avg_return'].abs()

            clean_volatility = merged[['sentiment_volatility', 'return_volatility']].dropna()
            if len(clean_volatility) >= 3:
                corr_volat, p_value_volat = stats.pearsonr(
                    clean_volatility['sentiment_volatility'],
                    clean_volatility['return_volatility']
                )
            else:
                corr_volat, p_value_volat = np.nan, np.nan
            results['volatility_correlation'] = {
                'correlation': float(corr_volat) if not np.isnan(corr_volat) else None,
                'p_value': float(p_value_volat) if not np.isnan(p_value_volat) else None,
                'significant': bool(p_value_volat < 0.05) if not np.isnan(p_value_volat) else None,
                'data_points': int(len(clean_volatility)) if 'clean_volatility' in locals() else int(len(merged))
            }

        logger.info("Correlation analysis complete")
        return results

    def analyze_high_sentiment_days(self, news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict:
        """Analyze market performance on high/low sentiment days.

        Args:
            news_data: Daily news sentiment data
            price_data: Daily price data

        Returns:
            Analysis results
        """
        logger.info("Analyzing high sentiment days...")

        merged = pd.merge(news_data, price_data, on='date', how='inner')

        if len(merged) < 5:
            return {
                "success": False,
                "error": "Insufficient data for sentiment analysis"
            }

        # Define high/low sentiment days
        sentiment_threshold = merged['avg_sentiment'].median()

        high_sentiment_days = merged[merged['avg_sentiment'] > sentiment_threshold]
        low_sentiment_days = merged[merged['avg_sentiment'] <= sentiment_threshold]

        results = {
            'high_sentiment_days': {
                'count': int(len(high_sentiment_days)),
                'avg_return': float(high_sentiment_days['avg_return'].mean()),
                'avg_sentiment': float(high_sentiment_days['avg_sentiment'].mean()),
                'positive_days': int((high_sentiment_days['avg_return'] > 0).sum()),
                'negative_days': int((high_sentiment_days['avg_return'] <= 0).sum())
            },
            'low_sentiment_days': {
                'count': int(len(low_sentiment_days)),
                'avg_return': float(low_sentiment_days['avg_return'].mean()),
                'avg_sentiment': float(low_sentiment_days['avg_sentiment'].mean()),
                'positive_days': int((low_sentiment_days['avg_return'] > 0).sum()),
                'negative_days': int((low_sentiment_days['avg_return'] <= 0).sum())
            }
        }

        # Statistical test for difference in returns
        t_stat, p_value = stats.ttest_ind(
            high_sentiment_days['avg_return'].dropna(),
            low_sentiment_days['avg_return'].dropna()
        )

        results['statistical_test'] = {
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'significant_difference': bool(p_value < 0.05) if not np.isnan(p_value) else None
        }

        logger.info("High sentiment analysis complete")
        return results

    def generate_correlation_report(self, correlation_results: Dict, sentiment_analysis: Dict) -> Dict:
        """Generate comprehensive correlation analysis report.

        Args:
            correlation_results: Correlation analysis results
            sentiment_analysis: Sentiment day analysis results

        Returns:
            Complete report dictionary
        """
        logger.info("Generating correlation analysis report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "news-price_correlation",
            "correlation_analysis": correlation_results,
            "sentiment_day_analysis": sentiment_analysis,
            "summary": self._generate_summary(correlation_results, sentiment_analysis)
        }

        # Save report
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        report_file = reports_dir / f"correlation_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Correlation report saved: {report_file}")

        return report

    def _generate_summary(self, correlation_results: Dict, sentiment_analysis: Dict) -> Dict:
        """Generate summary of findings.

        Args:
            correlation_results: Correlation analysis results
            sentiment_analysis: Sentiment day analysis results

        Returns:
            Summary dictionary
        """
        summary = {
            "key_findings": [],
            "statistical_significance": [],
            "recommendations": []
        }

        # Analyze same-day correlation
        if 'same_day_correlation' in correlation_results:
            same_day = correlation_results['same_day_correlation']
            if same_day.get('significant'):
                summary['key_findings'].append(
                    f"Same-day correlation found: {same_day['correlation']:.3f} (p={same_day['p_value']:.3f})"
                )
                summary['statistical_significance'].append('same_day_correlation')

        # Analyze next-day correlation
        if 'next_day_correlation' in correlation_results:
            next_day = correlation_results['next_day_correlation']
            if next_day.get('significant'):
                summary['key_findings'].append(
                    f"Next-day correlation found: {next_day['correlation']:.3f} (p={next_day['p_value']:.3f})"
                )
                summary['statistical_significance'].append('next_day_correlation')

        # Analyze high sentiment days
        if 'statistical_test' in sentiment_analysis:
            if sentiment_analysis['statistical_test'].get('significant_difference'):
                summary['key_findings'].append(
                    "Significant difference in returns between high and low sentiment days"
                )
                summary['statistical_significance'].append('sentiment_day_difference')

        # Generate recommendations
        if not summary['statistical_significance']:
            summary['recommendations'].append("No significant correlations found - consider enriching data or analysis methods")
        else:
            summary['recommendations'].append("Significant correlations detected - consider building predictive models")

        return summary

    def run_full_analysis(self) -> Dict:
        """Run complete correlation analysis pipeline.

        Returns:
            Complete analysis results
        """
        logger.info("Starting full correlation analysis...")

        try:
            # 1. Load data
            news_df = self.load_processed_news()
            price_df = self.load_price_data()

            # 2. Prepare data
            news_data = self.prepare_news_data(news_df)
            price_data = self.prepare_price_data(price_df)

            # 3. Calculate correlations
            correlation_results = self.calculate_correlation(news_data, price_data)

            # 4. Analyze sentiment days
            sentiment_analysis = self.analyze_high_sentiment_days(news_data, price_data)

            # 5. Generate report
            report = self.generate_correlation_report(correlation_results, sentiment_analysis)

            logger.info("Correlation analysis complete")

            return {
                "success": True,
                "report": report,
                "correlation_results": correlation_results,
                "sentiment_analysis": sentiment_analysis
            }

        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main function to run correlation analysis."""
    print("="*60)
    print("News-Price Correlation Analysis")
    print("="*60)

    analyzer = NewsPriceCorrelationAnalyzer()

    print("\n1. Running full correlation analysis...")
    result = analyzer.run_full_analysis()

    if result.get("success"):
        print(f"Analysis completed successfully")

        # Display key findings
        if "report" in result and "summary" in result["report"]:
            summary = result["report"]["summary"]
            print(f"\n2. Key Findings:")
            for finding in summary.get("key_findings", []):
                print(f"   - {finding}")

            print(f"\n3. Statistical Significance:")
            for sig in summary.get("statistical_significance", []):
                print(f"   - {sig}")

            print(f"\n4. Recommendations:")
            for rec in summary.get("recommendations", []):
                print(f"   - {rec}")

        print("\n" + "="*60)
        print("Correlation Analysis: COMPLETE [OK]")
        print("="*60)
        return True
    else:
        print(f"Analysis failed: {result.get('error')}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)