import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional, Union
import logging
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm

# Add the arimax directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arimax_model import StockARIMAX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAXPredictor:
    """
    Generate predictions using trained ARIMAX models.
    """

    def __init__(self, models_dir: str = 'arimaxmodels', results_dir: str = 'arimaxresults'):
        """
        Initialize predictor.

        Args:
            models_dir: Directory containing trained models
            results_dir: Directory to save prediction results
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.available_models = self._discover_models()

    def _discover_models(self) -> List[str]:
        """Discover available trained models."""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return []

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_arimax.pkl')]
        tickers = [f.replace('_arimax.pkl', '') for f in model_files]

        logger.info(f"Found {len(tickers)} trained models")
        return tickers

    def predict_single_stock(self, ticker: str, data_file: str, periods: int = 4,
                           return_confidence: bool = True) -> Dict[str, Any]:
        """
        Generate predictions for a single stock.

        Args:
            ticker: Stock ticker to predict
            data_file: Path to the dataset with latest data
            periods: Number of periods to predict
            return_confidence: Whether to include confidence intervals

        Returns:
            Prediction results dictionary
        """
        if ticker not in self.available_models:
            raise ValueError(f"No trained model found for {ticker}")

        # Load model
        model_path = os.path.join(self.models_dir, f"{ticker}_arimax.pkl")
        model = StockARIMAX.load_model(model_path)

        # Load data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])

        # Prepare data for this stock
        target, exog = model.prepare_data(df)

        if len(target) == 0:
            raise ValueError(f"No data available for {ticker}")

        # Get the most recent data for prediction (backtesting on last N periods)
        latest_exog = exog.tail(periods).copy()

        if len(latest_exog) < periods:
            logger.warning(f"{ticker}: Only {len(latest_exog)} periods available for prediction (requested {periods})")
            periods = len(latest_exog)

        # Get the actual historical dates corresponding to the exog data being used
        stock_data = df[df['ticker'] == ticker].copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)

        # Get the dates for the last N periods (the ones we're "predicting")
        actual_dates = stock_data['Date'].tail(periods).tolist()
        actual_date_strings = [date.strftime('%Y-%m-%d') for date in actual_dates]

        try:
            # Generate predictions (actually backtesting on recent historical data)
            if return_confidence:
                predictions, conf_intervals = model.predict(
                    latest_exog, steps=periods, return_conf_int=True
                )

                result = {
                    'ticker': ticker,
                    'periods': periods,
                    'predictions': predictions.tolist(),
                    'confidence_intervals': {
                        'lower': conf_intervals[:, 0].tolist(),
                        'upper': conf_intervals[:, 1].tolist()
                    },
                    'model_order': model.best_order,
                    'model_aic': model.aic_score
                }
            else:
                predictions = model.predict(latest_exog, steps=periods)

                result = {
                    'ticker': ticker,
                    'periods': periods,
                    'predictions': predictions.tolist(),
                    'model_order': model.best_order,
                    'model_aic': model.aic_score
                }

            # Use actual historical dates (these are backtesting results, not future predictions)
            result['prediction_dates'] = actual_date_strings
            result['backtest_note'] = "These are backtesting results on historical data, not future predictions"

            logger.info(f"{ticker}: Generated {periods} predictions")
            return result

        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'status': 'failed'
            }

    def predict_multiple_stocks(self, tickers: List[str], data_file: str, periods: int = 4,
                              return_confidence: bool = True) -> pd.DataFrame:
        """
        Generate predictions for multiple stocks.

        Args:
            tickers: List of stock tickers to predict
            data_file: Path to the dataset with latest data
            periods: Number of periods to predict
            return_confidence: Whether to include confidence intervals

        Returns:
            DataFrame with predictions for all stocks
        """
        results = []

        for ticker in tqdm(tickers, desc="Generating predictions", unit="stock"):
            try:
                result = self.predict_single_stock(
                    ticker, data_file, periods, return_confidence
                )

                if 'error' not in result:
                    # Convert to flat format for DataFrame
                    for i in range(periods):
                        row = {
                            'ticker': ticker,
                            'historical_date': result['prediction_dates'][i],  # Renamed to be clear
                            'predicted_return': result['predictions'][i],
                            'model_order': str(result['model_order']),
                            'model_aic': result['model_aic'],
                            'analysis_type': 'backtest'  # Clear labeling
                        }

                        if return_confidence and 'confidence_intervals' in result:
                            row['ci_lower'] = result['confidence_intervals']['lower'][i]
                            row['ci_upper'] = result['confidence_intervals']['upper'][i]

                        results.append(row)
                else:
                    logger.warning(f"Skipping {ticker}: {result['error']}")

            except Exception as e:
                logger.error(f"Failed to predict {ticker}: {e}")

        return pd.DataFrame(results)

    def predict_all_available(self, data_file: str, periods: int = 4,
                            return_confidence: bool = True) -> pd.DataFrame:
        """
        Generate predictions for all available models.

        Args:
            data_file: Path to the dataset with latest data
            periods: Number of periods to predict
            return_confidence: Whether to include confidence intervals

        Returns:
            DataFrame with predictions for all available stocks
        """
        logger.info(f"Generating predictions for {len(self.available_models)} stocks")

        return self.predict_multiple_stocks(
            self.available_models, data_file, periods, return_confidence
        )

    def create_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for predictions.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Summary statistics dictionary
        """
        if predictions_df.empty:
            return {'error': 'No predictions available'}

        summary = {
            'total_stocks': predictions_df['ticker'].nunique(),
            'total_predictions': len(predictions_df),
            'prediction_period': predictions_df['historical_date'].nunique(),
            'date_range': {
                'start': predictions_df['historical_date'].min(),
                'end': predictions_df['historical_date'].max()
            }
        }

        # Return statistics
        if 'predicted_return' in predictions_df.columns:
            return_stats = predictions_df['predicted_return'].describe()
            summary['return_statistics'] = {
                'mean': return_stats['mean'],
                'median': return_stats['50%'],
                'std': return_stats['std'],
                'min': return_stats['min'],
                'max': return_stats['max']
            }

            # Directional predictions
            positive_predictions = (predictions_df['predicted_return'] > 0).sum()
            summary['directional_split'] = {
                'positive_predictions': positive_predictions,
                'negative_predictions': len(predictions_df) - positive_predictions,
                'positive_percentage': positive_predictions / len(predictions_df) * 100
            }

            # Top/bottom predicted performers
            latest_predictions = predictions_df.groupby('ticker')['predicted_return'].first()
            summary['top_predicted_performers'] = latest_predictions.nlargest(10).to_dict()
            summary['bottom_predicted_performers'] = latest_predictions.nsmallest(10).to_dict()

        return summary

    def save_predictions(self, predictions_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save backtest results to CSV file.

        Args:
            predictions_df: DataFrame with backtest results
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_results_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        predictions_df.to_csv(filepath, index=False)
        logger.info(f"Backtest results saved to: {filepath}")

        return filepath

    def generate_prediction_report(self, data_file: str, periods: int = 4,
                                 save_results: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report.

        Args:
            data_file: Path to the dataset with historical data
            periods: Number of recent periods to backtest
            save_results: Whether to save results to file

        Returns:
            Complete backtest report
        """
        logger.info(f"Generating backtest report for {periods} recent periods")

        # Generate backtest results (not future predictions)
        predictions_df = self.predict_all_available(
            data_file, periods, return_confidence=True
        )

        if predictions_df.empty:
            return {'error': 'No backtest results could be generated'}

        # Create summary
        summary = self.create_prediction_summary(predictions_df)

        # Save results if requested
        saved_file = None
        if save_results:
            saved_file = self.save_predictions(predictions_df)

        # Combine into report
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': summary,
            'predictions_file': saved_file,
            'model_count': len(self.available_models),
            'successful_predictions': predictions_df['ticker'].nunique()
        }

        return report

def main():
    """Main backtest script."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate backtest results using trained ARIMAX models')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Backtest specific ticker (default: backtest all)')
    parser.add_argument('--data-file', type=str, default='../dataset/stock_dataset_with_lags.csv',
                       help='Path to the dataset with historical data')
    parser.add_argument('--periods', type=int, default=4,
                       help='Number of recent periods to backtest (default: 4)')
    parser.add_argument('--models-dir', type=str, default='arimaxmodels',
                       help='Directory containing trained models')
    parser.add_argument('--results-dir', type=str, default='arimaxresults',
                       help='Directory to save backtest results')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Skip confidence interval calculation')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Custom output filename')

    args = parser.parse_args()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    predictor = ARIMAXPredictor(args.models_dir, args.results_dir)

    try:
        if args.ticker:
            # Backtest single stock
            print(f"Generating backtest results for {args.ticker}...")

            result = predictor.predict_single_stock(
                args.ticker, args.data_file, args.periods,
                return_confidence=not args.no_confidence
            )

            if 'error' in result:
                print(f"Backtest failed: {result['error']}")
                sys.exit(1)

            print(f"\nBacktest Results for {args.ticker} (Recent Historical Performance):")
            print("-" * 70)
            for i, (date, pred) in enumerate(zip(result['prediction_dates'], result['predictions'])):
                line = f"  Historical Date {date}: Predicted Return {pred:.4f}"
                if 'confidence_intervals' in result:
                    ci_low = result['confidence_intervals']['lower'][i]
                    ci_high = result['confidence_intervals']['upper'][i]
                    line += f" [CI: {ci_low:.4f}, {ci_high:.4f}]"
                print(line)

        else:
            # Generate comprehensive report
            print("Generating backtest results for all available models...")

            report = predictor.generate_prediction_report(
                args.data_file, args.periods, save_results=True
            )

            if 'error' in report:
                print(f"Backtest report generation failed: {report['error']}")
                sys.exit(1)

            print("\n" + "="*60)
            print("BACKTEST SUMMARY (Historical Model Performance)")
            print("="*60)

            summary = report['summary']
            print(f"Total stocks: {summary['total_stocks']}")
            print(f"Total backtest results: {summary['total_predictions']}")
            print(f"Historical period: {summary['date_range']['start']} to {summary['date_range']['end']}")

            if 'return_statistics' in summary:
                stats = summary['return_statistics']
                print(f"\nModel Predicted Returns Statistics:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  Std Dev: {stats['std']:.4f}")
                print(f"  Range: {stats['min']:.4f} to {stats['max']:.4f}")

            if 'directional_split' in summary:
                direction = summary['directional_split']
                print(f"\nDirectional Predictions (Historical):")
                print(f"  Positive: {direction['positive_predictions']} ({direction['positive_percentage']:.1f}%)")
                print(f"  Negative: {direction['negative_predictions']} ({100-direction['positive_percentage']:.1f}%)")

            if 'top_predicted_performers' in summary:
                print(f"\nTop 5 Historical Model Predictions:")
                top_performers = list(summary['top_predicted_performers'].items())[:5]
                for ticker, predicted_return in top_performers:
                    print(f"  {ticker}: {predicted_return:.4f}")

            print(f"\nBacktest results saved to: {report['predictions_file']}")
            print("="*60)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()