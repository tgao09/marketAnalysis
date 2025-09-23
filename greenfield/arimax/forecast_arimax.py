#!/usr/bin/env python3
"""
True ARIMAX Predictor

This module extends the original ARIMAX predictor to enable genuine future
forecasting by integrating exogenous variable forecasting.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings
from datetime import datetime, timedelta

# Add the arimax directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arimax_model import StockARIMAX
from exogenous_forecaster import ExogenousForecaster, create_forecaster
from research.backtest_arimax import ARIMAXPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class TrueARIMAXPredictor(ARIMAXPredictor):
    """
    Extended ARIMAX predictor that can generate true future predictions
    by forecasting exogenous variables.
    """

    def __init__(self, models_dir: str = 'arimaxmodels', results_dir: str = 'arimaxresults'):
        """
        Initialize the true ARIMAX predictor.

        Args:
            models_dir: Directory containing trained ARIMAX models
            results_dir: Directory to save prediction results
        """
        super().__init__(models_dir, results_dir)
        self.exogenous_forecasters = {}  # Store forecasters per ticker

    def fit_exogenous_forecaster(self, ticker: str, data_file: str,
                                forecasting_mode: str = 'individual') -> None:
        """
        Fit exogenous variable forecaster for a specific ticker.

        Args:
            ticker: Stock ticker
            data_file: Path to historical data
            forecasting_mode: 'individual' or 'var'
        """
        # Load data
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])

        # Create and fit forecaster
        forecaster = create_forecaster(mode=forecasting_mode)
        forecaster.fit(df, ticker)

        self.exogenous_forecasters[ticker] = forecaster
        logger.info(f"Exogenous forecaster fitted for {ticker}")

    def predict_future_single_stock(self, ticker: str, data_file: str, periods: int = 4,
                                   return_confidence: bool = True,
                                   forecasting_mode: str = 'individual') -> Dict[str, Any]:
        """
        Generate true future predictions for a single stock.

        Args:
            ticker: Stock ticker to predict
            data_file: Path to the dataset with latest data
            periods: Number of future periods to predict
            return_confidence: Whether to include confidence intervals
            forecasting_mode: Mode for exogenous forecasting

        Returns:
            Prediction results dictionary with future dates
        """
        if ticker not in self.available_models:
            raise ValueError(f"No trained ARIMAX model found for {ticker}")

        # Load ARIMAX model
        model_path = os.path.join(self.models_dir, f"{ticker}_arimax.pkl")
        arimax_model = StockARIMAX.load_model(model_path)

        # Fit exogenous forecaster if not already done
        if ticker not in self.exogenous_forecasters:
            self.fit_exogenous_forecaster(ticker, data_file, forecasting_mode)

        forecaster = self.exogenous_forecasters[ticker]

        try:
            # Generate exogenous variable forecasts
            exog_features, uncertainty_info = forecaster.forecast_exogenous_features(
                periods, confidence_level=0.95
            )

            logger.info(f"{ticker}: Generated exogenous forecasts for {periods} periods")

            # Use ARIMAX model with forecasted exogenous variables
            if return_confidence:
                predictions, conf_intervals = arimax_model.predict(
                    exog_features, steps=periods, return_conf_int=True
                )

                result = {
                    'ticker': ticker,
                    'periods': periods,
                    'predictions': predictions.tolist(),
                    'confidence_intervals': {
                        'lower': conf_intervals[:, 0].tolist(),
                        'upper': conf_intervals[:, 1].tolist()
                    },
                    'model_order': arimax_model.best_order,
                    'model_aic': arimax_model.aic_score
                }
            else:
                predictions = arimax_model.predict(exog_features, steps=periods)

                result = {
                    'ticker': ticker,
                    'periods': periods,
                    'predictions': predictions.tolist(),
                    'model_order': arimax_model.best_order,
                    'model_aic': arimax_model.aic_score
                }

            # Generate true future dates
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            last_date = df[df['ticker'] == ticker]['Date'].max()

            future_dates = []
            for i in range(1, periods + 1):
                future_date = last_date + timedelta(weeks=i)
                future_dates.append(future_date.strftime('%Y-%m-%d'))

            result['future_dates'] = future_dates
            result['prediction_type'] = 'future_forecast'
            result['exogenous_forecasting_mode'] = forecasting_mode
            result['exogenous_uncertainty'] = uncertainty_info

            # Add validation information
            validation_results = forecaster.validate_forecasts(exog_features)
            result['forecast_validation'] = validation_results

            logger.info(f"{ticker}: Generated {periods} future predictions")
            return result

        except Exception as e:
            logger.error(f"Future prediction failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'status': 'failed',
                'prediction_type': 'future_forecast'
            }

    def predict_future_multiple_stocks(self, tickers: List[str], data_file: str,
                                     periods: int = 4, return_confidence: bool = True,
                                     forecasting_mode: str = 'individual') -> pd.DataFrame:
        """
        Generate future predictions for multiple stocks.

        Args:
            tickers: List of stock tickers to predict
            data_file: Path to the dataset with latest data
            periods: Number of future periods to predict
            return_confidence: Whether to include confidence intervals
            forecasting_mode: Mode for exogenous forecasting

        Returns:
            DataFrame with future predictions for all stocks
        """
        results = []

        for ticker in tickers:
            try:
                result = self.predict_future_single_stock(
                    ticker, data_file, periods, return_confidence, forecasting_mode
                )

                if 'error' not in result:
                    # Convert to flat format for DataFrame
                    for i in range(periods):
                        row = {
                            'ticker': ticker,
                            'future_date': result['future_dates'][i],
                            'predicted_return': result['predictions'][i],
                            'model_order': str(result['model_order']),
                            'model_aic': result['model_aic'],
                            'prediction_type': 'future_forecast',
                            'exogenous_mode': forecasting_mode
                        }

                        if return_confidence and 'confidence_intervals' in result:
                            row['ci_lower'] = result['confidence_intervals']['lower'][i]
                            row['ci_upper'] = result['confidence_intervals']['upper'][i]

                        # Add validation flags
                        validation = result.get('forecast_validation', {})
                        row['forecast_valid'] = all(validation.values()) if validation else False

                        results.append(row)
                else:
                    logger.warning(f"Skipping {ticker}: {result['error']}")

            except Exception as e:
                logger.error(f"Failed to predict {ticker}: {e}")

        return pd.DataFrame(results)

    def generate_future_forecast_report(self, data_file: str, periods: int = 4,
                                      forecasting_mode: str = 'individual',
                                      save_results: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive future forecast report.

        Args:
            data_file: Path to the dataset with latest data
            periods: Number of future periods to predict
            forecasting_mode: Mode for exogenous forecasting
            save_results: Whether to save results to file

        Returns:
            Complete future forecast report
        """
        logger.info(f"Generating future forecast report for {periods} periods")

        # Generate future predictions
        predictions_df = self.predict_future_multiple_stocks(
            self.available_models, data_file, periods, return_confidence=True,
            forecasting_mode=forecasting_mode
        )

        if predictions_df.empty:
            return {'error': 'No future predictions could be generated'}

        # Create summary
        summary = self.create_future_forecast_summary(predictions_df)

        # Save results if requested
        saved_file = None
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"future_forecasts_{timestamp}.csv"
            saved_file = self.save_predictions(predictions_df, filename)

        # Combine into report
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': summary,
            'forecasts_file': saved_file,
            'model_count': len(self.available_models),
            'successful_forecasts': predictions_df['ticker'].nunique(),
            'forecasting_mode': forecasting_mode,
            'periods_ahead': periods
        }

        return report

    def create_future_forecast_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for future forecasts.

        Args:
            predictions_df: DataFrame with future predictions

        Returns:
            Summary statistics dictionary
        """
        if predictions_df.empty:
            return {'error': 'No predictions available'}

        summary = {
            'total_stocks': predictions_df['ticker'].nunique(),
            'total_forecasts': len(predictions_df),
            'forecast_periods': predictions_df['future_date'].nunique(),
            'date_range': {
                'start': predictions_df['future_date'].min(),
                'end': predictions_df['future_date'].max()
            },
            'valid_forecasts': predictions_df['forecast_valid'].sum() if 'forecast_valid' in predictions_df.columns else len(predictions_df)
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

            # Top/bottom predicted performers (next week only)
            next_week_predictions = predictions_df.groupby('ticker')['predicted_return'].first()
            summary['top_predicted_performers'] = next_week_predictions.nlargest(10).to_dict()
            summary['bottom_predicted_performers'] = next_week_predictions.nsmallest(10).to_dict()

        return summary

    def save_predictions(self, predictions_df: pd.DataFrame, filename: str = None) -> str:
        """
        Save future predictions to CSV file.

        Args:
            predictions_df: DataFrame with predictions
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"future_forecasts_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        predictions_df.to_csv(filepath, index=False)
        logger.info(f"Future forecasts saved to: {filepath}")

        return filepath

def main():
    """Main future prediction script."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate true future predictions using ARIMAX models')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Predict specific ticker and save to CSV (default: predict all)')
    parser.add_argument('--data-file', type=str, default='../dataset/stock_dataset_with_lags.csv',
                       help='Path to the dataset with historical data')
    parser.add_argument('--periods', type=int, default=4,
                       help='Number of future periods to predict (default: 4)')
    parser.add_argument('--models-dir', type=str, default='arimaxmodels',
                       help='Directory containing trained models')
    parser.add_argument('--results-dir', type=str, default='arimaxresults',
                       help='Directory to save forecast results')
    parser.add_argument('--forecasting-mode', type=str, default='individual',
                       choices=['individual', 'var'],
                       help='Exogenous forecasting mode (default: individual)')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Skip confidence interval calculation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to CSV file (console output only)')

    args = parser.parse_args()

    # Suppress warnings
    warnings.filterwarnings('ignore')

    predictor = TrueARIMAXPredictor(args.models_dir, args.results_dir)

    try:
        if args.ticker:
            # Predict single stock
            print(f"Generating future forecasts for {args.ticker}...")

            result = predictor.predict_future_single_stock(
                args.ticker, args.data_file, args.periods,
                return_confidence=not args.no_confidence,
                forecasting_mode=args.forecasting_mode
            )

            if 'error' in result:
                print(f"Future forecasting failed: {result['error']}")
                sys.exit(1)

            print(f"\nFuture Forecasts for {args.ticker}:")
            print("-" * 50)
            for i, (date, pred) in enumerate(zip(result['future_dates'], result['predictions'])):
                line = f"  {date}: {pred:.4f}"
                if 'confidence_intervals' in result:
                    ci_low = result['confidence_intervals']['lower'][i]
                    ci_high = result['confidence_intervals']['upper'][i]
                    line += f" [{ci_low:.4f}, {ci_high:.4f}]"
                print(line)

            # Print forecast validation
            validation = result.get('forecast_validation', {})
            if validation:
                print(f"\nForecast Validation:")
                for check, passed in validation.items():
                    status = "PASS" if passed else "FAIL"
                    print(f"  {check}: {status}")

            # Save single stock results to CSV file (unless --no-save specified)
            if not args.no_save:
                try:
                    # Convert single stock result to DataFrame format
                    single_stock_df = predictor.predict_future_multiple_stocks(
                        [args.ticker], args.data_file, args.periods,
                        return_confidence=not args.no_confidence,
                        forecasting_mode=args.forecasting_mode
                    )

                    if not single_stock_df.empty:
                        # Generate filename with ticker and timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"future_forecasts_{args.ticker}_{timestamp}.csv"
                        saved_file = predictor.save_predictions(single_stock_df, filename)
                        print(f"\nResults saved to: {saved_file}")
                    else:
                        print(f"\nWarning: Could not save results to file")

                except Exception as save_error:
                    print(f"\nWarning: Could not save results to file: {save_error}")
                    print("Results are still available in the console output above.")
            else:
                print(f"\nNote: Results not saved (--no-save flag specified)")

        else:
            # Generate comprehensive report
            print("Generating future forecasts for all available models...")

            report = predictor.generate_future_forecast_report(
                args.data_file, args.periods, args.forecasting_mode, save_results=True
            )

            if 'error' in report:
                print(f"Future forecast generation failed: {report['error']}")
                sys.exit(1)

            print("\n" + "="*60)
            print("FUTURE FORECAST SUMMARY")
            print("="*60)

            summary = report['summary']
            print(f"Total stocks: {summary['total_stocks']}")
            print(f"Total forecasts: {summary['total_forecasts']}")
            print(f"Forecast period: {summary['date_range']['start']} to {summary['date_range']['end']}")
            print(f"Valid forecasts: {summary['valid_forecasts']}")
            print(f"Exogenous forecasting mode: {report['forecasting_mode']}")

            if 'return_statistics' in summary:
                stats = summary['return_statistics']
                print(f"\nPredicted Returns Statistics:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  Std Dev: {stats['std']:.4f}")
                print(f"  Range: {stats['min']:.4f} to {stats['max']:.4f}")

            if 'directional_split' in summary:
                direction = summary['directional_split']
                print(f"\nDirectional Predictions (Future):")
                print(f"  Positive: {direction['positive_predictions']} ({direction['positive_percentage']:.1f}%)")
                print(f"  Negative: {direction['negative_predictions']} ({100-direction['positive_percentage']:.1f}%)")

            if 'top_predicted_performers' in summary:
                print(f"\nTop 5 Predicted Performers (Next Week):")
                top_performers = list(summary['top_predicted_performers'].items())[:5]
                for ticker, predicted_return in top_performers:
                    print(f"  {ticker}: {predicted_return:.4f}")

            print(f"\nResults saved to: {report['forecasts_file']}")
            print("="*60)

    except Exception as e:
        logger.error(f"Future forecasting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()