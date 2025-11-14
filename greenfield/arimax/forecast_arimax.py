# !/usr/bin/env python3


import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings
from datetime import datetime, timedelta

# add the arimax directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arimax_model import StockARIMAX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ARIMAXPredictor:
    

    def __init__(self, models_dir: str = 'arimax/arimaxmodels', results_dir: str = 'arimax/arimaxresults'):
        
        self.models_dir = models_dir
        self.results_dir = results_dir

        # discover available models
        self.available_models = self._discover_models()
        logger.info(f"Discovered {len(self.available_models)} trained models")

    def _discover_models(self) -> List[str]:
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return []

        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_arimax.pkl'):
                ticker = filename.replace('_arimax.pkl', '')
                models.append(ticker)

        return sorted(models)

    def _shift_lags_for_forecast(self, current_exog: pd.DataFrame,
                                  new_prediction: float,
                                  feature_columns: List[str]) -> pd.DataFrame:
        
        next_exog = current_exog.copy()

        # identify stock-specific lag groups (weekly_return, high_return, low_return, volume_change, volatility)
        stock_features = ['weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']

        for base_feature in stock_features:
            # find all lags for this feature (e.g., weekly_return_lag_1, weekly_return_lag_2, weekly_return_lag_3)
            lag_cols = [col for col in feature_columns if col.startswith(f'{base_feature}_lag_')]

            if not lag_cols:
                continue

            # extract lag numbers and sort in descending order
            lag_numbers = []
            for col in lag_cols:
                try:
                    lag_num = int(col.split('_lag_')[-1])
                    lag_numbers.append((lag_num, col))
                except ValueError:
                    continue

            lag_numbers.sort(reverse=True)  # start from highest lag

            # shift: lag_3 <- lag_2, lag_2 <- lag_1
            for i in range(len(lag_numbers) - 1):
                higher_lag_num, higher_lag_col = lag_numbers[i]
                lower_lag_num, lower_lag_col = lag_numbers[i + 1]

                if lower_lag_col in next_exog.columns:
                    next_exog[higher_lag_col] = next_exog[lower_lag_col].values

            # set lag_1 to new prediction (for weekly_return only)
            if base_feature == 'weekly_return':
                lag_1_col = f'{base_feature}_lag_1'
                if lag_1_col in next_exog.columns:
                    next_exog[lag_1_col] = new_prediction

        return next_exog

    def predict_future_single_stock(self, ticker: str, data_file: str, periods: int = 4,
                                   return_confidence: bool = True) -> Dict[str, Any]:
        
        if ticker not in self.available_models:
            raise ValueError(f"No trained ARIMAX model found for {ticker}")

        # load arimax model
        model_path = os.path.join(self.models_dir, f"{ticker}_arimax.pkl")
        arimax_model = StockARIMAX.load_model(model_path)

        try:
            # load historical data
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'])

            # prepare data to get lagged features (only uses historical data)
            target, exog = arimax_model.prepare_data(df)

            if len(exog) == 0:
                raise ValueError(f"No feature data available for {ticker}")

            # iterative forecasting: shift lags at each step to simulate real-time forecasting
            # this prevents assuming lags remain constant across forecast horizon
            logger.info(f"{ticker}: Using iterative lag shifting for {periods} period forecast")

            predictions_list = []
            lower_bounds = []
            upper_bounds = []

            # get latest exogenous features
            current_exog = exog.tail(1).copy().reset_index(drop=True)

            for step in range(periods):
                # predict next step using current lagged features
                if return_confidence:
                    step_pred, step_conf = arimax_model.predict(
                        current_exog, steps=1, return_conf_int=True
                    )
                    predictions_list.append(step_pred[0])
                    lower_bounds.append(step_conf[0, 0])
                    upper_bounds.append(step_conf[0, 1])
                else:
                    step_pred = arimax_model.predict(current_exog, steps=1)
                    predictions_list.append(step_pred[0])

                # shift lags for next iteration: lag_2 <- lag_1, lag_1 <- current prediction
                # identify lag columns (e.g., feature_lag_1, feature_lag_2, feature_lag_3)
                if step < periods - 1:  # don't shift on last iteration
                    current_exog = self._shift_lags_for_forecast(
                        current_exog, step_pred[0], arimax_model.feature_columns
                    )

            # package results
            if return_confidence:
                result = {
                    'ticker': ticker,
                    'periods': periods,
                    'predictions': predictions_list,
                    'confidence_intervals': {
                        'lower': lower_bounds,
                        'upper': upper_bounds
                    },
                    'model_order': arimax_model.best_order,
                    'model_aic': arimax_model.aic_score,
                    'forecasting_mode': 'iterative'
                }
            else:
                result = {
                    'ticker': ticker,
                    'periods': periods,
                    'predictions': predictions_list,
                    'model_order': arimax_model.best_order,
                    'model_aic': arimax_model.aic_score,
                    'forecasting_mode': 'iterative'
                }

            # generate true future dates
            last_date = df[df['ticker'] == ticker]['Date'].max()

            future_dates = []
            for i in range(1, periods + 1):
                future_date = last_date + timedelta(weeks=i)
                future_dates.append(future_date.strftime('%Y-%m-%d'))

            result['future_dates'] = future_dates
            result['prediction_type'] = 'future_forecast'
            result['forecast_valid'] = True  # simple validation - features exist

            # extract latest sharpe_ratio_3m if available (stock quality metric)
            ticker_df = df[df['ticker'] == ticker]
            if 'sharpe_ratio_3m' in ticker_df.columns:
                result['sharpe_ratio_3m'] = ticker_df['sharpe_ratio_3m'].iloc[-1]
            else:
                result['sharpe_ratio_3m'] = None

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
                                     periods: int = 4, return_confidence: bool = True) -> pd.DataFrame:
        
        results = []

        for ticker in tickers:
            try:
                result = self.predict_future_single_stock(
                    ticker, data_file, periods, return_confidence
                )

                if 'error' not in result:
                    # convert to flat format for dataframe
                    for i in range(periods):
                        row = {
                            'ticker': ticker,
                            'future_date': result['future_dates'][i],
                            'predicted_return': result['predictions'][i],
                            'model_order': str(result['model_order']),
                            'model_aic': result['model_aic'],
                            'prediction_type': 'future_forecast',
                        }

                        if return_confidence and 'confidence_intervals' in result:
                            row['ci_lower'] = result['confidence_intervals']['lower'][i]
                            row['ci_upper'] = result['confidence_intervals']['upper'][i]

                        # add validation flag
                        row['forecast_valid'] = result.get('forecast_valid', False)

                        # add sharpe_ratio_3m (stock quality metric)
                        if 'sharpe_ratio_3m' in result:
                            row['sharpe_ratio_3m'] = result['sharpe_ratio_3m']

                        results.append(row)
                else:
                    logger.warning(f"Skipping {ticker}: {result['error']}")

            except Exception as e:
                logger.error(f"Failed to predict {ticker}: {e}")

        return pd.DataFrame(results)

    def generate_future_forecast_report(self, data_file: str, periods: int = 4,
                                      forecasting_mode: str = 'individual',
                                      save_results: bool = True) -> Dict[str, Any]:
        
        logger.info(f"Generating future forecast report for {periods} periods")

        # generate future predictions
        predictions_df = self.predict_future_multiple_stocks(
            self.available_models, data_file, periods, return_confidence=True,
            
        )

        if predictions_df.empty:
            return {'error': 'No future predictions could be generated'}

        # create summary
        summary = self.create_future_forecast_summary(predictions_df)

        # save results if requested
        saved_file = None
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"future_forecasts_{timestamp}.csv"
            saved_file = self.save_predictions(predictions_df, filename)

        # combine into report
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': summary,
            'forecasts_file': saved_file,
            'model_count': len(self.available_models),
            'successful_forecasts': predictions_df['ticker'].nunique(),
            'periods_ahead': periods
        }

        return report

    def create_future_forecast_summary(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        
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

        # return statistics
        if 'predicted_return' in predictions_df.columns:
            return_stats = predictions_df['predicted_return'].describe()
            summary['return_statistics'] = {
                'mean': return_stats['mean'],
                'median': return_stats['50%'],
                'std': return_stats['std'],
                'min': return_stats['min'],
                'max': return_stats['max']
            }

            # directional predictions
            positive_predictions = (predictions_df['predicted_return'] > 0).sum()
            summary['directional_split'] = {
                'positive_predictions': positive_predictions,
                'negative_predictions': len(predictions_df) - positive_predictions,
                'positive_percentage': positive_predictions / len(predictions_df) * 100
            }

            # top/bottom predicted performers (next week only)
            next_week_predictions = predictions_df.groupby('ticker')['predicted_return'].first()
            summary['top_predicted_performers'] = next_week_predictions.nlargest(10).to_dict()
            summary['bottom_predicted_performers'] = next_week_predictions.nsmallest(10).to_dict()

        return summary

    def save_predictions(self, predictions_df: pd.DataFrame, filename: str = None) -> str:
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"future_forecasts_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        predictions_df.to_csv(filepath, index=False)
        logger.info(f"Future forecasts saved to: {filepath}")

        return filepath

def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='Generate true future predictions using ARIMAX models')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Predict specific ticker and save to CSV (default: predict all)')
    parser.add_argument('--data-file', type=str, default='dataset/stock_dataset_with_lags.csv',
                       help='Path to the dataset with historical data (relative to greenfield/ directory)')
    parser.add_argument('--periods', type=int, default=4,
                       help='Number of future periods to predict (default: 4)')
    parser.add_argument('--models-dir', type=str, default='arimax/arimaxmodels',
                       help='Directory containing trained models')
    parser.add_argument('--results-dir', type=str, default='arimax/arimaxresults',
                       help='Directory to save forecast results')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Skip confidence interval calculation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to CSV file (console output only)')

    args = parser.parse_args()

    # suppress warnings
    warnings.filterwarnings('ignore')

    predictor = ARIMAXPredictor(args.models_dir, args.results_dir)

    try:
        if args.ticker:
            # predict single stock
            print(f"Generating future forecasts for {args.ticker}...")

            result = predictor.predict_future_single_stock(
                args.ticker, args.data_file, args.periods,
                return_confidence=not args.no_confidence,
                
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

            # print forecast validation status
            if result.get('forecast_valid'):
                print(f"\nForecast Status: VALID")

            # save single stock results to csv file (unless --no-save specified)
            if not args.no_save:
                try:
                    # convert single stock result to dataframe format
                    single_stock_df = predictor.predict_future_multiple_stocks(
                        [args.ticker], args.data_file, args.periods,
                        return_confidence=not args.no_confidence,
                        
                    )

                    if not single_stock_df.empty:
                        # generate filename with ticker and timestamp
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
            # generate comprehensive report
            print("Generating future forecasts for all available models...")

            report = predictor.generate_future_forecast_report(
                args.data_file, args.periods, save_results=True
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
            print()

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