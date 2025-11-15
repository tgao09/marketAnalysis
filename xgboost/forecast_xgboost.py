

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xgboost_model import StockXGBoost
from feature_engineering import TechnicalFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    

    def __init__(self, models_dir: str = 'xgboost/models',
                 results_dir: str = 'xgboost/results/forecasts',
                 recompute_technical: bool = True):
        
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.recompute_technical = recompute_technical
        os.makedirs(results_dir, exist_ok=True)

        # initialize feature engineer
        self.feature_engineer = TechnicalFeatureEngineer() if recompute_technical else None

        # discover models
        self.available_models = self._discover_models()
        logger.info(f"Found {len(self.available_models)} trained models")

    def _discover_models(self) -> Dict[int, str]:
        
        models = {}
        if not os.path.exists(self.models_dir):
            return models

        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl') and 'xgboost_' in filename:
                # extract horizon from filename: xgboost_1w_all.pkl -> 1
                parts = filename.split('_')
                if len(parts) >= 2:
                    horizon_str = parts[1].replace('w', '')
                    try:
                        horizon = int(horizon_str)
                        models[horizon] = os.path.join(self.models_dir, filename)
                    except ValueError:
                        continue

        return models

    def recompute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if not self.recompute_technical or self.feature_engineer is None:
            logger.warning("Using pre-computed technical indicators (not recommended for production)")
            return df

        logger.info("Recomputing technical indicators for production forecasting...")

        # calculate technical features (rsi, macd, bollinger bands, etc.)
        result = self.feature_engineer.calculate_technical_features(df)

        # calculate cross-sectional features (rankings, z-scores)
        result = self.feature_engineer.calculate_cross_sectional_features(result)

        # calculate interaction features
        result = self.feature_engineer.calculate_interaction_features(result)

        # calculate time features
        result = self.feature_engineer.calculate_time_features(result)

        logger.info("Technical indicators recomputed successfully")
        return result

    def forecast_multi_horizon(self, data_file: str,
                              horizons: Optional[List[int]] = None) -> pd.DataFrame:
        
        logger.info("Loading data...")
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows for {df['ticker'].nunique()} stocks")

        # critical: recompute technical indicators using all historical data
        # this simulates production conditions where features are computed from expanding window
        if self.recompute_technical:
            df = self.recompute_technical_indicators(df)
        else:
            logger.warning("Skipping technical indicator recomputation - forecasts may not match backtest performance")

        # use only latest data point per stock for forecasting
        latest_df = df.groupby('ticker').tail(1).reset_index(drop=True)
        logger.info(f"Forecasting for {len(latest_df)} stocks")

        # determine horizons to forecast
        if horizons is None:
            horizons = sorted(self.available_models.keys())

        forecasts = []

        for horizon in horizons:
            if horizon not in self.available_models:
                logger.warning(f"No model found for {horizon}-week horizon, skipping")
                continue

            logger.info(f"Forecasting {horizon}-week horizon...")

            # load model
            model = StockXGBoost.load_model(self.available_models[horizon])

            # generate predictions
            predictions = model.predict(latest_df)

            # create forecast records
            for idx, (_, row) in enumerate(latest_df.iterrows()):
                forecast_date = row['Date'] + timedelta(weeks=horizon)

                forecast_row = {
                    'ticker': row['ticker'],
                    'forecast_date': forecast_date,
                    'horizon_weeks': horizon,
                    'predicted_return': predictions[idx],
                    'sharpe_ratio_3m': row.get('sharpe_ratio_3m', None),
                    'latest_date': row['Date']
                }

                forecasts.append(forecast_row)

        forecasts_df = pd.DataFrame(forecasts)

        # save forecasts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(self.results_dir, f'forecasts_{timestamp}.csv')
        forecasts_df.to_csv(output_file, index=False)

        logger.info(f"Forecasts saved to {output_file}")

        return forecasts_df


def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='Generate XGBoost forecasts')
    parser.add_argument('--data-file', type=str, default='xgboost/features_engineered.csv',
                       help='Path to features dataset')
    parser.add_argument('--models-dir', type=str, default='xgboost/models',
                       help='Directory with trained models')
    parser.add_argument('--results-dir', type=str, default='xgboost/results/forecasts',
                       help='Output directory for forecast files')
    parser.add_argument('--horizons', type=int, nargs='*', default=None,
                       help='Horizons to forecast (default: all available)')
    parser.add_argument('--no-recompute-technical', action='store_true',
                       help='Skip technical indicator recomputation (faster but less reliable)')

    args = parser.parse_args()

    # initialize forecaster with same pipeline as walk_forward_test.py
    forecaster = XGBoostForecaster(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        recompute_technical=not args.no_recompute_technical
    )

    forecasts_df = forecaster.forecast_multi_horizon(args.data_file, args.horizons)

    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Pipeline: {'Production (with recomputation)' if not args.no_recompute_technical else 'Fast (pre-computed features)'}")
    print(f"Total forecasts: {len(forecasts_df)}")
    print(f"Stocks: {forecasts_df['ticker'].nunique()}")
    print(f"Horizons: {sorted(forecasts_df['horizon_weeks'].unique())}")
    print(f"\nReturn Statistics by Horizon:")
    for horizon in sorted(forecasts_df['horizon_weeks'].unique()):
        horizon_df = forecasts_df[forecasts_df['horizon_weeks'] == horizon]
        print(f"  {horizon}w: mean={horizon_df['predicted_return'].mean():.4f}, "
              f"std={horizon_df['predicted_return'].std():.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
