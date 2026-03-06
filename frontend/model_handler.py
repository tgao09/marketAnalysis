#!/usr/bin/env python3
"""
Model Handler for Forecasting Frontend
Supports loading ARIMAX and XGBoost models and managing predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add paths for importing modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'arimax'))
sys.path.append(str(PROJECT_ROOT / 'dataset'))
sys.path.append(str(PROJECT_ROOT / 'xgboost'))
logger = logging.getLogger(__name__)

class ModelHandler:
    """
    Handles model operations and prediction loading for supported model types
    """

    SUPPORTED_MODELS = ('arimax', 'xgboost')

    def __init__(self, model_type: str = 'arimax'):
        """Initialize the model handler with project paths"""
        self.project_root = PROJECT_ROOT
        self.dataset_dir = self.project_root / 'dataset'
        self.model_type = model_type.lower()

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type '{model_type}'. Supported: {self.SUPPORTED_MODELS}")

        if self.model_type == 'arimax':
            self.models_dir = self.project_root / 'arimax' / 'arimaxmodels'
            self.results_dir = self.project_root / 'arimax' / 'arimaxresults'
        else:
            # Tuned XGBoost models live in the dedicated directory
            self.models_dir = self.project_root / 'xgboost' / 'xgboostmodels_tuned'
            self.results_dir = self.project_root / 'xgboost' / 'xgboostresults'
            os.makedirs(self.results_dir, exist_ok=True)
        
        
        # Cache for loaded predictions
        self._predictions_cache = {}
        self._cache_timestamp = None

    def get_available_tickers(self) -> List[str]:
        """
        Get list of tickers with trained models for current model type

        Returns:
            List of ticker symbols
        """
        try:
            if self.model_type == 'arimax':
                if not os.path.exists(self.models_dir):
                    logger.warning(f"Models directory not found: {self.models_dir}")
                    return []

                # Find all model files
                model_files = glob.glob(os.path.join(self.models_dir, '*_arimax.pkl'))

                # Extract ticker symbols from filenames
                tickers = []
                for model_file in model_files:
                    filename = os.path.basename(model_file)
                    ticker = filename.replace('_arimax.pkl', '')
                    tickers.append(ticker)

                return sorted(tickers)

            # XGBoost models are cross-sectional; discover tickers from latest predictions
            predictions_df = self.get_latest_predictions()
            if predictions_df is None or predictions_df.empty:
                logger.warning("No XGBoost predictions available to determine tickers")
                return []

            return sorted(predictions_df['ticker'].dropna().unique().tolist())

        except Exception as e:
            logger.error(f"Error getting available tickers: {e}")
            return []

    def get_latest_predictions(self) -> Optional[pd.DataFrame]:
        """
        Load the most recent prediction results

        Returns:
            DataFrame with latest predictions or None if not found
        """
        try:
            if not os.path.exists(self.results_dir):
                logger.warning(f"Results directory not found: {self.results_dir}")
                return None

            if self.model_type == 'arimax':
                pattern = 'future_forecasts_*.csv'
            else:
                pattern = 'forecasts_*.csv'

            # Find prediction files
            forecast_files = glob.glob(os.path.join(self.results_dir, pattern))

            if not forecast_files:
                logger.warning("No prediction files found")
                return None

            # Get the most recent file
            latest_file = max(forecast_files, key=os.path.getctime)
            logger.info(f"Loading predictions from: {latest_file}")

            # Load predictions
            df = pd.read_csv(latest_file)

            # Standardize date column
            if self.model_type == 'arimax':
                if 'future_date' in df.columns:
                    df['future_date'] = pd.to_datetime(df['future_date'])
                    df['date'] = df['future_date']
                elif 'prediction_date' in df.columns:
                    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
                    df['date'] = df['prediction_date']
            else:
                if 'forecast_date' in df.columns:
                    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
                    df['date'] = df['forecast_date']
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                if 'latest_date' in df.columns:
                    df['latest_date'] = pd.to_datetime(df['latest_date'])

            if 'predicted_return' in df.columns:
                df['predicted_return'] = pd.to_numeric(df['predicted_return'], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return None

    def get_predictions(self, ticker: str, periods: int = 4,
                       include_confidence: bool = True) -> Optional[pd.DataFrame]:
        """
        Get predictions for a specific ticker

        Args:
            ticker: Stock ticker symbol
            periods: Number of periods to return (if available)
            include_confidence: Whether to include confidence intervals

        Returns:
            DataFrame with predictions for the ticker
        """
        try:
            # Load latest predictions if not cached or outdated
            if self._should_refresh_cache():
                self._predictions_cache = {}
                predictions_df = self.get_latest_predictions()
                if predictions_df is not None:
                    # Cache predictions by ticker
                    for t in predictions_df['ticker'].unique():
                        self._predictions_cache[t] = predictions_df[predictions_df['ticker'] == t].copy()
                    self._cache_timestamp = datetime.now()

            # Get predictions for this ticker from cache
            if ticker not in self._predictions_cache:
                logger.warning(f"No predictions found for ticker: {ticker}")
                return None

            ticker_predictions = self._predictions_cache[ticker].copy()

            # Sort predictions chronologically and by horizon if available
            sort_columns: List[str] = []
            if self.model_type == 'xgboost' and 'horizon_weeks' in ticker_predictions.columns:
                sort_columns.append('horizon_weeks')
            if 'date' in ticker_predictions.columns:
                sort_columns.append('date')

            if sort_columns:
                ticker_predictions = ticker_predictions.sort_values(sort_columns).reset_index(drop=True)

            # For XGBoost, multiple horizons may exist; limit entries based on requested periods
            if len(ticker_predictions) > periods:
                ticker_predictions = ticker_predictions.head(periods)

            if 'predicted_return' in ticker_predictions.columns:
                ticker_predictions = ticker_predictions.dropna(subset=['predicted_return']).reset_index(drop=True)

            ticker_predictions = self.convert_returns_to_prices(ticker_predictions, ticker)

            return ticker_predictions

        except Exception as e:
            logger.error(f"Error getting predictions for {ticker}: {e}")
            return None

    def _discover_xgboost_models(self) -> Dict[int, str]:
        """Discover available XGBoost models keyed by prediction horizon"""
        models = {}

        try:
            if not os.path.exists(self.models_dir):
                return models

            for filename in os.listdir(self.models_dir):
                if not filename.endswith('.pkl') or 'xgboost_' not in filename:
                    continue

                parts = filename.split('_')
                if len(parts) < 2:
                    continue

                horizon_str = parts[1].replace('w', '')
                try:
                    horizon = int(horizon_str)
                except ValueError:
                    continue

                models[horizon] = os.path.join(self.models_dir, filename)

        except Exception as e:
            logger.debug(f"Error discovering XGBoost models: {e}")

        return models

    def get_historical_data(self, ticker: str, periods: int = 28) -> Optional[pd.DataFrame]:
        """
        Get historical stock data for a ticker

        Args:
            ticker: Stock ticker symbol
            periods: Number of historical periods to fetch

        Returns:
            DataFrame with historical data
        """
        try:
            import yfinance as yf

            logger.info(f"Fetching historical data for {ticker}...")
            stock = yf.Ticker(ticker)

            # Get recent historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods + 10)

            # Convert to string format for yfinance to avoid pandas timestamp issues
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')

            hist = stock.history(start=start_date_str, end=end_date_str)

            if not hist.empty:
                hist = hist.reset_index()
                hist['ticker'] = ticker

                # Handle timezone-aware dates properly
                if 'Date' in hist.columns:
                    hist['date'] = pd.to_datetime(hist['Date'])
                    if hist['date'].dt.tz is not None:
                        hist['date'] = hist['date'].dt.tz_localize(None)
                else:
                    hist['date'] = pd.to_datetime(hist.index)
                    if hist['date'].dt.tz is not None:
                        hist['date'] = hist['date'].dt.tz_localize(None)

                hist['stock_price'] = hist['Close']
                return hist[['ticker', 'date', 'stock_price']].tail(periods)

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")

        return None

    def convert_returns_to_prices(self, predictions_df: pd.DataFrame,
                                ticker: str) -> pd.DataFrame:
        """
        Convert predicted returns to actual stock prices

        Args:
            predictions_df: DataFrame with predicted returns
            ticker: Stock ticker symbol

        Returns:
            DataFrame with stock prices calculated
        """
        try:
            if predictions_df.empty:
                return predictions_df

            df = predictions_df.copy()

            if 'predicted_return' not in df.columns:
                logger.debug("No predicted_return column available for price conversion")
                return df

            # Get starting price from current market data
            starting_price = self._get_current_price(ticker)
            if starting_price is None:
                logger.warning(f"Could not get current price for {ticker}, using default")
                starting_price = 100.0

            # Convert returns to prices
            df['stock_price'] = 0.0

            for i in range(len(df)):
                predicted_return = df.iloc[i]['predicted_return']

                # Handle both decimal and percentage formats
                if abs(predicted_return) < 1:
                    return_multiplier = predicted_return  # Already in decimal form
                else:
                    return_multiplier = predicted_return / 100  # Convert percentage

                if i == 0:
                    # First prediction: apply return to starting price
                    week_close = starting_price * (1 + return_multiplier)
                    df.iloc[i, df.columns.get_loc('stock_price')] = week_close
                else:
                    # Subsequent predictions: compound from previous week
                    prev_week_close = df.iloc[i-1]['stock_price']
                    current_week_close = prev_week_close * (1 + return_multiplier)
                    df.iloc[i, df.columns.get_loc('stock_price')] = current_week_close

            # Add confidence interval prices if available
            if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                df = self._convert_ci_to_prices(df, starting_price)

            return df

        except Exception as e:
            logger.error(f"Error converting returns to prices for {ticker}: {e}")
            return predictions_df

    def _convert_ci_to_prices(self, df: pd.DataFrame, starting_price: float) -> pd.DataFrame:
        """Convert confidence interval returns to prices"""
        try:
            df['ci_lower_price'] = 0.0
            df['ci_upper_price'] = 0.0

            for i in range(len(df)):
                ci_lower = df.iloc[i]['ci_lower']
                ci_upper = df.iloc[i]['ci_upper']

                # Handle both decimal and percentage formats
                if abs(ci_lower) < 1 and abs(ci_upper) < 1:
                    ci_lower_mult = ci_lower
                    ci_upper_mult = ci_upper
                else:
                    ci_lower_mult = ci_lower / 100
                    ci_upper_mult = ci_upper / 100

                if i == 0:
                    base_price = starting_price
                else:
                    # Use previous CI prices as base
                    base_price_lower = df.iloc[i-1]['ci_lower_price']
                    base_price_upper = df.iloc[i-1]['ci_upper_price']
                    base_price = (base_price_lower + base_price_upper) / 2

                df.iloc[i, df.columns.get_loc('ci_lower_price')] = base_price * (1 + ci_lower_mult)
                df.iloc[i, df.columns.get_loc('ci_upper_price')] = base_price * (1 + ci_upper_mult)

            return df

        except Exception as e:
            logger.error(f"Error converting CI to prices: {e}")
            return df

    def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get current stock price for a ticker"""
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")

            if not hist.empty:
                return float(hist['Close'].iloc[-1])

        except Exception as e:
            logger.debug(f"Could not fetch current price for {ticker}: {e}")

        return None

    def _should_refresh_cache(self) -> bool:
        """Check if predictions cache should be refreshed"""
        if self._cache_timestamp is None:
            return True

        # Refresh cache if it's older than 5 minutes
        try:
            cache_age = datetime.now() - self._cache_timestamp
            return cache_age.total_seconds() > 300
        except TypeError as e:
            logger.debug(f"Error calculating cache age: {e}")
            # If there's an error, refresh the cache
            return True

    def get_model_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get information about a specific model

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with model information
        """
        try:
            if self.model_type == 'arimax':
                model_path = os.path.join(self.models_dir, f"{ticker}_arimax.pkl")

                if not os.path.exists(model_path):
                    return {'exists': False, 'error': f'Model not found for {ticker}'}
            else:
                # XGBoost uses cross-sectional models named by horizon
                horizon_models = self._discover_xgboost_models()
                if not horizon_models:
                    return {'exists': False, 'error': 'No XGBoost models found'}
                model_path = list(horizon_models.values())[0]

            # Get file stats
            stat = os.stat(model_path)
            info = {
                'exists': True,
                'ticker': ticker,
                'model_path': model_path,
                'file_size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

            # Try to load model to get more details
            try:
                if self.model_type == 'arimax':
                    import joblib
                    model_data = joblib.load(model_path)

                    if isinstance(model_data, dict):
                        info.update({
                            'model_order': model_data.get('best_order'),
                            'aic_score': model_data.get('aic_score'),
                            'feature_columns': model_data.get('feature_columns', [])
                        })
                else:
                    info['model_type'] = 'xgboost'
                    info['available_horizons'] = sorted(self._discover_xgboost_models().keys())

            except Exception as e:
                logger.debug(f"Could not load model details for {ticker}: {e}")

            return info

        except Exception as e:
            logger.error(f"Error getting model info for {ticker}: {e}")
            return {'exists': False, 'error': str(e)}

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all models and predictions

        Returns:
            Dict with summary information
        """
        try:
            available_tickers = self.get_available_tickers()
            summary = {
                'model_type': self.model_type,
                'available_models': len(available_tickers),
                'tickers': available_tickers
            }

            # Get predictions info
            predictions_df = self.get_latest_predictions()
            if predictions_df is not None:
                summary.update({
                    'total_predictions': len(predictions_df),
                    'prediction_stocks': predictions_df['ticker'].nunique(),
                    'prediction_date_range': {
                        'start': predictions_df['date'].min().strftime('%Y-%m-%d'),
                        'end': predictions_df['date'].max().strftime('%Y-%m-%d')
                    }
                })

                if 'predicted_return' in predictions_df.columns:
                    returns = predictions_df['predicted_return']
                    summary['return_stats'] = {
                        'mean': float(returns.mean()),
                        'std': float(returns.std()),
                        'min': float(returns.min()),
                        'max': float(returns.max())
                    }

            return summary

        except Exception as e:
            logger.error(f"Error getting summary stats: {e}")
            return {'error': str(e)}
