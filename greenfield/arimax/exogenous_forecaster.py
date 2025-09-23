#!/usr/bin/env python3
"""
Exogenous Forecasting Pipeline

This module coordinates individual regressors to forecast all exogenous variables
needed for true ARIMAX predictions. It handles lag feature generation and
uncertainty propagation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings

from exogenous_regressors import (
    create_regressor,
    create_var_regressor,
    ARIMARegressor,
    VARRegressor,
    ExogenousRegressor
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExogenousForecaster:
    """
    Main pipeline for forecasting exogenous variables.

    Coordinates individual regressors and generates the complete set of
    exogenous features (including lags) needed for ARIMAX prediction.
    """

    def __init__(self, forecasting_mode: str = 'individual'):
        """
        Initialize the forecaster.

        Args:
            forecasting_mode: 'individual' (separate models) or 'var' (joint VAR model)
        """
        self.forecasting_mode = forecasting_mode
        self.base_variables = ['high_return', 'low_return', 'volume_change', 'volatility']
        self.lag_steps = 3

        # Individual regressors
        self.regressors = {}

        # VAR regressor for joint modeling
        self.var_regressor = None

        # Historical data for lag generation
        self.historical_data = None
        self.is_fitted = False

    def _prepare_stock_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare data for a specific stock."""
        stock_data = df[df['ticker'] == ticker].copy()

        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Sort by date
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)

        # Check for required variables
        missing_vars = [var for var in self.base_variables if var not in stock_data.columns]
        if missing_vars:
            raise ValueError(f"Missing required variables for {ticker}: {missing_vars}")

        return stock_data

    def fit_individual_regressors(self, df: pd.DataFrame, ticker: str) -> None:
        """Fit individual regressors for each exogenous variable."""
        stock_data = self._prepare_stock_data(df, ticker)

        # Store historical data for lag generation
        self.historical_data = stock_data[self.base_variables].tail(self.lag_steps).copy()

        logger.info(f"Fitting individual regressors for {ticker}")

        # Fit each variable separately
        for variable in self.base_variables:
            try:
                regressor = create_regressor(variable, regressor_type='auto')
                regressor.fit(stock_data[variable])
                self.regressors[variable] = regressor

                # Log model info
                info = regressor.get_model_info()
                logger.info(f"  {variable}: {info.get('model_type', 'Unknown')} fitted")

            except Exception as e:
                logger.warning(f"Failed to fit {variable} for {ticker}: {e}")
                # Use persistence as fallback
                fallback = create_regressor(variable, regressor_type='persistence')
                fallback.fit(stock_data[variable])
                self.regressors[variable] = fallback
                logger.info(f"  {variable}: Using persistence fallback")

    def fit_var_regressor(self, df: pd.DataFrame, ticker: str) -> None:
        """Fit VAR regressor for joint modeling."""
        stock_data = self._prepare_stock_data(df, ticker)

        # Store historical data for lag generation
        self.historical_data = stock_data[self.base_variables].tail(self.lag_steps).copy()

        logger.info(f"Fitting VAR regressor for {ticker}")

        try:
            self.var_regressor = create_var_regressor(self.base_variables)
            self.var_regressor.fit(stock_data[self.base_variables])

            info = self.var_regressor.get_model_info()
            logger.info(f"  VAR({info.get('optimal_lags', 'Unknown')}) fitted")

        except Exception as e:
            logger.error(f"VAR fitting failed for {ticker}: {e}")
            # Fallback to individual regressors
            logger.info("Falling back to individual regressors")
            self.forecasting_mode = 'individual'
            self.fit_individual_regressors(df, ticker)
            return

    def fit(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Fit forecasting models for the specified ticker.

        Args:
            df: DataFrame with historical data
            ticker: Stock ticker to fit models for
        """
        if self.forecasting_mode == 'var':
            self.fit_var_regressor(df, ticker)
        else:
            self.fit_individual_regressors(df, ticker)

        self.is_fitted = True

    def forecast_base_variables(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Forecast the base exogenous variables.

        Args:
            steps: Number of steps to forecast
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with forecasts for each variable
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before prediction")

        forecasts = {}

        if self.forecasting_mode == 'var' and self.var_regressor:
            # Use VAR for joint forecasting
            var_forecasts = self.var_regressor.predict(steps, confidence_level)
            forecasts.update(var_forecasts)

        else:
            # Use individual regressors
            for variable in self.base_variables:
                if variable in self.regressors:
                    predictions, lower, upper = self.regressors[variable].predict(steps, confidence_level)
                    forecasts[variable] = (predictions, lower, upper)
                else:
                    # Create zero forecasts as fallback
                    predictions = np.zeros(steps)
                    bounds = np.zeros(steps)
                    forecasts[variable] = (predictions, bounds, bounds)
                    logger.warning(f"No regressor for {variable}, using zero forecasts")

        return forecasts

    def generate_lag_features(self, base_forecasts: Dict[str, np.ndarray], steps: int) -> pd.DataFrame:
        """
        Generate lagged features for the forecasted periods.

        Args:
            base_forecasts: Forecasted values for base variables
            steps: Number of forecasting steps

        Returns:
            DataFrame with all features including lags
        """
        if self.historical_data is None:
            raise ValueError("No historical data available for lag generation")

        # Create extended time series combining historical and forecasted data
        extended_data = {}

        for variable in self.base_variables:
            # Get historical values
            historical_values = self.historical_data[variable].values

            # Get forecasted values
            if variable in base_forecasts:
                forecasted_values = base_forecasts[variable]
            else:
                forecasted_values = np.zeros(steps)

            # Combine historical and forecasted
            extended_series = np.concatenate([historical_values, forecasted_values])
            extended_data[variable] = extended_series

        # Generate features for each forecasting step
        feature_rows = []

        for step in range(steps):
            row = {}

            # Current period features (from forecasts)
            for variable in self.base_variables:
                # Position in extended series: len(historical) + step
                current_idx = len(self.historical_data) + step
                row[variable] = extended_data[variable][current_idx]

            # Lag features
            for lag in range(1, self.lag_steps + 1):
                for variable in self.base_variables:
                    # Lag position: current - lag
                    lag_idx = len(self.historical_data) + step - lag

                    if lag_idx >= 0:
                        lag_value = extended_data[variable][lag_idx]
                    else:
                        # If we need lags beyond available data, use the earliest available
                        lag_value = extended_data[variable][0]

                    row[f"{variable}_lag_{lag}"] = lag_value

            feature_rows.append(row)

        return pd.DataFrame(feature_rows)

    def forecast_exogenous_features(self, steps: int, confidence_level: float = 0.95) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate complete exogenous feature forecasts including lags.

        Args:
            steps: Number of steps to forecast
            confidence_level: Confidence level for uncertainty bounds

        Returns:
            features_df: DataFrame with all exogenous features
            uncertainty_info: Dictionary with uncertainty information
        """
        # Forecast base variables
        base_forecasts = self.forecast_base_variables(steps, confidence_level)

        # Extract point forecasts for lag generation
        point_forecasts = {
            var: forecasts[0] for var, forecasts in base_forecasts.items()
        }

        # Generate complete feature set with lags
        features_df = self.generate_lag_features(point_forecasts, steps)

        # Collect uncertainty information
        uncertainty_info = {
            'base_forecasts': base_forecasts,
            'confidence_level': confidence_level,
            'forecasting_mode': self.forecasting_mode
        }

        return features_df, uncertainty_info

    def get_model_summary(self) -> Dict:
        """Get summary of fitted models."""
        if not self.is_fitted:
            return {'error': 'No models fitted'}

        summary = {
            'forecasting_mode': self.forecasting_mode,
            'base_variables': self.base_variables,
            'lag_steps': self.lag_steps
        }

        if self.forecasting_mode == 'var' and self.var_regressor:
            summary['var_model'] = self.var_regressor.get_model_info()
        else:
            summary['individual_models'] = {}
            for var, regressor in self.regressors.items():
                summary['individual_models'][var] = regressor.get_model_info()

        return summary

    def validate_forecasts(self, features_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that forecasted features are reasonable.

        Args:
            features_df: DataFrame with forecasted features

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Check for NaN values
        validation_results['no_nan_values'] = not features_df.isnull().any().any()

        # Check for extreme values (beyond 5 standard deviations of historical data)
        if self.historical_data is not None:
            extreme_values = False
            for variable in self.base_variables:
                if variable in features_df.columns:
                    historical_std = self.historical_data[variable].std()
                    historical_mean = self.historical_data[variable].mean()

                    forecasted_values = features_df[variable]
                    z_scores = abs((forecasted_values - historical_mean) / historical_std)

                    if (z_scores > 5).any():
                        extreme_values = True
                        break

            validation_results['no_extreme_values'] = not extreme_values

        # Check feature completeness
        expected_features = []
        for var in self.base_variables:
            expected_features.append(var)
            for lag in range(1, self.lag_steps + 1):
                expected_features.append(f"{var}_lag_{lag}")

        validation_results['all_features_present'] = all(
            feature in features_df.columns for feature in expected_features
        )

        return validation_results

# Convenience functions for easy usage

def create_forecaster(mode: str = 'individual') -> ExogenousForecaster:
    """Create and return a configured ExogenousForecaster."""
    return ExogenousForecaster(forecasting_mode=mode)

def quick_forecast(df: pd.DataFrame, ticker: str, steps: int = 4, mode: str = 'individual') -> Tuple[pd.DataFrame, Dict]:
    """
    Quick forecast generation for a single stock.

    Args:
        df: Historical data
        ticker: Stock ticker
        steps: Number of periods to forecast
        mode: Forecasting mode ('individual' or 'var')

    Returns:
        features_df: Forecasted exogenous features
        summary: Model and forecast summary
    """
    forecaster = create_forecaster(mode)
    forecaster.fit(df, ticker)

    features_df, uncertainty_info = forecaster.forecast_exogenous_features(steps)

    summary = {
        'ticker': ticker,
        'forecast_steps': steps,
        'model_summary': forecaster.get_model_summary(),
        'uncertainty_info': uncertainty_info,
        'validation': forecaster.validate_forecasts(features_df)
    }

    return features_df, summary