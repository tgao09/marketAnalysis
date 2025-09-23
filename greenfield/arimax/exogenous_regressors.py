#!/usr/bin/env python3
"""
Individual regressor classes for forecasting exogenous variables.
Each regressor is designed based on the variable characteristics analysis.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExogenousRegressor(ABC):
    """
    Abstract base class for exogenous variable regressors.
    """

    def __init__(self, variable_name: str):
        self.variable_name = variable_name
        self.is_fitted = False
        self.model = None
        self.fit_history = None

    @abstractmethod
    def fit(self, data: pd.Series) -> None:
        """Fit the regressor to historical data."""
        pass

    @abstractmethod
    def predict(self, steps: int, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts.

        Returns:
            predictions: Point forecasts
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        pass

class ARIMARegressor(ExogenousRegressor):
    """
    ARIMA regressor for stationary variables with autocorrelation.
    Best for: high_return, low_return
    """

    def __init__(self, variable_name: str, max_p: int = 5, max_d: int = 1, max_q: int = 3):
        super().__init__(variable_name)
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order = None
        self.aic_score = None

    def _find_best_order(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using AIC."""
        best_aic = np.inf
        best_order = (1, 0, 1)

        # Grid search for best parameters
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)

                    except Exception:
                        continue

        return best_order

    def fit(self, data: pd.Series) -> None:
        """Fit ARIMA model to the data."""
        clean_data = data.dropna()

        if len(clean_data) < 30:
            raise ValueError(f"Insufficient data for {self.variable_name}: need at least 30 observations")

        # Find best order
        self.best_order = self._find_best_order(clean_data)

        # Fit final model
        self.model = ARIMA(clean_data, order=self.best_order)
        self.fitted_model = self.model.fit()
        self.aic_score = self.fitted_model.aic
        self.fit_history = clean_data
        self.is_fitted = True

        logger.info(f"{self.variable_name}: ARIMA{self.best_order} fitted, AIC={self.aic_score:.2f}")

    def predict(self, steps: int, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ARIMA forecasts."""
        if not self.is_fitted:
            raise ValueError(f"Model for {self.variable_name} must be fitted before prediction")

        # Get forecast with confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1-confidence_level)

        lower_bounds = conf_int.iloc[:, 0].values
        upper_bounds = conf_int.iloc[:, 1].values

        return predictions.values, lower_bounds, upper_bounds

    def get_model_info(self) -> Dict:
        """Get ARIMA model information."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        return {
            'model_type': 'ARIMA',
            'order': self.best_order,
            'aic': self.aic_score,
            'variable': self.variable_name,
            'n_observations': len(self.fit_history)
        }

class VARRegressor(ExogenousRegressor):
    """
    Vector Autoregression for jointly modeling correlated variables.
    Best for: modeling high_return, low_return, volume_change, volatility together
    """

    def __init__(self, variable_names: List[str], max_lags: int = 5):
        # For VAR, we use the first variable name as primary
        super().__init__(variable_names[0] if variable_names else "VAR")
        self.variable_names = variable_names
        self.max_lags = max_lags
        self.optimal_lags = None
        self.variable_order = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit VAR model to multivariate data."""
        # Select only the specified variables
        model_data = data[self.variable_names].dropna()

        if len(model_data) < 50:
            raise ValueError(f"Insufficient data for VAR: need at least 50 observations")

        # Fit VAR model
        self.model = VAR(model_data)

        # Select optimal lag order using AIC
        lag_results = self.model.select_order(maxlags=self.max_lags)
        self.optimal_lags = lag_results.aic

        # Fit with optimal lags
        self.fitted_model = self.model.fit(self.optimal_lags)
        self.variable_order = self.fitted_model.names
        self.fit_history = model_data
        self.is_fitted = True

        logger.info(f"VAR({self.optimal_lags}) fitted for {self.variable_names}, AIC={self.fitted_model.aic:.2f}")

    def predict(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate VAR forecasts for all variables."""
        if not self.is_fitted:
            raise ValueError("VAR model must be fitted before prediction")

        # Get forecast
        forecast_result = self.fitted_model.forecast_interval(
            y=self.fit_history.values[-self.optimal_lags:],
            steps=steps,
            alpha=1-confidence_level
        )

        predictions = forecast_result[0]
        lower_bounds = forecast_result[1]
        upper_bounds = forecast_result[2]

        # Organize results by variable
        results = {}
        for i, var_name in enumerate(self.variable_order):
            results[var_name] = (
                predictions[:, i],
                lower_bounds[:, i],
                upper_bounds[:, i]
            )

        return results

    def get_model_info(self) -> Dict:
        """Get VAR model information."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        return {
            'model_type': 'VAR',
            'optimal_lags': self.optimal_lags,
            'aic': self.fitted_model.aic,
            'variables': self.variable_names,
            'n_observations': len(self.fit_history)
        }

class GARCHRegressor(ExogenousRegressor):
    """
    GARCH regressor for variables with volatility clustering.
    Best for: volatility, volume_change
    Note: Using SARIMAX with time-varying variance as proxy for GARCH
    """

    def __init__(self, variable_name: str, ar_lags: int = 2, ma_lags: int = 1):
        super().__init__(variable_name)
        self.ar_lags = ar_lags
        self.ma_lags = ma_lags
        self.model_order = None

    def fit(self, data: pd.Series) -> None:
        """Fit GARCH-like model using SARIMAX."""
        clean_data = data.dropna()

        if len(clean_data) < 50:
            raise ValueError(f"Insufficient data for {self.variable_name}: need at least 50 observations")

        # Use SARIMAX as approximation to GARCH
        # This captures some volatility clustering through AR/MA terms
        self.model_order = (self.ar_lags, 0, self.ma_lags)

        self.model = SARIMAX(
            clean_data,
            order=self.model_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        self.fitted_model = self.model.fit(disp=False)
        self.fit_history = clean_data
        self.is_fitted = True

        logger.info(f"{self.variable_name}: GARCH-like SARIMAX{self.model_order} fitted")

    def predict(self, steps: int, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate GARCH-like forecasts."""
        if not self.is_fitted:
            raise ValueError(f"Model for {self.variable_name} must be fitted before prediction")

        # Get forecast with prediction intervals
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1-confidence_level)

        lower_bounds = conf_int.iloc[:, 0].values
        upper_bounds = conf_int.iloc[:, 1].values

        return predictions.values, lower_bounds, upper_bounds

    def get_model_info(self) -> Dict:
        """Get GARCH model information."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        return {
            'model_type': 'GARCH-like (SARIMAX)',
            'order': self.model_order,
            'aic': self.fitted_model.aic,
            'variable': self.variable_name,
            'n_observations': len(self.fit_history)
        }

class PersistenceRegressor(ExogenousRegressor):
    """
    Simple persistence/random walk regressor as baseline.
    Useful as fallback or benchmark.
    """

    def __init__(self, variable_name: str):
        super().__init__(variable_name)
        self.last_value = None
        self.historical_std = None

    def fit(self, data: pd.Series) -> None:
        """Fit persistence model (just store last value and volatility)."""
        clean_data = data.dropna()

        if len(clean_data) < 10:
            raise ValueError(f"Insufficient data for {self.variable_name}: need at least 10 observations")

        self.last_value = clean_data.iloc[-1]
        self.historical_std = clean_data.std()
        self.fit_history = clean_data
        self.is_fitted = True

        logger.info(f"{self.variable_name}: Persistence model fitted, last_value={self.last_value:.4f}")

    def predict(self, steps: int, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate persistence forecasts (constant prediction with expanding uncertainty)."""
        if not self.is_fitted:
            raise ValueError(f"Model for {self.variable_name} must be fitted before prediction")

        # Constant predictions
        predictions = np.full(steps, self.last_value)

        # Expanding uncertainty
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% confidence
        expanding_std = self.historical_std * np.sqrt(np.arange(1, steps + 1))

        lower_bounds = predictions - z_score * expanding_std
        upper_bounds = predictions + z_score * expanding_std

        return predictions, lower_bounds, upper_bounds

    def get_model_info(self) -> Dict:
        """Get persistence model information."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        return {
            'model_type': 'Persistence',
            'last_value': self.last_value,
            'historical_std': self.historical_std,
            'variable': self.variable_name,
            'n_observations': len(self.fit_history)
        }

def create_regressor(variable_name: str, regressor_type: str = 'auto') -> ExogenousRegressor:
    """
    Factory function to create appropriate regressor based on variable characteristics.

    Args:
        variable_name: Name of the exogenous variable
        regressor_type: Type of regressor ('auto', 'arima', 'garch', 'persistence')

    Returns:
        Configured regressor instance
    """
    if regressor_type == 'auto':
        # Automatic selection based on variable analysis
        if variable_name in ['high_return', 'low_return']:
            return ARIMARegressor(variable_name, max_p=5, max_d=0, max_q=3)
        elif variable_name == 'volatility':
            return GARCHRegressor(variable_name, ar_lags=3, ma_lags=2)
        elif variable_name == 'volume_change':
            return GARCHRegressor(variable_name, ar_lags=2, ma_lags=1)
        else:
            return PersistenceRegressor(variable_name)
    elif regressor_type == 'arima':
        return ARIMARegressor(variable_name)
    elif regressor_type == 'garch':
        return GARCHRegressor(variable_name)
    elif regressor_type == 'persistence':
        return PersistenceRegressor(variable_name)
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

def create_var_regressor(variable_names: List[str]) -> VARRegressor:
    """
    Create VAR regressor for joint modeling of correlated variables.

    Args:
        variable_names: List of variable names to model jointly

    Returns:
        Configured VAR regressor
    """
    return VARRegressor(variable_names, max_lags=5)