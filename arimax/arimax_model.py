import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockARIMAX:
    

    def __init__(self, ticker: str, max_p: int = 5, max_d: int = 2, max_q: int = 5,
                 selected_features: List[str] = None):
        
        self.ticker = ticker
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.selected_features = selected_features
        self.model = None
        self.fitted_model = None
        self.best_order = None
        self.aic_score = None
        self.feature_columns = None
        self.is_fitted = False

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        
        # ensure we have data for this ticker
        stock_data = df[df['ticker'] == self.ticker].copy()

        if stock_data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")

        # sort by date to ensure proper time series order
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)

        # target variable
        target = stock_data['weekly_return']

        # exclude target's own lags, non-feature columns, and contemporaneous features (data leakage)
        exclude_columns = [
            'ticker', 'Date', 'weekly_return',
            'weekly_return_lag_1', 'weekly_return_lag_2', 'weekly_return_lag_3',
            'weekly_return_lag_4', 'weekly_return_lag_5',
            # exclude contemporaneous features from same week as target (data leakage)
            'high_return', 'low_return', 'volume_change', 'volatility'
        ]

        # get all available columns
        all_columns = stock_data.columns.tolist()

        # select exogenous variables (everything except excluded columns)
        exog_columns = [col for col in all_columns if col not in exclude_columns]
        exog = stock_data[exog_columns]

        # drop columns that are entirely nan (e.g., missing tickers like ^vix)
        exog = exog.dropna(axis=1, how='all')

        # filter to selected features if specified
        if self.selected_features is not None:
            available_features = [f for f in self.selected_features if f in exog.columns]
            if len(available_features) < len(self.selected_features):
                missing = set(self.selected_features) - set(available_features)
                logger.warning(f"{self.ticker}: {len(missing)} selected features not found in data")
            exog = exog[available_features]
            logger.info(f"{self.ticker}: Using {len(available_features)} selected features")

        # update feature columns after filtering
        self.feature_columns = exog.columns.tolist()

        # remove rows with nan values and reset indices
        valid_indices = (~target.isna()) & (~exog.isna().any(axis=1))
        target = target[valid_indices].reset_index(drop=True)
        exog = exog[valid_indices].reset_index(drop=True)

        logger.info(f"{self.ticker}: Prepared {len(target)} observations with {len(self.feature_columns)} exogenous variables")

        return target, exog

    def check_stationarity(self, series: pd.Series, alpha: float = 0.05) -> bool:
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(series.dropna())
                p_value = result[1]
                return p_value < alpha
        except Exception as e:
            logger.warning(f"{self.ticker}: Stationarity test failed: {e}")
            return False

    def time_series_cv_split(self, target: pd.Series, exog: pd.DataFrame,
                            n_splits: int = 5, min_train_size: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        
        n_samples = len(target)

        if n_samples < min_train_size + n_splits:
            raise ValueError(f"{self.ticker}: Not enough data for {n_splits} CV splits")

        # calculate test size for each split
        available_for_testing = n_samples - min_train_size
        test_size = max(1, available_for_testing // n_splits)

        splits = []
        for i in range(n_splits):
            # expanding window: train size grows with each split
            train_end = min_train_size + i * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            # ensure we have at least 1 test sample
            if len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.info(f"{self.ticker}: Created {len(splits)} CV splits with min_train_size={min_train_size}")
        return splits

    def find_optimal_order(self, target: pd.Series, exog: pd.DataFrame, use_cv: bool = True) -> Tuple[int, int, int]:
        
        # determine max differencing needed
        max_d_needed = 0
        temp_series = target.copy()

        for d in range(self.max_d + 1):
            if self.check_stationarity(temp_series):
                max_d_needed = d
                break
            if d < self.max_d:
                temp_series = temp_series.diff().dropna()

        logger.info(f"{self.ticker}: Testing ARIMA orders up to ({self.max_p}, {max_d_needed}, {self.max_q})")

        if use_cv and len(target) >= 50:  # use cv only if we have sufficient data
            return self._find_optimal_order_cv(target, exog, max_d_needed)
        else:
            return self._find_optimal_order_aic(target, exog, max_d_needed)

    def _find_optimal_order_aic(self, target: pd.Series, exog: pd.DataFrame, max_d_needed: int) -> Tuple[int, int, int]:
        
        best_aic = np.inf
        best_order = (1, 0, 1)  # default fallback

        # grid search over (p, d, q)
        for p in range(self.max_p + 1):
            for d in range(min(max_d_needed + 1, self.max_d + 1)):
                for q in range(self.max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # ensure target and exog have consistent indices and clean data
                            target_clean = target.copy().reset_index(drop=True)
                            exog_clean = exog.copy().reset_index(drop=True)

                            # convert to numpy arrays to avoid pandas index issues
                            target_array = target_clean.values
                            exog_array = exog_clean.values

                            # validate data shapes and types
                            if len(target_array) != len(exog_array):
                                continue

                            # fit arima model with numpy arrays
                            model = ARIMA(target_array, exog=exog_array, order=(p, d, q))
                            fitted = model.fit()

                            # check if aic is better
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)

                    except Exception as e:
                        # skip this combination if it fails
                        logger.debug(f"{self.ticker}: Failed to fit order ({p},{d},{q}): {str(e)[:50]}")
                        continue

        logger.info(f"{self.ticker}: Optimal order {best_order} with AIC={best_aic:.2f}")
        return best_order

    def _find_optimal_order_cv(self, target: pd.Series, exog: pd.DataFrame, max_d_needed: int) -> Tuple[int, int, int]:
        
        best_cv_score = np.inf
        best_order = (1, 0, 1)  # default fallback

        # create cv splits
        try:
            cv_splits = self.time_series_cv_split(target, exog, n_splits=3, min_train_size=30)
        except ValueError as e:
            logger.warning(f"{self.ticker}: CV failed, falling back to AIC: {e}")
            return self._find_optimal_order_aic(target, exog, max_d_needed)

        # grid search over (p, d, q) with cv
        for p in range(self.max_p + 1):
            for d in range(min(max_d_needed + 1, self.max_d + 1)):
                for q in range(self.max_q + 1):
                    cv_scores = []

                    # evaluate on each cv fold
                    for train_idx, test_idx in cv_splits:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")

                                # split data for this fold
                                train_target = target.iloc[train_idx].values
                                train_exog = exog.iloc[train_idx].values
                                test_target = target.iloc[test_idx].values
                                test_exog = exog.iloc[test_idx].values

                                # fit model on training fold
                                model = ARIMA(train_target, exog=train_exog, order=(p, d, q))
                                fitted = model.fit()

                                # predict on test fold
                                pred = fitted.forecast(steps=len(test_target), exog=test_exog)

                                # calculate rmse for this fold
                                rmse = np.sqrt(mean_squared_error(test_target, pred))
                                cv_scores.append(rmse)

                        except Exception as e:
                            # if this fold fails, assign high penalty score
                            cv_scores.append(np.inf)
                            logger.debug(f"{self.ticker}: CV fold failed for order ({p},{d},{q}): {str(e)[:50]}")

                    # calculate mean cv score (skip if all folds failed)
                    if cv_scores and any(score < np.inf for score in cv_scores):
                        mean_cv_score = np.mean([s for s in cv_scores if s < np.inf])

                        if mean_cv_score < best_cv_score:
                            best_cv_score = mean_cv_score
                            best_order = (p, d, q)

        logger.info(f"{self.ticker}: Optimal order {best_order} with CV RMSE={best_cv_score:.4f}")
        return best_order

    def fit(self, df: pd.DataFrame, train_size: float = 0.8, use_cv: bool = True) -> Dict[str, Any]:
        
        # prepare data
        target, exog = self.prepare_data(df)

        if len(target) < 20:
            raise ValueError(f"{self.ticker}: Insufficient data for modeling (need at least 20 observations)")

        # split data
        split_idx = int(len(target) * train_size)
        train_target = target[:split_idx].copy()
        train_exog = exog[:split_idx].copy()
        test_target = target[split_idx:].copy()
        test_exog = exog[split_idx:].copy()

        # reset indices to ensure clean data
        train_target = train_target.reset_index(drop=True)
        train_exog = train_exog.reset_index(drop=True)
        test_target = test_target.reset_index(drop=True)
        test_exog = test_exog.reset_index(drop=True)

        # find optimal order with optional cv
        self.best_order = self.find_optimal_order(train_target, train_exog, use_cv=use_cv)

        # fit final model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # convert to numpy arrays to avoid pandas series comparison issues
                train_target_array = train_target.values
                train_exog_array = train_exog.values

                self.model = ARIMA(train_target_array, exog=train_exog_array, order=self.best_order)
                self.fitted_model = self.model.fit()
                self.aic_score = self.fitted_model.aic
                self.is_fitted = True

                cv_method = "CV" if use_cv and len(train_target) >= 50 else "AIC"
                logger.info(f"{self.ticker}: Model fitted successfully with order {self.best_order} (selected via {cv_method})")

        except Exception as e:
            logger.error(f"{self.ticker}: Model fitting failed: {e}")
            raise

        # validate on test set if available
        results = {
            'ticker': self.ticker,
            'order': self.best_order,
            'aic': self.aic_score,
            'train_size': len(train_target),
            'test_size': len(test_target),
            'total_features': len(self.feature_columns),
            'selection_method': "CV" if use_cv and len(train_target) >= 50 else "AIC"
        }

        if len(test_target) > 0:
            # generate predictions for test set
            test_pred = self.predict(test_exog, steps=len(test_target))

            # calculate test metrics
            results['test_mae'] = mean_absolute_error(test_target, test_pred)
            results['test_rmse'] = np.sqrt(mean_squared_error(test_target, test_pred))
            results['test_mape'] = np.mean(np.abs((test_target - test_pred) / test_target)) * 100

            # directional accuracy
            direction_correct = np.sign(test_target) == np.sign(test_pred)
            results['directional_accuracy'] = np.mean(direction_correct) * 100

            logger.info(f"{self.ticker}: Test RMSE={results['test_rmse']:.4f}, "
                       f"Directional Accuracy={results['directional_accuracy']:.1f}%")

        return results

    def predict(self, exog: pd.DataFrame, steps: int = 1, return_conf_int: bool = False) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError(f"{self.ticker}: Model must be fitted before prediction")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # convert exog to numpy array if it's a dataframe
                if hasattr(exog, 'values'):
                    exog_array = exog[:steps].values
                else:
                    exog_array = exog[:steps]

                if return_conf_int:
                    # use get_forecast for confidence intervals
                    forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog_array)
                    pred = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int()
                    return pred, conf_int
                else:
                    pred = self.fitted_model.forecast(steps=steps, exog=exog_array)
                    return pred

        except Exception as e:
            logger.error(f"{self.ticker}: Prediction failed: {e}")
            raise

    def save_model(self, filepath: str) -> None:
        
        if not self.is_fitted:
            raise ValueError(f"{self.ticker}: Cannot save unfitted model")

        model_data = {
            'ticker': self.ticker,
            'fitted_model': self.fitted_model,
            'best_order': self.best_order,
            'aic_score': self.aic_score,
            'feature_columns': self.feature_columns
        }

        joblib.dump(model_data, filepath)
        logger.info(f"{self.ticker}: Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'StockARIMAX':
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        # create instance with selected_features from saved model
        instance = cls(model_data['ticker'], selected_features=model_data['feature_columns'])
        instance.fitted_model = model_data['fitted_model']
        instance.best_order = model_data['best_order']
        instance.aic_score = model_data['aic_score']
        instance.feature_columns = model_data['feature_columns']
        instance.is_fitted = True

        logger.info(f"{instance.ticker}: Model loaded from {filepath}")
        return instance

    def get_feature_importance(self) -> pd.DataFrame:
        
        if not self.is_fitted:
            raise ValueError(f"{self.ticker}: Model must be fitted first")

        # get exogenous variable coefficients
        if hasattr(self.fitted_model, 'params') and len(self.feature_columns) > 0:
            # find exog coefficients (skip arima parameters)
            total_params = len(self.fitted_model.params)
            arima_params = self.best_order[0] + self.best_order[2]  # p + q parameters

            if total_params > arima_params:
                exog_coeffs = self.fitted_model.params[-len(self.feature_columns):]

                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'coefficient': exog_coeffs,
                    'abs_coefficient': np.abs(exog_coeffs)
                }).sort_values('abs_coefficient', ascending=False)

                return importance_df

        return pd.DataFrame()  # return empty if no exog variables

    def cross_validate_model(self, df: pd.DataFrame, n_splits: int = 5,
                           min_train_size: int = 30) -> Dict[str, Any]:
        
        # prepare data
        target, exog = self.prepare_data(df)

        if len(target) < min_train_size + n_splits:
            raise ValueError(f"{self.ticker}: Insufficient data for {n_splits} CV splits")

        # create cv splits
        cv_splits = self.time_series_cv_split(target, exog, n_splits, min_train_size)

        # find optimal order using cv (if not already done)
        if self.best_order is None:
            self.best_order = self._find_optimal_order_cv(target, exog, max_d_needed=2)

        cv_results = {
            'ticker': self.ticker,
            'optimal_order': self.best_order,
            'n_splits': len(cv_splits),
            'fold_results': []
        }

        all_rmse = []
        all_mae = []
        all_mape = []
        all_direction_acc = []

        # evaluate model on each fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # split data for this fold
                    train_target = target.iloc[train_idx].values
                    train_exog = exog.iloc[train_idx].values
                    test_target = target.iloc[test_idx].values
                    test_exog = exog.iloc[test_idx].values

                    # fit model on training fold
                    model = ARIMA(train_target, exog=train_exog, order=self.best_order)
                    fitted = model.fit()

                    # predict on test fold
                    pred = fitted.forecast(steps=len(test_target), exog=test_exog)

                    # calculate metrics for this fold
                    fold_rmse = np.sqrt(mean_squared_error(test_target, pred))
                    fold_mae = mean_absolute_error(test_target, pred)
                    fold_mape = np.mean(np.abs((test_target - pred) / test_target)) * 100

                    # directional accuracy
                    direction_correct = np.sign(test_target) == np.sign(pred)
                    fold_direction_acc = np.mean(direction_correct) * 100

                    fold_result = {
                        'fold': fold_idx + 1,
                        'train_size': len(train_target),
                        'test_size': len(test_target),
                        'rmse': fold_rmse,
                        'mae': fold_mae,
                        'mape': fold_mape,
                        'directional_accuracy': fold_direction_acc,
                        'aic': fitted.aic
                    }

                    cv_results['fold_results'].append(fold_result)

                    all_rmse.append(fold_rmse)
                    all_mae.append(fold_mae)
                    all_mape.append(fold_mape)
                    all_direction_acc.append(fold_direction_acc)

                    logger.debug(f"{self.ticker} Fold {fold_idx + 1}: RMSE={fold_rmse:.4f}, "
                               f"MAE={fold_mae:.4f}, Dir_Acc={fold_direction_acc:.1f}%")

            except Exception as e:
                logger.warning(f"{self.ticker} Fold {fold_idx + 1} failed: {e}")
                # skip this fold but continue with others

        # calculate aggregate cv metrics
        if all_rmse:
            cv_results['cv_metrics'] = {
                'mean_rmse': np.mean(all_rmse),
                'std_rmse': np.std(all_rmse),
                'mean_mae': np.mean(all_mae),
                'std_mae': np.std(all_mae),
                'mean_mape': np.mean(all_mape),
                'std_mape': np.std(all_mape),
                'mean_directional_accuracy': np.mean(all_direction_acc),
                'std_directional_accuracy': np.std(all_direction_acc)
            }

            logger.info(f"{self.ticker}: CV completed - Mean RMSE: {np.mean(all_rmse):.4f} "
                       f"(±{np.std(all_rmse):.4f}), Mean Dir Acc: {np.mean(all_direction_acc):.1f}%")
        else:
            cv_results['cv_metrics'] = {'error': 'All folds failed'}
            logger.error(f"{self.ticker}: All CV folds failed")

        return cv_results