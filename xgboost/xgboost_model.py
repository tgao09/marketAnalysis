

import importlib.util
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _load_external_xgboost() -> Any:
    """Load the installed xgboost package, avoiding this project's local package."""
    current_pkg_path = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)

    # Remove entries that would resolve to this repo (so we don't shadow the real package).
    sanitized_path = []
    for entry in original_sys_path:
        resolved = Path(entry or ".").resolve()
        skip = False
        # Skip if this path is the package itself or an ancestor (e.g., repo root).
        try:
            current_pkg_path.relative_to(resolved)
            skip = True
        except ValueError:
            pass

        if skip or resolved == current_pkg_path:
            continue

        sanitized_path.append(entry)

    try:
        sys.path = sanitized_path
        spec = importlib.util.find_spec("xgboost")
        if spec is None or spec.origin is None:
            raise ImportError

        module = importlib.util.module_from_spec(spec)
        # Ensure intra-package imports inside xgboost resolve to this loaded module.
        sys.modules["xgboost"] = module
        if spec.loader is None:
            raise ImportError
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module
    except ImportError as exc:
        raise ImportError(
            "Installed xgboost package not found. Please install it via 'pip install xgboost'."
        ) from exc
    finally:
        sys.path = original_sys_path


xgb = _load_external_xgboost()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PurgedWalkForwardCV:
    

    def __init__(self, n_splits: int = 5, test_size: int = 13,
                 purge_weeks: int = 2, embargo_weeks: int = 2,
                 min_train_size: int = 52):
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_weeks = purge_weeks
        self.embargo_weeks = embargo_weeks
        self.min_train_size = min_train_size

    def split(self, df: pd.DataFrame, date_col: str = 'Date') -> List[Tuple[np.ndarray, np.ndarray]]:
        
        # sort by date
        df = df.sort_values(date_col).reset_index(drop=True)
        n_samples = len(df)

        if n_samples < self.min_train_size + self.test_size:
            raise ValueError(f"Insufficient data: {n_samples} samples")

        splits = []

        # calculate split points
        available_for_testing = n_samples - self.min_train_size
        step_size = max(self.test_size, available_for_testing // self.n_splits)

        for i in range(self.n_splits):
            # expanding window: train grows with each split
            train_end = self.min_train_size + i * step_size

            # apply purge: remove data right before test
            train_end_purged = train_end - self.purge_weeks

            # test period
            test_start = train_end
            test_end = min(test_start + self.test_size, n_samples)

            # apply embargo: remove data right after test (for forward-looking labels)
            test_end_embargoed = min(test_end + self.embargo_weeks, n_samples)

            # skip if no test data
            if test_start >= n_samples:
                break

            # create indices
            train_idx = np.arange(0, train_end_purged)
            test_idx = np.arange(test_start, test_end)

            # ensure we have data
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        logger.info(f"Created {len(splits)} purged walk-forward splits")
        logger.info(f"  Purge: {self.purge_weeks} weeks, Embargo: {self.embargo_weeks} weeks")

        return splits


class StockXGBoost:
    

    def __init__(self, ticker: Optional[str] = None, horizon: int = 1,
                 xgb_params: Optional[Dict[str, Any]] = None,
                 feature_columns: Optional[List[str]] = None):
        
        self.ticker = ticker
        self.horizon = horizon
        self.feature_columns = feature_columns
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None
        self.best_iteration = None

        # default xgboost parameters
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
        self.xgb_params = xgb_params

    def prepare_data(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        
        if target_col is None:
            target_col = f'forward_return_{self.horizon}w'

        # filter to specific ticker if specified
        if self.ticker is not None:
            df = df[df['ticker'] == self.ticker].copy()

        # auto-detect feature columns if not provided
        if self.feature_columns is None:
            # exclude non-feature columns
            exclude_cols = ['ticker', 'Date', 'date'] + [c for c in df.columns if 'forward_' in c]
            self.feature_columns = [c for c in df.columns if c not in exclude_cols]
            logger.info(f"Auto-detected {len(self.feature_columns)} features")

        # extract features
        X = df[self.feature_columns].copy()

        # handle infinity values (replace with nan, then fill with median)
        X = X.replace([np.inf, -np.inf], np.nan)

        # handle missing values
        X = X.fillna(X.median())

        # final check: if any column is all nan, fill with 0
        X = X.fillna(0)

        # extract target
        y = df[target_col].copy() if target_col in df.columns else None

        return X, y

    def fit(self, df: pd.DataFrame, target_col: str = None,
            eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
            sample_weights: Optional[np.ndarray] = None,
            early_stopping_rounds: int = 50,
            verbose: bool = False) -> Dict[str, Any]:
        
        logger.info(f"Training XGBoost model (horizon={self.horizon}w, ticker={self.ticker or 'ALL'})")

        # prepare data
        X_train, y_train = self.prepare_data(df, target_col)

        # remove rows with missing targets
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]

        if sample_weights is not None:
            sample_weights = sample_weights[valid_idx]

        if len(X_train) < 10:
            raise ValueError(f"Insufficient training data: {len(X_train)} samples")

        # prepare eval set if provided
        if eval_set is not None:
            eval_set_prepared = []
            for X_val, y_val in eval_set:
                valid_idx_val = ~y_val.isna()
                eval_set_prepared.append((X_val[valid_idx_val], y_val[valid_idx_val]))
        else:
            eval_set_prepared = None

        # train model
        # for xgboost 1.x, don't use early_stopping_rounds or callbacks
        # train with fixed n_estimators
        self.model = xgb.XGBRegressor(**self.xgb_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # simple fit without early stopping for xgboost 1.0
            self.model.fit(X_train, y_train, sample_weight=sample_weights, verbose=verbose)

        self.is_fitted = True
        self.best_iteration = getattr(self.model, 'best_iteration', None)

        # get feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Model trained: {len(X_train)} samples, {len(self.feature_columns)} features")
        if self.best_iteration:
            logger.info(f"Best iteration: {self.best_iteration}")

        return {
            'n_samples': len(X_train),
            'n_features': len(self.feature_columns),
            'best_iteration': self.best_iteration
        }

    def predict(self, df: pd.DataFrame, return_std: bool = False) -> np.ndarray:
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X, _ = self.prepare_data(df, target_col=None)

        predictions = self.model.predict(X)

        if return_std:
            logger.warning("Prediction uncertainty not implemented for XGBoost, returning zeros")
            return predictions, np.zeros_like(predictions)

        return predictions

    def cross_validate(self, df: pd.DataFrame, cv_splitter: PurgedWalkForwardCV,
                      target_col: str = None, sample_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        
        logger.info(f"Running purged walk-forward CV with {cv_splitter.n_splits} splits")

        # prepare data
        X, y = self.prepare_data(df, target_col)

        # remove rows with missing targets
        valid_idx = ~y.isna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        df_valid = df[valid_idx].reset_index(drop=True)

        if sample_weights is not None:
            sample_weights = sample_weights[valid_idx]

        # get cv splits
        splits = cv_splitter.split(df_valid)

        cv_results = {
            'fold_metrics': [],
            'predictions': [],
            'feature_importance': []
        }

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Fold {fold_idx + 1}/{len(splits)}: train={len(train_idx)}, test={len(test_idx)}")

            # split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_weights = sample_weights[train_idx] if sample_weights is not None else None

            # train fold model
            fold_model = xgb.XGBRegressor(**self.xgb_params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # simple fit without early stopping for xgboost 1.0
                fold_model.fit(X_train, y_train, sample_weight=fold_weights, verbose=False)

            # predict
            y_pred = fold_model.predict(X_test)

            # calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # directional accuracy
            direction_correct = (np.sign(y_test) == np.sign(y_pred)).mean()

            fold_metrics = {
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': direction_correct * 100
            }

            cv_results['fold_metrics'].append(fold_metrics)
            cv_results['predictions'].extend(list(zip(test_idx, y_test, y_pred)))

            # feature importance for this fold
            fold_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': fold_model.feature_importances_,
                'fold': fold_idx + 1
            })
            cv_results['feature_importance'].append(fold_importance)

            logger.info(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Dir Acc: {direction_correct*100:.1f}%")

        # aggregate metrics
        metrics_df = pd.DataFrame(cv_results['fold_metrics'])
        cv_results['mean_metrics'] = {
            'mean_rmse': metrics_df['rmse'].mean(),
            'std_rmse': metrics_df['rmse'].std(),
            'mean_mae': metrics_df['mae'].mean(),
            'std_mae': metrics_df['mae'].std(),
            'mean_r2': metrics_df['r2'].mean(),
            'std_r2': metrics_df['r2'].std(),
            'mean_directional_accuracy': metrics_df['directional_accuracy'].mean(),
            'std_directional_accuracy': metrics_df['directional_accuracy'].std()
        }

        # aggregate feature importance across folds
        all_importance = pd.concat(cv_results['feature_importance'], ignore_index=True)
        cv_results['avg_feature_importance'] = all_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)

        logger.info(f"CV Results: RMSE={cv_results['mean_metrics']['mean_rmse']:.4f} ± {cv_results['mean_metrics']['std_rmse']:.4f}")
        logger.info(f"            Dir Acc={cv_results['mean_metrics']['mean_directional_accuracy']:.1f}% ± {cv_results['mean_metrics']['std_directional_accuracy']:.1f}%")

        return cv_results

    def save_model(self, filepath: str) -> None:
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'ticker': self.ticker,
            'horizon': self.horizon,
            'model': self.model,
            'feature_columns': self.feature_columns,
            'xgb_params': self.xgb_params,
            'feature_importance': self.feature_importance_,
            'best_iteration': self.best_iteration
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'StockXGBoost':
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        instance = cls(
            ticker=model_data['ticker'],
            horizon=model_data['horizon'],
            xgb_params=model_data['xgb_params'],
            feature_columns=model_data['feature_columns']
        )
        instance.model = model_data['model']
        instance.feature_importance_ = model_data['feature_importance']
        instance.best_iteration = model_data['best_iteration']
        instance.is_fitted = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted first")

        return self.feature_importance_.head(n)


def calculate_sample_weights(df: pd.DataFrame, date_col: str = 'Date',
                            decay_half_life: int = 52) -> np.ndarray:
    
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col])

    # calculate weeks from most recent data
    max_date = dates.max()
    weeks_ago = (max_date - dates).dt.days / 7

    # exponential decay: weight = 2^(-weeks / half_life)
    weights = 2 ** (-weeks_ago / decay_half_life)

    # normalize to sum to number of samples
    weights = weights * len(weights) / weights.sum()

    return weights.values


if __name__ == "__main__":
    # example usage
    logger.info("StockXGBoost model module loaded")
    logger.info("Use this module to train XGBoost models with purged walk-forward CV")
