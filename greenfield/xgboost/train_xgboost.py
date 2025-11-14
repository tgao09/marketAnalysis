

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import warnings
from pathlib import Path

# add xgboost directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_model import StockXGBoost, PurgedWalkForwardCV, calculate_sample_weights
from model_diagnostics import ModelDiagnostics, detect_feature_leakage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class XGBoostTrainingPipeline:
    

    def __init__(self, data_file: str = 'xgboost/features_engineered.csv',
                 models_dir: str = 'xgboost/xgboostmodels',
                 results_dir: str = 'xgboost/xgboostresults',
                 horizons: List[int] = [1, 2, 4, 8],
                 cross_sectional: bool = True):
        
        self.data_file = data_file
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.horizons = horizons
        self.cross_sectional = cross_sectional

        # create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        
        logger.info(f"Loading data from {self.data_file}")

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        df = pd.read_csv(self.data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows for {df['ticker'].nunique()} stocks")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    def filter_features(self, df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
        
        if exclude_patterns is None:
            exclude_patterns = []

        # always exclude these
        base_exclude = ['ticker', 'Date', 'date', 'forward_return', 'forward_direction']

        # find columns to exclude
        exclude_cols = set()
        for col in df.columns:
            # check base patterns
            if any(pattern in col for pattern in base_exclude):
                exclude_cols.add(col)
            # check custom patterns
            if any(pattern in col for pattern in exclude_patterns):
                exclude_cols.add(col)

        # get feature columns
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        logger.info(f"Identified {len(feature_cols)} feature columns")

        return feature_cols

    def get_best_hyperparameters(self, horizon: int) -> Dict[str, Any]:
        
        # base parameters
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        # horizon-specific tuning - stronger regularization
        if horizon == 1:
            # 1-week: prevent short-term overfitting
            params = {
                **base_params,
                'max_depth': 3,              # reduced from 6
                'learning_rate': 0.01,       # reduced from 0.05
                'n_estimators': 200,         # reduced from 800
                'subsample': 0.7,            # reduced from 0.8
                'colsample_bytree': 0.6,     # reduced from 0.8
                'colsample_bynode': 0.6,     # new: additional feature sampling per split
                'min_child_weight': 5,       # increased from 2
                'gamma': 0.3,                # increased from 0.05
                'reg_alpha': 2.0,            # increased from 0.3 (l1)
                'reg_lambda': 5.0,           # increased from 1.0 (l2)
                'max_delta_step': 1,         # new: limit extreme predictions
            }
        elif horizon == 2:
            # 2-week: moderate regularization
            params = {
                **base_params,
                'max_depth': 3,              # reduced from 5
                'learning_rate': 0.01,       # reduced from 0.05
                'n_estimators': 180,         # reduced from 700
                'subsample': 0.65,           # reduced from 0.8
                'colsample_bytree': 0.6,     # reduced from 0.8
                'colsample_bynode': 0.6,
                'min_child_weight': 7,       # increased from 3
                'gamma': 0.4,                # increased from 0.1
                'reg_alpha': 2.5,            # increased from 0.5
                'reg_lambda': 6.0,           # increased from 1.5
                'max_delta_step': 1,
            }
        elif horizon == 4:
            # 4-week: strong regularization
            params = {
                **base_params,
                'max_depth': 2,              # reduced from 4
                'learning_rate': 0.01,       # reduced from 0.04
                'n_estimators': 150,         # reduced from 600
                'subsample': 0.6,            # reduced from 0.75
                'colsample_bytree': 0.5,     # reduced from 0.75
                'colsample_bynode': 0.5,
                'min_child_weight': 10,      # increased from 5
                'gamma': 0.5,                # increased from 0.15
                'reg_alpha': 3.0,            # increased from 0.8
                'reg_lambda': 8.0,           # increased from 2.0
                'max_delta_step': 0.5,       # smaller for longer horizon
            }
        else:  # 8-week or longer
            # long-term: very heavy regularization
            params = {
                **base_params,
                'max_depth': 2,              # reduced from 3
                'learning_rate': 0.008,      # reduced from 0.03
                'n_estimators': 120,         # reduced from 500
                'subsample': 0.55,           # reduced from 0.7
                'colsample_bytree': 0.5,     # reduced from 0.7
                'colsample_bynode': 0.4,
                'min_child_weight': 15,      # increased from 8
                'gamma': 0.6,                # increased from 0.2
                'reg_alpha': 4.0,            # increased from 1.0
                'reg_lambda': 10.0,          # increased from 3.0
                'max_delta_step': 0.3,       # very conservative
            }

        return params

    def train_single_horizon_model(self, df: pd.DataFrame, horizon: int,
                                   feature_cols: List[str],
                                   use_cv: bool = True,
                                   use_sample_weights: bool = True) -> Dict[str, Any]:
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {horizon}-week model")
        logger.info(f"{'='*60}")

        target_col = f'forward_return_{horizon}w'

        # check if target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")

        # filter to rows with valid targets
        df_train = df[df[target_col].notna()].copy()
        logger.info(f"Training samples: {len(df_train)}")

        # calculate sample weights (recent data weighted higher)
        sample_weights = None
        if use_sample_weights:
            sample_weights = calculate_sample_weights(df_train, 'Date', decay_half_life=52)
            logger.info("Using time-decayed sample weights")

        # get hyperparameters
        xgb_params = self.get_best_hyperparameters(horizon)

        # initialize model
        ticker = None if self.cross_sectional else df_train['ticker'].iloc[0]
        model = StockXGBoost(
            ticker=ticker,
            horizon=horizon,
            xgb_params=xgb_params,
            feature_columns=feature_cols
        )

        results = {
            'horizon': horizon,
            'ticker': ticker or 'ALL',
            'n_samples': len(df_train),
            'n_features': len(feature_cols)
        }

        # cross-validation
        if use_cv:
            cv_splitter = PurgedWalkForwardCV(
                n_splits=5,
                test_size=13,  # ~3 months
                purge_weeks=2,
                embargo_weeks=2,
                min_train_size=52  # ~1 year
            )

            cv_results = model.cross_validate(
                df_train,
                cv_splitter=cv_splitter,
                target_col=target_col,
                sample_weights=sample_weights
            )

            results['cv_metrics'] = cv_results['mean_metrics']
            results['cv_fold_results'] = cv_results['fold_metrics']

            # save feature importance from cv
            top_features = cv_results['avg_feature_importance'].head(20)
            results['top_features'] = top_features.to_dict()

            logger.info(f"CV RMSE: {cv_results['mean_metrics']['mean_rmse']:.4f} ± {cv_results['mean_metrics']['std_rmse']:.4f}")
            logger.info(f"CV Dir Acc: {cv_results['mean_metrics']['mean_directional_accuracy']:.1f}% ± {cv_results['mean_metrics']['std_directional_accuracy']:.1f}%")

        # train final model on all data
        logger.info("Training final model on all data...")
        train_results = model.fit(
            df_train,
            target_col=target_col,
            sample_weights=sample_weights,
            early_stopping_rounds=50,
            verbose=False
        )

        results.update(train_results)

        # get feature importance from final model
        top_features_final = model.get_top_features(20)
        results['final_top_features'] = top_features_final.to_dict('records')

        logger.info(f"Model trained successfully")
        logger.info(f"Top 5 features: {', '.join(top_features_final.head(5)['feature'].tolist())}")

        # check for feature leakage
        suspicious_features = detect_feature_leakage(
            model.feature_importance_,
            feature_cols,
            threshold_importance=0.3
        )
        if suspicious_features:
            logger.warning(f"\n⚠️  POTENTIAL FEATURE LEAKAGE DETECTED:")
            for item in suspicious_features:
                logger.warning(f"  • {item['feature']}: {item['importance']*100:.1f}% - {item['reason']}")
            results['suspicious_features'] = suspicious_features
        else:
            logger.info("✅ No obvious feature leakage detected")
            results['suspicious_features'] = []

        # save model
        model_filename = f"xgboost_{horizon}w_{'all' if self.cross_sectional else ticker}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        model.save_model(model_path)

        results['model_path'] = model_path

        return results

    def train_all_horizons(self, use_cv: bool = True,
                          use_sample_weights: bool = True) -> pd.DataFrame:
        
        logger.info(f"Starting training pipeline for {len(self.horizons)} horizons")
        logger.info(f"Mode: {'Cross-sectional (all stocks)' if self.cross_sectional else 'Per-stock'}")

        # load data
        df = self.load_data()

        # identify features
        feature_cols = self.filter_features(df)

        # train models for each horizon
        all_results = []

        for horizon in self.horizons:
            try:
                results = self.train_single_horizon_model(
                    df, horizon, feature_cols,
                    use_cv=use_cv,
                    use_sample_weights=use_sample_weights
                )
                all_results.append(results)

            except Exception as e:
                logger.error(f"Failed to train {horizon}-week model: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # save results summary
        if all_results:
            # flatten results for dataframe
            results_flat = []
            for res in all_results:
                flat_res = {
                    'horizon': res['horizon'],
                    'ticker': res['ticker'],
                    'n_samples': res['n_samples'],
                    'n_features': res['n_features'],
                    'model_path': res['model_path']
                }

                # add cv metrics if available
                if 'cv_metrics' in res:
                    for key, value in res['cv_metrics'].items():
                        flat_res[f'cv_{key}'] = value

                results_flat.append(flat_res)

            results_df = pd.DataFrame(results_flat)

            # save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(self.results_dir, f'training_results_{timestamp}.csv')
            results_df.to_csv(results_file, index=False)
            logger.info(f"Training results saved to {results_file}")

            # print summary
            print(f"\n{'='*80}")
            print("TRAINING SUMMARY")
            print(f"{'='*80}")
            print(f"Successfully trained: {len(results_flat)}/{len(self.horizons)} models")

            if 'cv_mean_rmse' in results_df.columns:
                print(f"\nCross-Validation Performance:")
                for _, row in results_df.iterrows():
                    print(f"  {row['horizon']}-week: RMSE={row['cv_mean_rmse']:.4f} ± {row['cv_std_rmse']:.4f}, "
                          f"Dir Acc={row['cv_mean_directional_accuracy']:.1f}% ± {row['cv_std_directional_accuracy']:.1f}%")

            print(f"\nModels saved to: {self.models_dir}")
            print(f"Results saved to: {results_file}")
            print(f"{'='*80}\n")

            return results_df

        else:
            logger.error("No models trained successfully")
            return pd.DataFrame()


def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='Train XGBoost models for stock prediction')
    parser.add_argument('--data-file', type=str, default='xgboost/features_engineered.csv',
                       help='Path to engineered features CSV')
    parser.add_argument('--models-dir', type=str, default='xgboost/xgboostmodels',
                       help='Directory to save trained models')
    parser.add_argument('--results-dir', type=str, default='xgboost/xgboostresults',
                       help='Directory to save results')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Prediction horizons in weeks (default: 1 2 4 8)')
    parser.add_argument('--per-stock', action='store_true',
                       help='Train per-stock models instead of cross-sectional')
    parser.add_argument('--no-cv', action='store_true',
                       help='Skip cross-validation (faster but less validation)')
    parser.add_argument('--no-sample-weights', action='store_true',
                       help='Disable time-decayed sample weights')

    args = parser.parse_args()

    # initialize pipeline
    pipeline = XGBoostTrainingPipeline(
        data_file=args.data_file,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        horizons=args.horizons,
        cross_sectional=not args.per_stock
    )

    # train models
    try:
        results_df = pipeline.train_all_horizons(
            use_cv=not args.no_cv,
            use_sample_weights=not args.no_sample_weights
        )

        if not results_df.empty:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
