

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_xgboost import XGBoostTrainingPipeline
from walk_forward_test import XGBoostWalkForwardTester
from model_selection import ModelSelector, BALANCED_PRESETS_BY_HORIZON
from xgboost_model import StockXGBoost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    

    def __init__(self, data_file: str = 'xgboost/features_engineered.csv',
                 models_dir: str = 'xgboost/xgboostmodels_tuned',
                 results_dir: str = 'xgboost/tuning_results'):
        
        self.data_file = data_file
        self.models_dir = models_dir
        self.results_dir = results_dir

        # create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def tune_single_horizon(self, horizon: int,
                           preset_names: Optional[List[str]] = None,
                           test_weeks: int = 52,
                           use_cv: bool = True) -> Dict:
        
        if preset_names is None:
            preset_names = ['ultra_conservative', 'conservative', 'balanced',
                          'moderate', 'aggressive']

        logger.info(f"\n{'='*80}")
        logger.info(f"HYPERPARAMETER TUNING: {horizon}-week model")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(preset_names)} configurations: {', '.join(preset_names)}")

        # get presets for this horizon
        if horizon not in BALANCED_PRESETS_BY_HORIZON:
            raise ValueError(f"No balanced presets defined for horizon={horizon}")

        horizon_presets = BALANCED_PRESETS_BY_HORIZON[horizon]

        # validate all preset names exist
        for name in preset_names:
            if name not in horizon_presets:
                raise ValueError(f"Preset '{name}' not found for horizon={horizon}")

        # train each configuration
        all_results = []

        for preset_name in preset_names:
            logger.info(f"\n{'-'*80}")
            logger.info(f"Training {horizon}w model with '{preset_name}' preset")
            logger.info(f"{'-'*80}")

            config = horizon_presets[preset_name]

            try:
                # create training pipeline
                pipeline = XGBoostTrainingPipeline(
                    data_file=self.data_file,
                    models_dir=self.models_dir,
                    results_dir=self.results_dir,
                    horizons=[horizon],
                    cross_sectional=True
                )

                # monkey-patch get_best_hyperparameters to use this config
                def get_custom_hyperparameters(h: int) -> Dict:
                    base_params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'random_state': 42,
                        'n_jobs': -1,
                        'tree_method': 'hist'
                    }
                    return {**base_params, **config}

                original_method = pipeline.get_best_hyperparameters
                pipeline.get_best_hyperparameters = get_custom_hyperparameters

                # load data
                df = pipeline.load_data()
                feature_cols = pipeline.filter_features(df)

                # train model
                train_result = pipeline.train_single_horizon_model(
                    df, horizon, feature_cols,
                    use_cv=use_cv,
                    use_sample_weights=True
                )

                # restore original method
                pipeline.get_best_hyperparameters = original_method

                logger.info(f"✅ Training completed")

                # run walk-forward test
                logger.info(f"Running walk-forward validation...")

                tester = XGBoostWalkForwardTester(
                    models_dir=self.models_dir,
                    data_file=self.data_file,
                    recompute_technical=False  # use pre-computed for speed
                )

                wf_result = tester.run_test(
                    horizon=horizon,
                    test_weeks=test_weeks,
                    num_positions=5,
                    allow_short=False
                )

                logger.info(f"✅ Walk-forward test completed")

                # combine results
                combined_result = {
                    'config_name': preset_name,
                    'horizon': horizon,
                    'hyperparameters': config,
                    'training': train_result,
                    'validation': wf_result,
                }

                all_results.append(combined_result)

                # print summary
                logger.info(f"\n📊 Results Summary for '{preset_name}':")
                if 'test_rmse' in wf_result:
                    logger.info(f"  RMSE: {wf_result['test_rmse']:.4f}")
                if 'direction_accuracy' in wf_result:
                    logger.info(f"  Dir Acc: {wf_result['direction_accuracy']:.1f}%")
                if 'pct_positive' in wf_result:
                    logger.info(f"  Sign balance: {wf_result['pct_positive']:.1f}% positive")
                if 'std_ratio' in wf_result:
                    logger.info(f"  Std ratio: {wf_result['std_ratio']:.3f}")
                if 'has_sign_bias' in wf_result:
                    status = "⚠️ YES" if wf_result['has_sign_bias'] else "✅ NO"
                    logger.info(f"  Sign bias: {status}")
                if 'validation_passed' in wf_result:
                    status = "✅ PASSED" if wf_result['validation_passed'] else "❌ FAILED"
                    logger.info(f"  Validation: {status}")

            except Exception as e:
                logger.error(f"❌ Failed to train '{preset_name}': {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        if not all_results:
            raise ValueError("All configurations failed - no models to compare")

        # select best configuration using composite scoring
        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL SELECTION")
        logger.info(f"{'='*80}")

        # use custom weights prioritizing distribution quality
        weights = {
            'accuracy': 0.30,      # rmse + directional accuracy
            'distribution': 0.50,  # sign balance + std_ratio (prioritized)
            'diversity': 0.20      # prediction diversity
        }

        selection = ModelSelector.select_best_hyperparameters(
            all_results,
            weights=weights
        )

        # print comparison
        comparison_df = selection['comparison_table']
        logger.info(f"\n📊 Configuration Comparison (sorted by composite score):")
        print(comparison_df[[
            'score', 'rmse', 'dir_acc', 'pct_positive',
            'pred_diversity', 'has_sign_bias', 'validation_passed'
        ]].to_string(index=False))

        # print best config details
        best_idx = 0  # already sorted by score
        best_row = comparison_df.iloc[0]
        best_config_name = all_results[best_idx]['config_name']

        logger.info(f"\n🏆 BEST CONFIGURATION: {best_config_name}")
        logger.info(f"  Composite Score: {best_row['score']:.1f}/100")
        logger.info(f"  RMSE: {best_row['rmse']:.4f}")
        logger.info(f"  Dir Acc: {best_row['dir_acc']:.1f}%")
        logger.info(f"  Sign Balance: {best_row['pct_positive']:.1f}% positive")
        logger.info(f"  Std Ratio: {all_results[best_idx]['validation'].get('std_ratio', 'N/A'):.3f}")
        logger.info(f"  Sign Bias: {'⚠️ YES' if best_row['has_sign_bias'] else '✅ NO'}")
        logger.info(f"  Validation: {'✅ PASSED' if best_row['validation_passed'] else '❌ FAILED'}")

        # save results
        self._save_tuning_results(horizon, all_results, selection)

        return {
            'horizon': horizon,
            'all_results': all_results,
            'selection': selection,
            'comparison_df': comparison_df,
        }

    def _save_tuning_results(self, horizon: int, all_results: List[Dict],
                            selection: Dict):
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # get best config name from comparison table
        comparison_df = selection['comparison_table']
        best_idx = 0  # already sorted by score
        best_config_name = all_results[best_idx]['config_name']

        # save detailed results
        results_file = os.path.join(
            self.results_dir,
            f'tuning_results_{horizon}w_{timestamp}.json'
        )

        save_data = {
            'horizon': horizon,
            'timestamp': timestamp,
            'all_results': [
                {
                    'config_name': r['config_name'],
                    'hyperparameters': r['hyperparameters'],
                    'validation_metrics': {
                        'rmse': float(r['validation'].get('test_rmse')) if r['validation'].get('test_rmse') is not None else None,
                        'direction_accuracy': float(r['validation'].get('direction_accuracy')) if r['validation'].get('direction_accuracy') is not None else None,
                        'pct_positive': float(r['validation'].get('pct_positive')) if r['validation'].get('pct_positive') is not None else None,
                        'std_ratio': float(r['validation'].get('std_ratio')) if r['validation'].get('std_ratio') is not None else None,
                        'has_sign_bias': bool(r['validation'].get('has_sign_bias')) if r['validation'].get('has_sign_bias') is not None else False,
                        'validation_passed': bool(r['validation'].get('validation_passed')) if r['validation'].get('validation_passed') is not None else True,
                    }
                }
                for r in all_results
            ],
            'best_config': best_config_name,
            'best_score': float(selection['best_score']),
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"\n💾 Results saved to {results_file}")

        # save comparison csv
        csv_file = os.path.join(
            self.results_dir,
            f'tuning_comparison_{horizon}w_{timestamp}.csv'
        )
        selection['comparison_table'].to_csv(csv_file, index=False)
        logger.info(f"💾 Comparison saved to {csv_file}")

    def save_best_hyperparameters(self, tuning_results: Dict[int, Dict],
                                 output_file: str = 'xgboost/tuned_hyperparameters.json'):
        
        best_params = {}

        for horizon, results in tuning_results.items():
            # extract best config name from all_results (sorted by score)
            all_results = results['all_results']
            best_idx = 0  # first entry is best
            best_config_name = all_results[best_idx]['config_name']
            best_config = BALANCED_PRESETS_BY_HORIZON[horizon][best_config_name]

            best_params[str(horizon)] = {
                'preset_name': best_config_name,
                'hyperparameters': best_config,
                'composite_score': float(results['selection']['best_score']),
            }

        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"\n💾 Best hyperparameters saved to {output_file}")


def main():
    
    parser = argparse.ArgumentParser(description='Tune XGBoost hyperparameters')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to tune (default: 1)')
    parser.add_argument('--presets', type=str, nargs='+',
                       default=['ultra_conservative', 'conservative', 'balanced', 'moderate', 'aggressive'],
                       help='Preset names to test')
    parser.add_argument('--test-weeks', type=int, default=52,
                       help='Weeks for walk-forward test')
    parser.add_argument('--no-cv', action='store_true',
                       help='Skip cross-validation (faster)')
    parser.add_argument('--save-best', action='store_true',
                       help='Save best hyperparameters to JSON')

    args = parser.parse_args()

    tuner = HyperparameterTuner()

    all_tuning_results = {}

    for horizon in args.horizons:
        logger.info(f"\n{'#'*80}")
        logger.info(f"STARTING TUNING FOR {horizon}-WEEK MODEL")
        logger.info(f"{'#'*80}")

        try:
            results = tuner.tune_single_horizon(
                horizon=horizon,
                preset_names=args.presets,
                test_weeks=args.test_weeks,
                use_cv=not args.no_cv
            )

            all_tuning_results[horizon] = results

        except Exception as e:
            logger.error(f"❌ Tuning failed for {horizon}-week model: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # save best hyperparameters if requested
    if args.save_best and all_tuning_results:
        tuner.save_best_hyperparameters(all_tuning_results)

    logger.info(f"\n{'#'*80}")
    logger.info("TUNING COMPLETE")
    logger.info(f"{'#'*80}")

    # print final summary
    logger.info(f"\n📊 FINAL SUMMARY:")
    for horizon in args.horizons:
        if horizon in all_tuning_results:
            results = all_tuning_results[horizon]
            # extract best config name from all_results (sorted by score)
            all_results = results['all_results']
            best_name = all_results[0]['config_name']  # first entry is best
            best_score = results['selection']['best_score']
            logger.info(f"  {horizon}-week: {best_name} (score={best_score:.1f})")


if __name__ == "__main__":
    main()
