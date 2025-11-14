

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelSelector:
    

    @staticmethod
    def score_std_ratio(std_ratio: float) -> float:
        
        if np.isnan(std_ratio) or std_ratio <= 0:
            return 0.0

        if std_ratio < 0.5:
            # severe underfitting penalty
            return std_ratio / 0.5 * 0.4  # max 0.4 score
        elif 0.7 <= std_ratio <= 1.2:
            # ideal zone - full score
            return 1.0
        elif 0.5 <= std_ratio < 0.7:
            # mild underfitting - linear penalty
            return 0.4 + (std_ratio - 0.5) / 0.2 * 0.6
        elif 1.2 < std_ratio <= 2.0:
            # mild overfitting - linear penalty
            return 1.0 - (std_ratio - 1.2) / 0.8 * 0.4
        else:
            # severe overfitting (>2.0)
            return 0.3 * np.exp(-(std_ratio - 2.0) / 2.0)

    @staticmethod
    def score_model(val_results: Dict, weights: Optional[Dict] = None) -> float:
        
        if weights is None:
            weights = {
                'accuracy': 0.4,
                'distribution': 0.3,
                'diversity': 0.3
            }

        # component 1: predictive accuracy (0-1 scale)
        rmse = val_results.get('rmse', val_results.get('test_rmse', 1.0))
        dir_acc = val_results.get('directional_accuracy', val_results.get('direction_accuracy', 50.0))

        rmse_score = 1 / (1 + rmse)  # 0-1 scale (lower rmse = higher score)
        dir_acc_score = dir_acc / 100  # convert percentage to 0-1

        accuracy_score = 0.6 * rmse_score + 0.4 * dir_acc_score

        # component 2: distribution realism (0-1 scale)
        pct_pos = val_results.get('pct_positive', 50.0)
        pred_std = val_results.get('pred_std', 0.01)
        true_std = val_results.get('true_std', 0.01)

        # penalize if not 40-60% positive (stocks should be ~50/50)
        balance_score = 1 - abs(pct_pos - 50) / 50  # 1 at 50%, 0 at 0/100%

        # use enhanced std_ratio scoring (with underfitting bias)
        std_ratio = val_results.get('std_ratio', pred_std / true_std if true_std > 0 else 1.0)
        std_score = ModelSelector.score_std_ratio(std_ratio)

        distribution_score = 0.5 * balance_score + 0.5 * std_score

        # component 3: diversity/stability (0-1 scale)
        diversity = val_results.get('prediction_diversity', 0)
        diversity_score = min(diversity / 200, 1.0)  # 1.0 at 200+ unique predictions

        # final composite
        composite = (
            weights['accuracy'] * accuracy_score +
            weights['distribution'] * distribution_score +
            weights['diversity'] * diversity_score
        )

        return 100 * composite

    @staticmethod
    def select_best_hyperparameters(
        candidate_results: List[Dict],
        weights: Optional[Dict] = None
    ) -> Dict:
        
        scored = []

        for result in candidate_results:
            val_metrics = result.get('validation', result)
            score = ModelSelector.score_model(val_metrics, weights)

            scored_entry = {
                'score': score,
                'rmse': val_metrics.get('rmse', val_metrics.get('test_rmse', np.nan)),
                'dir_acc': val_metrics.get('directional_accuracy',
                                          val_metrics.get('direction_accuracy', np.nan)),
                'pct_positive': val_metrics.get('pct_positive', np.nan),
                'pred_diversity': val_metrics.get('prediction_diversity', np.nan),
                'has_sign_bias': val_metrics.get('has_sign_bias', False),
                'validation_passed': val_metrics.get('validation_passed', True),
            }

            # add hyperparameters if available
            if 'hyperparameters' in result:
                scored_entry['params'] = result['hyperparameters']

            scored.append(scored_entry)

        # sort by composite score
        scored = sorted(scored, key=lambda x: x['score'], reverse=True)

        comparison_df = pd.DataFrame(scored)

        return {
            'best_params': scored[0].get('params', None),
            'best_score': scored[0]['score'],
            'comparison_table': comparison_df,
            'top_candidate': scored[0]
        }

    @staticmethod
    def compare_models(results_list: List[Dict], model_names: Optional[List[str]] = None) -> pd.DataFrame:
        
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(results_list))]

        comparisons = []

        for name, results in zip(model_names, results_list):
            score = ModelSelector.score_model(results)

            comparison = {
                'model': name,
                'composite_score': score,
                'rmse': results.get('rmse', results.get('test_rmse', np.nan)),
                'directional_acc': results.get('directional_accuracy',
                                               results.get('direction_accuracy', np.nan)),
                'pct_positive': results.get('pct_positive', np.nan),
                'pct_negative': results.get('pct_negative', np.nan),
                'pred_diversity': results.get('prediction_diversity', np.nan),
                'std_ratio': results.get('std_ratio', np.nan),
                'has_sign_bias': results.get('has_sign_bias', False),
                'validation_passed': results.get('validation_passed', True),
            }

            comparisons.append(comparison)

        df = pd.DataFrame(comparisons)
        df = df.sort_values('composite_score', ascending=False)

        return df

    @staticmethod
    def print_selection_report(
        comparison_df: pd.DataFrame,
        best_model_name: str,
        title: str = "MODEL SELECTION REPORT"
    ):
        
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}")

        print(f"\n🏆 BEST MODEL: {best_model_name}")
        best_row = comparison_df[comparison_df['model'] == best_model_name].iloc[0]
        print(f"   Composite Score: {best_row['composite_score']:.1f}/100")
        print(f"   RMSE: {best_row['rmse']:.4f}")
        print(f"   Directional Accuracy: {best_row['directional_acc']:.1f}%")
        print(f"   Prediction Balance: {best_row['pct_positive']:.1f}% positive / {best_row['pct_negative']:.1f}% negative")
        print(f"   Diversity: {best_row['pred_diversity']} unique predictions")
        print(f"   Sign Bias: {'⚠️ YES' if best_row['has_sign_bias'] else '✅ NO'}")
        print(f"   Validation: {'✅ PASSED' if best_row['validation_passed'] else '❌ FAILED'}")

        print(f"\n📊 ALL MODELS RANKED:")
        print(comparison_df[['model', 'composite_score', 'rmse', 'directional_acc',
                            'pct_positive', 'pred_diversity', 'has_sign_bias']].to_string(index=False))

        print(f"\n{'='*80}\n")


def evaluate_regularization_levels(
    train_func,
    regularization_configs: List[Dict],
    config_names: List[str],
    **train_kwargs
) -> Dict:
    
    logger.info(f"Evaluating {len(regularization_configs)} regularization configurations...")

    all_results = []

    for config, name in zip(regularization_configs, config_names):
        logger.info(f"\nTesting configuration: {name}")
        logger.info(f"Parameters: {config}")

        try:
            # train with this configuration
            results = train_func(hyperparameters=config, **train_kwargs)
            results['config_name'] = name
            results['hyperparameters'] = config
            all_results.append(results)

            logger.info(f"✅ {name} completed")
            if 'test_rmse' in results:
                logger.info(f"   RMSE: {results['test_rmse']:.4f}")
            if 'direction_accuracy' in results:
                logger.info(f"   Dir Acc: {results['direction_accuracy']:.1f}%")

        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            continue

    if not all_results:
        raise ValueError("All regularization configurations failed")

    # compare and select best
    comparison_df = ModelSelector.compare_models(
        all_results,
        model_names=[r['config_name'] for r in all_results]
    )

    best_config_name = comparison_df.iloc[0]['model']
    best_result = next(r for r in all_results if r['config_name'] == best_config_name)

    # print report
    ModelSelector.print_selection_report(
        comparison_df,
        best_config_name,
        title="REGULARIZATION LEVEL SELECTION"
    )

    return {
        'best_config': best_result['hyperparameters'],
        'best_config_name': best_config_name,
        'best_score': comparison_df.iloc[0]['composite_score'],
        'comparison_df': comparison_df,
        'all_results': all_results
    }


# predefined regularization levels for experimentation
REGULARIZATION_PRESETS = {
    'conservative': {
        # very strong regularization - lowest overfitting risk
        'max_depth': 2,
        'learning_rate': 0.005,
        'n_estimators': 100,
        'subsample': 0.5,
        'colsample_bytree': 0.4,
        'colsample_bynode': 0.4,
        'min_child_weight': 20,
        'gamma': 0.8,
        'reg_alpha': 5.0,
        'reg_lambda': 12.0,
        'max_delta_step': 0.2,
    },
    'moderate': {
        # balanced regularization - recommended starting point
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 180,
        'subsample': 0.65,
        'colsample_bytree': 0.6,
        'colsample_bynode': 0.6,
        'min_child_weight': 7,
        'gamma': 0.4,
        'reg_alpha': 2.5,
        'reg_lambda': 6.0,
        'max_delta_step': 1.0,
    },
    'aggressive': {
        # lighter regularization - may overfit but higher potential
        'max_depth': 4,
        'learning_rate': 0.02,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'colsample_bynode': 0.7,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'max_delta_step': 2.0,
    }
}


# horizon-specific balanced presets (designed to fix underfitting while avoiding sign bias)
# target: pred_std/true_std ∈ [0.5, 1.2], sign_bias < 70%
BALANCED_PRESETS_BY_HORIZON = {
    1: {  # 1-week: currently 0.26 std_ratio (too conservative)
        'ultra_conservative': {  # current params (baseline)
            'max_depth': 3,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'colsample_bynode': 0.6,
            'min_child_weight': 5,
            'gamma': 0.3,
            'reg_alpha': 2.0,
            'reg_lambda': 5.0,
            'max_delta_step': 1.0,
        },
        'conservative': {  # slightly more expressive
            'max_depth': 3,
            'learning_rate': 0.015,
            'n_estimators': 220,
            'subsample': 0.72,
            'colsample_bytree': 0.65,
            'colsample_bynode': 0.65,
            'min_child_weight': 4,
            'gamma': 0.25,
            'reg_alpha': 1.5,
            'reg_lambda': 3.5,
            'max_delta_step': 1.5,
        },
        'balanced': {  # target: std_ratio ~0.8 ⭐
            'max_depth': 4,
            'learning_rate': 0.02,
            'n_estimators': 250,
            'subsample': 0.75,
            'colsample_bytree': 0.7,
            'colsample_bynode': 0.7,
            'min_child_weight': 3,
            'gamma': 0.15,
            'reg_alpha': 1.2,
            'reg_lambda': 3.0,
            'max_delta_step': 2.0,
        },
        'moderate': {  # more expressive
            'max_depth': 4,
            'learning_rate': 0.025,
            'n_estimators': 280,
            'subsample': 0.78,
            'colsample_bytree': 0.75,
            'colsample_bynode': 0.75,
            'min_child_weight': 2,
            'gamma': 0.1,
            'reg_alpha': 0.8,
            'reg_lambda': 2.0,
            'max_delta_step': 2.5,
        },
        'aggressive': {  # maximum expressiveness (may overfit)
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 320,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bynode': 0.8,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'max_delta_step': 3.0,
        },
    },
    2: {  # 2-week: currently 0.20 std_ratio (too conservative)
        'ultra_conservative': {  # current params (baseline)
            'max_depth': 3,
            'learning_rate': 0.01,
            'n_estimators': 180,
            'subsample': 0.65,
            'colsample_bytree': 0.6,
            'colsample_bynode': 0.6,
            'min_child_weight': 7,
            'gamma': 0.4,
            'reg_alpha': 2.5,
            'reg_lambda': 6.0,
            'max_delta_step': 1.0,
        },
        'conservative': {
            'max_depth': 3,
            'learning_rate': 0.014,
            'n_estimators': 200,
            'subsample': 0.68,
            'colsample_bytree': 0.65,
            'colsample_bynode': 0.65,
            'min_child_weight': 6,
            'gamma': 0.3,
            'reg_alpha': 1.8,
            'reg_lambda': 4.5,
            'max_delta_step': 1.4,
        },
        'balanced': {  # target: std_ratio ~0.7 ⭐
            'max_depth': 3,
            'learning_rate': 0.018,
            'n_estimators': 220,
            'subsample': 0.72,
            'colsample_bytree': 0.68,
            'colsample_bynode': 0.68,
            'min_child_weight': 5,
            'gamma': 0.2,
            'reg_alpha': 1.5,
            'reg_lambda': 3.5,
            'max_delta_step': 1.8,
        },
        'moderate': {
            'max_depth': 4,
            'learning_rate': 0.022,
            'n_estimators': 250,
            'subsample': 0.75,
            'colsample_bytree': 0.72,
            'colsample_bynode': 0.72,
            'min_child_weight': 4,
            'gamma': 0.15,
            'reg_alpha': 1.0,
            'reg_lambda': 2.5,
            'max_delta_step': 2.2,
        },
        'aggressive': {
            'max_depth': 4,
            'learning_rate': 0.028,
            'n_estimators': 280,
            'subsample': 0.78,
            'colsample_bytree': 0.75,
            'colsample_bynode': 0.75,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.7,
            'reg_lambda': 1.8,
            'max_delta_step': 2.6,
        },
    },
    4: {  # 4-week: currently 0.14 std_ratio + 94% positive bias
        'ultra_conservative': {  # current params (baseline)
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 150,
            'subsample': 0.6,
            'colsample_bytree': 0.5,
            'colsample_bynode': 0.5,
            'min_child_weight': 10,
            'gamma': 0.5,
            'reg_alpha': 3.0,
            'reg_lambda': 8.0,
            'max_delta_step': 0.5,
        },
        'conservative': {
            'max_depth': 3,
            'learning_rate': 0.012,
            'n_estimators': 170,
            'subsample': 0.64,
            'colsample_bytree': 0.55,
            'colsample_bynode': 0.55,
            'min_child_weight': 8,
            'gamma': 0.4,
            'reg_alpha': 2.3,
            'reg_lambda': 6.0,
            'max_delta_step': 0.9,
        },
        'balanced': {  # target: std_ratio ~0.6, sign_bias ~55% ⭐
            'max_depth': 3,
            'learning_rate': 0.015,
            'n_estimators': 200,
            'subsample': 0.68,
            'colsample_bytree': 0.6,
            'colsample_bynode': 0.6,
            'min_child_weight': 7,
            'gamma': 0.3,
            'reg_alpha': 2.0,
            'reg_lambda': 4.5,
            'max_delta_step': 1.2,  # key: 2.4x higher than current
        },
        'moderate': {
            'max_depth': 3,
            'learning_rate': 0.018,
            'n_estimators': 230,
            'subsample': 0.72,
            'colsample_bytree': 0.65,
            'colsample_bynode': 0.65,
            'min_child_weight': 6,
            'gamma': 0.25,
            'reg_alpha': 1.5,
            'reg_lambda': 3.5,
            'max_delta_step': 1.6,
        },
        'aggressive': {
            'max_depth': 4,
            'learning_rate': 0.022,
            'n_estimators': 260,
            'subsample': 0.75,
            'colsample_bytree': 0.7,
            'colsample_bynode': 0.7,
            'min_child_weight': 5,
            'gamma': 0.15,
            'reg_alpha': 1.0,
            'reg_lambda': 2.5,
            'max_delta_step': 2.0,
        },
    },
    8: {  # 8-week: currently 0.09 std_ratio + 98% positive bias
        'ultra_conservative': {  # current params (baseline)
            'max_depth': 2,
            'learning_rate': 0.008,
            'n_estimators': 120,
            'subsample': 0.55,
            'colsample_bytree': 0.5,
            'colsample_bynode': 0.4,
            'min_child_weight': 15,
            'gamma': 0.6,
            'reg_alpha': 4.0,
            'reg_lambda': 10.0,
            'max_delta_step': 0.3,
        },
        'conservative': {
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 140,
            'subsample': 0.6,
            'colsample_bytree': 0.52,
            'colsample_bynode': 0.48,
            'min_child_weight': 12,
            'gamma': 0.5,
            'reg_alpha': 3.0,
            'reg_lambda': 7.5,
            'max_delta_step': 0.6,
        },
        'balanced': {  # target: std_ratio ~0.5, sign_bias ~60% ⭐
            'max_depth': 3,
            'learning_rate': 0.012,
            'n_estimators': 180,
            'subsample': 0.65,
            'colsample_bytree': 0.55,
            'colsample_bynode': 0.55,
            'min_child_weight': 10,
            'gamma': 0.4,
            'reg_alpha': 2.5,
            'reg_lambda': 6.0,
            'max_delta_step': 0.8,  # key: 2.7x higher than current
        },
        'moderate': {
            'max_depth': 3,
            'learning_rate': 0.015,
            'n_estimators': 200,
            'subsample': 0.68,
            'colsample_bytree': 0.6,
            'colsample_bynode': 0.6,
            'min_child_weight': 8,
            'gamma': 0.35,
            'reg_alpha': 2.0,
            'reg_lambda': 5.0,
            'max_delta_step': 1.1,
        },
        'aggressive': {
            'max_depth': 3,
            'learning_rate': 0.018,
            'n_estimators': 220,
            'subsample': 0.72,
            'colsample_bytree': 0.65,
            'colsample_bynode': 0.65,
            'min_child_weight': 7,
            'gamma': 0.25,
            'reg_alpha': 1.5,
            'reg_lambda': 4.0,
            'max_delta_step': 1.4,
        },
    },
}
