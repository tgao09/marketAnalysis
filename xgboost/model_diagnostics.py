

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelDiagnostics:
    

    @staticmethod
    def calculate_prediction_distribution_metrics(
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        
        metrics = {
            'pct_positive': 100 * (y_pred > 0).sum() / len(y_pred),
            'pct_negative': 100 * (y_pred < 0).sum() / len(y_pred),
            'pred_mean': float(y_pred.mean()),
            'pred_std': float(y_pred.std()),
            'pred_median': float(np.median(y_pred)),
            'true_mean': float(y_true.mean()),
            'true_std': float(y_true.std()),
            'true_median': float(np.median(y_true)),
            'pred_min': float(y_pred.min()),
            'pred_max': float(y_pred.max()),
            'true_min': float(y_true.min()),
            'true_max': float(y_true.max()),
            'pred_range': float(y_pred.max() - y_pred.min()),
            'true_range': float(y_true.max() - y_true.min()),
            'prediction_diversity': int(len(np.unique(y_pred.round(4)))),
        }

        # ratio metrics
        if y_true.std() > 0:
            metrics['std_ratio'] = float(y_pred.std() / y_true.std())
        else:
            metrics['std_ratio'] = np.nan

        if y_true.max() - y_true.min() > 0:
            metrics['range_ratio'] = float(
                (y_pred.max() - y_pred.min()) / (y_true.max() - y_true.min())
            )
        else:
            metrics['range_ratio'] = np.nan

        return metrics

    @staticmethod
    def check_sign_bias(y_pred: np.ndarray, threshold: float = 0.95) -> bool:
        
        pct_pos = (y_pred > 0).sum() / len(y_pred)
        return pct_pos > threshold or pct_pos < (1 - threshold)

    @staticmethod
    def calculate_calibration_metrics(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        
        # bin predictions into deciles/quantiles
        try:
            bins = pd.qcut(y_pred, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # if not enough unique values, use fewer bins
            try:
                bins = pd.qcut(y_pred, q=5, labels=False, duplicates='drop')
            except ValueError:
                return {
                    'calibration_error': np.nan,
                    'calibration_data': [],
                    'note': 'Too few unique predictions for calibration analysis'
                }

        calibration_data = []
        calibration_errors = []

        for i in range(int(bins.max()) + 1):
            mask = bins == i
            if mask.sum() > 0:
                pred_mean = float(y_pred[mask].mean())
                actual_mean = float(y_true[mask].mean())
                calib_error = abs(pred_mean - actual_mean)

                calibration_data.append({
                    'bin': i,
                    'pred_mean': pred_mean,
                    'actual_mean': actual_mean,
                    'calibration_error': calib_error,
                    'count': int(mask.sum())
                })
                calibration_errors.append(calib_error)

        # overall calibration error (mean absolute error between predicted and actual means)
        mean_calib_error = float(np.mean(calibration_errors)) if calibration_errors else np.nan

        return {
            'calibration_error': mean_calib_error,
            'calibration_data': calibration_data
        }

    @staticmethod
    def calculate_robustness_metrics(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_bootstrap: int = 30
    ) -> Dict:
        
        pred_means = []
        pred_stds = []
        pred_sign_ratios = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            preds = model.predict(X.iloc[idx])
            pred_means.append(preds.mean())
            pred_stds.append(preds.std())
            pred_sign_ratios.append((preds > 0).sum() / len(preds))

        return {
            'mean_stability': float(np.std(pred_means)),
            'std_stability': float(np.std(pred_stds)),
            'sign_ratio_stability': float(np.std(pred_sign_ratios)),
            'mean_of_means': float(np.mean(pred_means)),
            'mean_of_stds': float(np.mean(pred_stds)),
        }

    @staticmethod
    def validate_model_quality(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        thresholds: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        
        if thresholds is None:
            thresholds = {
                'max_sign_bias': 0.85,          # fail if >85% same sign
                'min_prediction_diversity': 100, # at least 100 unique predictions
                'max_pred_std_ratio': 3.0,      # pred_std < 3x true_std
                'min_pred_std_ratio': 0.3,      # pred_std > 0.3x true_std
                'max_calibration_error': 0.05,  # mean calib error < 5%
            }

        failures = []

        # check sign bias
        pct_pos = (y_pred > 0).sum() / len(y_pred)
        if pct_pos > thresholds['max_sign_bias'] or pct_pos < (1 - thresholds['max_sign_bias']):
            failures.append(
                f"SIGN_BIAS: {pct_pos*100:.1f}% positive (threshold: {thresholds['max_sign_bias']*100:.1f}%)"
            )

        # check prediction diversity
        n_unique = len(np.unique(y_pred.round(4)))
        if n_unique < thresholds['min_prediction_diversity']:
            failures.append(
                f"LOW_DIVERSITY: {n_unique} unique predictions (min: {thresholds['min_prediction_diversity']})"
            )

        # check std ratio
        std_ratio = y_pred.std() / y_true.std() if y_true.std() > 0 else np.inf
        if std_ratio > thresholds['max_pred_std_ratio']:
            failures.append(
                f"OVERFITTING: pred_std/true_std = {std_ratio:.2f} (max: {thresholds['max_pred_std_ratio']})"
            )
        elif std_ratio < thresholds['min_pred_std_ratio']:
            failures.append(
                f"UNDERFITTING: pred_std/true_std = {std_ratio:.2f} (min: {thresholds['min_pred_std_ratio']})"
            )

        # check calibration
        calib_metrics = ModelDiagnostics.calculate_calibration_metrics(y_pred, y_true)
        if not np.isnan(calib_metrics['calibration_error']):
            if calib_metrics['calibration_error'] > thresholds['max_calibration_error']:
                failures.append(
                    f"MISCALIBRATION: error = {calib_metrics['calibration_error']:.4f} "
                    f"(max: {thresholds['max_calibration_error']})"
                )

        is_valid = len(failures) == 0

        return is_valid, failures

    @staticmethod
    def print_diagnostic_report(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "Model"
    ):
        
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC REPORT: {model_name}")
        print(f"{'='*70}")

        # distribution metrics
        dist_metrics = ModelDiagnostics.calculate_prediction_distribution_metrics(y_pred, y_true)
        print(f"\n📊 DISTRIBUTION METRICS:")
        print(f"  Positive predictions: {dist_metrics['pct_positive']:.1f}%")
        print(f"  Negative predictions: {dist_metrics['pct_negative']:.1f}%")
        print(f"  Prediction diversity: {dist_metrics['prediction_diversity']} unique values")
        print(f"\n  Predicted:  mean={dist_metrics['pred_mean']:.4f}, "
              f"std={dist_metrics['pred_std']:.4f}, range=[{dist_metrics['pred_min']:.4f}, {dist_metrics['pred_max']:.4f}]")
        print(f"  Actual:     mean={dist_metrics['true_mean']:.4f}, "
              f"std={dist_metrics['true_std']:.4f}, range=[{dist_metrics['true_min']:.4f}, {dist_metrics['true_max']:.4f}]")
        print(f"\n  Std ratio: {dist_metrics['std_ratio']:.3f} (ideal: 0.8-1.2)")
        print(f"  Range ratio: {dist_metrics['range_ratio']:.3f} (ideal: 0.8-1.2)")

        # sign bias check
        has_bias = ModelDiagnostics.check_sign_bias(y_pred, threshold=0.85)
        bias_status = "⚠️ WARNING" if has_bias else "✅ OK"
        print(f"\n🎯 SIGN BIAS: {bias_status}")

        # calibration
        calib_metrics = ModelDiagnostics.calculate_calibration_metrics(y_pred, y_true)
        print(f"\n📏 CALIBRATION:")
        print(f"  Mean calibration error: {calib_metrics['calibration_error']:.4f}")
        if calib_metrics['calibration_data']:
            print(f"  Sample bins:")
            for item in calib_metrics['calibration_data'][:3]:
                print(f"    Bin {item['bin']}: pred={item['pred_mean']:+.4f}, "
                      f"actual={item['actual_mean']:+.4f}, error={item['calibration_error']:.4f}")

        # validation
        is_valid, failures = ModelDiagnostics.validate_model_quality(y_pred, y_true)
        print(f"\n✓ VALIDATION: {'PASSED ✅' if is_valid else 'FAILED ❌'}")
        if failures:
            print(f"  Issues detected:")
            for failure in failures:
                print(f"    • {failure}")

        print(f"{'='*70}\n")


def detect_feature_leakage(
    feature_importance: pd.DataFrame,
    feature_cols: List[str],
    threshold_importance: float = 0.3
) -> List[Dict]:
    
    suspicious = []

    if feature_importance.empty:
        return suspicious

    # normalize to percentage
    importance_df = feature_importance.copy()
    importance_df['importance_pct'] = (
        importance_df['importance'] / importance_df['importance'].sum()
    )

    # check 1: dominant single feature
    if importance_df['importance_pct'].iloc[0] > threshold_importance:
        top_feature = importance_df.iloc[0]
        suspicious.append({
            'feature': top_feature['feature'],
            'importance': float(top_feature['importance_pct']),
            'reason': f'Dominates model (>{threshold_importance*100:.0f}% importance)'
        })

    # check 2: forward-looking feature names
    for _, row in importance_df.iterrows():
        if any(keyword in str(row['feature']).lower()
               for keyword in ['forward', 'future', 'next', 'ahead']):
            suspicious.append({
                'feature': row['feature'],
                'importance': float(row['importance_pct']),
                'reason': 'Suspicious name (forward-looking keyword)'
            })

    # check 3: cross-sectional features too dominant
    cross_sectional_keywords = ['rank', 'zscore', 'percentile', 'quantile', 'decile']
    cross_sectional = importance_df[
        importance_df['feature'].str.contains('|'.join(cross_sectional_keywords), case=False, na=False)
    ]
    if not cross_sectional.empty and cross_sectional['importance_pct'].sum() > 0.5:
        suspicious.append({
            'feature': 'cross_sectional_features_combined',
            'importance': float(cross_sectional['importance_pct'].sum()),
            'reason': 'Cross-sectional features >50% importance (may leak via ranking)'
        })

    # check 4: target-related features (should never happen but check anyway)
    target_keywords = ['return', 'price', 'close', 'high', 'low']
    for _, row in importance_df.head(10).iterrows():  # check top 10 features
        feature_name = str(row['feature']).lower()
        # check if feature contains target keywords but not lag indicators
        if any(kw in feature_name for kw in target_keywords):
            if not any(lag in feature_name for lag in ['lag', '_l', 'prev', 'past']):
                suspicious.append({
                    'feature': row['feature'],
                    'importance': float(row['importance_pct']),
                    'reason': 'Contains target-related keyword without lag indicator'
                })

    return suspicious
