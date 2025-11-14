# !/usr/bin/env python3


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.linalg import cholesky
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UncertaintyPropagator:
    

    def __init__(self, confidence_levels: List[float] = [0.68, 0.95]):
        
        self.confidence_levels = confidence_levels
        self.z_scores = {
            0.68: 1.0,   # ~68% confidence (1 sigma)
            0.90: 1.645, # 90% confidence
            0.95: 1.96,  # 95% confidence
            0.99: 2.576  # 99% confidence
        }

    def compute_exogenous_uncertainty_matrix(self, exog_forecasts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                           steps: int) -> Tuple[np.ndarray, np.ndarray]:
        
        n_variables = len(exog_forecasts)
        variable_names = list(exog_forecasts.keys())

        # initialize uncertainty storage
        uncertainty_std = np.zeros((steps, n_variables))

        # compute standard deviations from confidence intervals (assuming normal distribution)
        for i, var_name in enumerate(variable_names):
            predictions, lower_bounds, upper_bounds = exog_forecasts[var_name]

            # convert confidence intervals to standard deviations
            # assuming 95% confidence intervals
            std_devs = (upper_bounds - lower_bounds) / (2 * self.z_scores[0.95])
            uncertainty_std[:, i] = std_devs

        # create covariance matrix (assuming independence for simplicity)
        # in practice, you might want to estimate cross-correlations
        uncertainty_matrices = []
        for step in range(steps):
            step_std = uncertainty_std[step, :]
            # diagonal covariance matrix (independence assumption)
            cov_matrix = np.diag(step_std ** 2)
            uncertainty_matrices.append(cov_matrix)

        return np.array(uncertainty_matrices), uncertainty_std

    def propagate_model_uncertainty(self, arimax_predictions: np.ndarray,
                                  arimax_confidence_intervals: np.ndarray,
                                  exog_uncertainty_std: np.ndarray) -> Dict[str, np.ndarray]:
        
        steps = len(arimax_predictions)

        # extract arimax model uncertainty
        arimax_lower = arimax_confidence_intervals[:, 0]
        arimax_upper = arimax_confidence_intervals[:, 1]
        arimax_std = (arimax_upper - arimax_lower) / (2 * self.z_scores[0.95])

        # estimate sensitivity of arimax to exogenous variables
        # this is a simplified approach - in practice, you'd want to compute actual gradients
        exog_sensitivity = self._estimate_exogenous_sensitivity(exog_uncertainty_std, arimax_std)

        # combine uncertainties
        combined_std = np.sqrt(arimax_std**2 + exog_sensitivity**2)

        # generate bounds for different confidence levels
        uncertainty_bounds = {}
        for conf_level in self.confidence_levels:
            z_score = self.z_scores.get(conf_level, 1.96)
            lower_bounds = arimax_predictions - z_score * combined_std
            upper_bounds = arimax_predictions + z_score * combined_std
            uncertainty_bounds[conf_level] = (lower_bounds, upper_bounds)

        return uncertainty_bounds

    def _estimate_exogenous_sensitivity(self, exog_uncertainty_std: np.ndarray,
                                      arimax_std: np.ndarray) -> np.ndarray:
        
        # simple heuristic: assume exogenous uncertainty contributes proportionally
        # this could be improved with actual sensitivity analysis

        # sum uncertainty across all exogenous variables (assuming they contribute additively)
        total_exog_uncertainty = np.sqrt(np.sum(exog_uncertainty_std**2, axis=1))

        # scale by a factor that represents the typical sensitivity
        # this factor should ideally be estimated from the arimax model coefficients
        sensitivity_factor = 0.3  # rough estimate - could be calibrated

        return sensitivity_factor * total_exog_uncertainty

    def monte_carlo_uncertainty(self, exog_forecasts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                              arimax_model, steps: int, n_simulations: int = 1000) -> Dict[str, np.ndarray]:
        
        try:
            variable_names = list(exog_forecasts.keys())
            simulated_predictions = []

            logger.info(f"Running Monte Carlo uncertainty propagation with {n_simulations} simulations")

            for sim in range(n_simulations):
                # generate random samples for exogenous variables
                sim_exog_data = []

                for step in range(steps):
                    step_data = {}
                    for var_name in variable_names:
                        pred, lower, upper = exog_forecasts[var_name]

                        # assume normal distribution
                        std_dev = (upper[step] - lower[step]) / (2 * self.z_scores[0.95])
                        sample = np.random.normal(pred[step], std_dev)
                        step_data[var_name] = sample

                    sim_exog_data.append(step_data)

                # convert to dataframe format expected by arimax
                sim_exog_df = pd.DataFrame(sim_exog_data)

                # get arimax prediction for this simulation
                try:
                    sim_prediction = arimax_model.predict(sim_exog_df, steps=steps)
                    simulated_predictions.append(sim_prediction)
                except Exception as e:
                    # skip failed simulations
                    continue

            if not simulated_predictions:
                raise ValueError("All Monte Carlo simulations failed")

            # convert to array
            simulated_predictions = np.array(simulated_predictions)

            # compute empirical confidence intervals
            uncertainty_bounds = {}
            for conf_level in self.confidence_levels:
                alpha = 1 - conf_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                lower_bounds = np.percentile(simulated_predictions, lower_percentile, axis=0)
                upper_bounds = np.percentile(simulated_predictions, upper_percentile, axis=0)

                uncertainty_bounds[conf_level] = (lower_bounds, upper_bounds)

            logger.info(f"Monte Carlo uncertainty propagation completed with {len(simulated_predictions)} successful simulations")
            return uncertainty_bounds

        except Exception as e:
            logger.error(f"Monte Carlo uncertainty propagation failed: {e}")
            return {}

    def validate_uncertainty_bounds(self, predictions: np.ndarray,
                                  uncertainty_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, bool]:
        
        validation_results = {}

        # check that bounds are properly ordered
        properly_ordered = True
        for conf_level, (lower, upper) in uncertainty_bounds.items():
            if not np.all(lower <= predictions) or not np.all(predictions <= upper):
                properly_ordered = False
                break

        validation_results['properly_ordered'] = properly_ordered

        # check that higher confidence levels have wider bounds
        confidence_levels = sorted(uncertainty_bounds.keys())
        widths_increasing = True

        for i in range(1, len(confidence_levels)):
            prev_level = confidence_levels[i-1]
            curr_level = confidence_levels[i]

            prev_width = uncertainty_bounds[prev_level][1] - uncertainty_bounds[prev_level][0]
            curr_width = uncertainty_bounds[curr_level][1] - uncertainty_bounds[curr_level][0]

            if not np.all(curr_width >= prev_width):
                widths_increasing = False
                break

        validation_results['widths_increasing'] = widths_increasing

        # check for reasonable bounds (not too wide or too narrow)
        bounds_reasonable = True
        for conf_level, (lower, upper) in uncertainty_bounds.items():
            width = upper - lower
            # check if bounds are unreasonably wide (more than 100% return in either direction)
            if np.any(width > 2.0):  # 200% total width
                bounds_reasonable = False
                break
            # check if bounds are unreasonably narrow (less than 0.1% width)
            if np.any(width < 0.001):
                bounds_reasonable = False
                break

        validation_results['bounds_reasonable'] = bounds_reasonable

        return validation_results

    def create_uncertainty_report(self, predictions: np.ndarray,
                                uncertainty_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                future_dates: List[str]) -> Dict[str, any]:
        
        report = {
            'generation_time': pd.Timestamp.now().isoformat(),
            'forecast_steps': len(predictions),
            'confidence_levels': list(uncertainty_bounds.keys()),
            'future_dates': future_dates
        }

        # compute uncertainty metrics
        uncertainty_metrics = {}
        for conf_level, (lower, upper) in uncertainty_bounds.items():
            width = upper - lower
            relative_width = width / np.abs(predictions)

            uncertainty_metrics[conf_level] = {
                'mean_width': np.mean(width),
                'max_width': np.max(width),
                'min_width': np.min(width),
                'mean_relative_width': np.mean(relative_width),
                'expanding_uncertainty': np.all(np.diff(width) >= 0)  # check if uncertainty expands over time
            }

        report['uncertainty_metrics'] = uncertainty_metrics

        # validation results
        report['validation'] = self.validate_uncertainty_bounds(predictions, uncertainty_bounds)

        # summary statistics
        report['summary'] = {
            'prediction_range': {
                'min': np.min(predictions),
                'max': np.max(predictions),
                'mean': np.mean(predictions)
            },
            'uncertainty_expanding': all(
                metrics['expanding_uncertainty']
                for metrics in uncertainty_metrics.values()
            )
        }

        return report

def create_uncertainty_propagator(confidence_levels: List[float] = [0.68, 0.95]) -> UncertaintyPropagator:
    
    return UncertaintyPropagator(confidence_levels)