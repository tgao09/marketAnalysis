import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path

# add the arimax directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "arimax" / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "arimax" / "models"
DEFAULT_DATA_FILE = PROJECT_ROOT / "dataset" / "stock_dataset_with_lags.csv"


def resolve_path(path_str: str) -> Path:
    
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path

from arimax_model import StockARIMAX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    

    def __init__(self, results_dir: str = str(DEFAULT_RESULTS_DIR), models_dir: str = str(DEFAULT_MODELS_DIR)):
        
        self.results_dir = resolve_path(results_dir)
        self.models_dir = resolve_path(models_dir)
        self.summary_df = None

    def load_training_results(self) -> pd.DataFrame:
        
        summary_file = self.results_dir / 'model_summary.csv'

        if not summary_file.exists():
            raise FileNotFoundError(f"Training results not found: {summary_file}")

        self.summary_df = pd.read_csv(summary_file)
        logger.info(f"Loaded results for {len(self.summary_df)} models")

        return self.summary_df

    def evaluate_single_model(self, ticker: str, data_file: str) -> Dict[str, Any]:
        
        # load model
        model_path = self.models_dir / f"{ticker}_arimax.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = StockARIMAX.load_model(model_path)

        # load test data
        data_path = resolve_path(data_file)
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # prepare data for this stock
        target, exog = model.prepare_data(df)

        # split into train/test
        split_idx = int(len(target) * 0.8)
        test_target = target[split_idx:]
        test_exog = exog[split_idx:]

        if len(test_target) == 0:
            return {'error': 'No test data available'}

        # generate predictions
        try:
            predictions = model.predict(test_exog, steps=len(test_target))
            pred_with_conf, conf_intervals = model.predict(test_exog, steps=len(test_target),
                                                          return_conf_int=True)

            # calculate detailed metrics
            residuals = test_target - predictions

            metrics = {
                'ticker': ticker,
                'test_observations': len(test_target),
                'mae': np.mean(np.abs(residuals)),
                'rmse': np.sqrt(np.mean(residuals**2)),
                'mape': np.mean(np.abs(residuals / test_target)) * 100,
                'mse': np.mean(residuals**2),
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals),
                'min_residual': np.min(residuals),
                'max_residual': np.max(residuals),

                # directional accuracy
                'directional_accuracy': np.mean(np.sign(test_target) == np.sign(predictions)) * 100,

                # statistical tests
                'residual_normality_pvalue': stats.jarque_bera(residuals)[1],
                'residual_autocorr': self._ljung_box_test(residuals),

                # confidence interval coverage
                'ci_coverage_95': self._calculate_coverage(test_target, conf_intervals, 0.95),

                # feature importance
                'feature_importance': model.get_feature_importance().to_dict('records') if not model.get_feature_importance().empty else [],

                # model info
                'arima_order': model.best_order,
                'aic': model.aic_score,
                'num_features': len(model.feature_columns)
            }

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}

    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=lags, return_df=False)
            return result[1][-1]  # p-value for the last lag
        except Exception:
            return np.nan

    def _calculate_coverage(self, actual: np.ndarray, conf_intervals: np.ndarray,
                          confidence_level: float) -> float:
        
        try:
            lower_bound = conf_intervals[:, 0]
            upper_bound = conf_intervals[:, 1]
            coverage = np.mean((actual >= lower_bound) & (actual <= upper_bound)) * 100
            return coverage
        except Exception:
            return np.nan

    def generate_performance_report(self, data_file: str = str(DEFAULT_DATA_FILE),
                                  top_n: int = 10) -> Dict[str, Any]:
        
        if self.summary_df is None:
            self.load_training_results()

        successful_models = self.summary_df[self.summary_df['status'] == 'success']

        if len(successful_models) == 0:
            return {'error': 'No successful models found'}

        report = {
            'total_models': len(self.summary_df),
            'successful_models': len(successful_models),
            'failed_models': len(self.summary_df) - len(successful_models),
            'success_rate': len(successful_models) / len(self.summary_df) * 100
        }

        # performance statistics
        if 'test_rmse' in successful_models.columns:
            rmse_stats = successful_models['test_rmse'].describe()
            report['rmse_statistics'] = {
                'mean': rmse_stats['mean'],
                'median': rmse_stats['50%'],
                'std': rmse_stats['std'],
                'min': rmse_stats['min'],
                'max': rmse_stats['max'],
                'q25': rmse_stats['25%'],
                'q75': rmse_stats['75%']
            }

            # top and bottom performers
            report['best_performers'] = successful_models.nsmallest(top_n, 'test_rmse')[
                ['ticker', 'test_rmse', 'directional_accuracy', 'order']].to_dict('records')
            report['worst_performers'] = successful_models.nlargest(top_n, 'test_rmse')[
                ['ticker', 'test_rmse', 'directional_accuracy', 'order']].to_dict('records')

        # directional accuracy statistics
        if 'directional_accuracy' in successful_models.columns:
            da_stats = successful_models['directional_accuracy'].describe()
            report['directional_accuracy_statistics'] = {
                'mean': da_stats['mean'],
                'median': da_stats['50%'],
                'std': da_stats['std'],
                'min': da_stats['min'],
                'max': da_stats['max']
            }

        # arima order distribution
        if 'order' in successful_models.columns:
            order_counts = successful_models['order'].value_counts().head(10)
            report['popular_arima_orders'] = order_counts.to_dict()

        # aic distribution
        if 'aic' in successful_models.columns:
            aic_stats = successful_models['aic'].describe()
            report['aic_statistics'] = {
                'mean': aic_stats['mean'],
                'median': aic_stats['50%'],
                'std': aic_stats['std'],
                'min': aic_stats['min'],
                'max': aic_stats['max']
            }

        return report

    def create_performance_plots(self, output_dir: str = None) -> None:
        
        if output_dir is None:
            output_dir_path = self.results_dir
        else:
            output_dir_path = resolve_path(str(output_dir))

        if self.summary_df is None:
            self.load_training_results()

        successful_models = self.summary_df[self.summary_df['status'] == 'success']

        if len(successful_models) == 0:
            logger.warning("No successful models to plot")
            return

        os.makedirs(output_dir_path, exist_ok=True)

        # set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ARIMAX Model Performance Analysis', fontsize=16, fontweight='bold')

        # plot 1: rmse distribution
        if 'test_rmse' in successful_models.columns:
            axes[0, 0].hist(successful_models['test_rmse'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Test RMSE')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Test RMSE')
            axes[0, 0].axvline(successful_models['test_rmse'].mean(), color='red',
                              linestyle='--', label=f"Mean: {successful_models['test_rmse'].mean():.4f}")
            axes[0, 0].legend()

        # plot 2: directional accuracy distribution
        if 'directional_accuracy' in successful_models.columns:
            axes[0, 1].hist(successful_models['directional_accuracy'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Directional Accuracy (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Directional Accuracy')
            axes[0, 1].axvline(successful_models['directional_accuracy'].mean(), color='red',
                              linestyle='--', label=f"Mean: {successful_models['directional_accuracy'].mean():.1f}%")
            axes[0, 1].legend()

        # plot 3: rmse vs directional accuracy scatter plot
        if 'test_rmse' in successful_models.columns and 'directional_accuracy' in successful_models.columns:
            scatter = axes[1, 0].scatter(successful_models['test_rmse'], successful_models['directional_accuracy'],
                                       alpha=0.6)
            axes[1, 0].set_xlabel('Test RMSE')
            axes[1, 0].set_ylabel('Directional Accuracy (%)')
            axes[1, 0].set_title('RMSE vs Directional Accuracy')

            # add correlation coefficient
            corr = successful_models['test_rmse'].corr(successful_models['directional_accuracy'])
            axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                           transform=axes[1, 0].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

        # plot 4: aic distribution
        if 'aic' in successful_models.columns:
            axes[1, 1].hist(successful_models['aic'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('AIC Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of AIC Scores')
            axes[1, 1].axvline(successful_models['aic'].mean(), color='red',
                              linestyle='--', label=f"Mean: {successful_models['aic'].mean():.1f}")
            axes[1, 1].legend()

        plt.tight_layout()

        # save plot
        plot_file = output_dir_path / 'performance_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Performance plots saved to: {plot_file}")

    def print_summary_report(self, data_file: str = str(DEFAULT_DATA_FILE)) -> None:
        
        report = self.generate_performance_report(data_file)

        print("\n" + "="*80)
        print("ARIMAX MODEL EVALUATION SUMMARY")
        print("="*80)

        print(f"Total models trained: {report['total_models']}")
        print(f"Successful models: {report['successful_models']}")
        print(f"Failed models: {report['failed_models']}")
        print(f"Success rate: {report['success_rate']:.1f}%")

        if 'rmse_statistics' in report:
            print(f"\nRMSE Statistics:")
            rmse = report['rmse_statistics']
            print(f"  Mean: {rmse['mean']:.4f}")
            print(f"  Median: {rmse['median']:.4f}")
            print(f"  Std Dev: {rmse['std']:.4f}")
            print(f"  Range: {rmse['min']:.4f} - {rmse['max']:.4f}")

        if 'directional_accuracy_statistics' in report:
            print(f"\nDirectional Accuracy Statistics:")
            da = report['directional_accuracy_statistics']
            print(f"  Mean: {da['mean']:.1f}%")
            print(f"  Median: {da['median']:.1f}%")
            print(f"  Std Dev: {da['std']:.1f}%")
            print(f"  Range: {da['min']:.1f}% - {da['max']:.1f}%")

        if 'best_performers' in report:
            print(f"\nTop 5 Best Performers (by RMSE):")
            for i, model in enumerate(report['best_performers'][:5], 1):
                print(f"  {i}. {model['ticker']}: RMSE={model['test_rmse']:.4f}, "
                     f"Dir.Acc={model['directional_accuracy']:.1f}%, Order={model['order']}")

        if 'popular_arima_orders' in report:
            print(f"\nMost Popular ARIMA Orders:")
            for order, count in list(report['popular_arima_orders'].items())[:5]:
                print(f"  {order}: {count} models")

        print("="*80)

def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained ARIMAX models')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Evaluate specific ticker (default: evaluate all)')
    parser.add_argument('--data-file', type=str, default=str(DEFAULT_DATA_FILE),
                       help='Path to the test dataset')
    parser.add_argument('--results-dir', type=str, default=str(DEFAULT_RESULTS_DIR),
                       help='Directory containing results')
    parser.add_argument('--models-dir', type=str, default=str(DEFAULT_MODELS_DIR),
                       help='Directory containing trained models')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create performance visualization plots')

    args = parser.parse_args()

    # suppress warnings
    warnings.filterwarnings('ignore')

    evaluator = ModelEvaluator(args.results_dir, args.models_dir)

    try:
        if args.ticker:
            # evaluate single model
            print(f"Evaluating model for {args.ticker}...")
            result = evaluator.evaluate_single_model(args.ticker, args.data_file)

            if 'error' in result:
                print(f"Evaluation failed: {result['error']}")
            else:
                print(f"\nDetailed evaluation for {args.ticker}:")
                print("-" * 50)
                for key, value in result.items():
                    if key != 'feature_importance':
                        print(f"{key}: {value}")

        else:
            # evaluate all models
            print("Evaluating all models...")
            evaluator.load_training_results()
            evaluator.print_summary_report(args.data_file)

            if args.create_plots:
                print("\nCreating performance plots...")
                evaluator.create_performance_plots()

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
