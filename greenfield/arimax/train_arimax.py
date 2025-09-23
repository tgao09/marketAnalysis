import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
from tqdm import tqdm

# Add the arimax directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arimax_model import StockARIMAX

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_single_stock(args: tuple) -> Dict[str, Any]:
    """
    Train ARIMAX model for a single stock.

    Args:
        args: Tuple of (ticker, df, train_size, models_dir, use_cv)

    Returns:
        Training results dictionary
    """
    ticker, df, train_size, models_dir, use_cv = args

    try:
        logger.info(f"Training model for {ticker} with {'CV' if use_cv else 'AIC'} selection")

        # Create model instance
        model = StockARIMAX(ticker=ticker, max_p=6, max_d=2, max_q=6)

        # Fit the model with optional cross-validation
        results = model.fit(df, train_size=train_size, use_cv=use_cv)

        # Save the model
        model_path = os.path.join(models_dir, f"{ticker}_arimax.pkl")
        model.save_model(model_path)

        # Add model path to results
        results['model_path'] = model_path
        results['status'] = 'success'

        selection_method = results.get('selection_method', 'AIC')
        rmse_str = f"{results.get('test_rmse', 0):.4f}" if results.get('test_rmse') is not None else 'N/A'
        logger.info(f"Successfully trained {ticker}: RMSE={rmse_str} (selected via {selection_method})")

        return results

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to train {ticker}: {error_msg}")

        # Add more specific error information
        import traceback
        logger.debug(f"Full traceback for {ticker}: {traceback.format_exc()}")

        return {
            'ticker': ticker,
            'status': 'failed',
            'error': error_msg,
            'error_type': type(e).__name__,
            'order': None,
            'aic': None,
            'test_rmse': None,
            'test_mae': None,
            'test_mape': None,
            'directional_accuracy': None,
            'selection_method': None
        }

def prepare_training_data(data_file: str = '../dataset/stock_dataset_with_lags.csv') -> pd.DataFrame:
    """
    Load and prepare the training dataset.

    Args:
        data_file: Path to the lagged dataset file

    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading training data from {data_file}")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Training data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by ticker and date
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} records for {df['ticker'].nunique()} unique stocks")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df

def filter_stocks_for_training(df: pd.DataFrame, min_observations: int = 50) -> List[str]:
    """
    Filter stocks that have sufficient data for training.

    Args:
        df: Full dataset
        min_observations: Minimum number of observations required per stock

    Returns:
        List of tickers suitable for training
    """
    stock_counts = df.groupby('ticker').size()
    valid_stocks = stock_counts[stock_counts >= min_observations].index.tolist()

    logger.info(f"Found {len(valid_stocks)} stocks with at least {min_observations} observations")
    logger.info(f"Excluded {df['ticker'].nunique() - len(valid_stocks)} stocks with insufficient data")

    return valid_stocks

def train_all_models(data_file: str = '../dataset/stock_dataset_with_lags.csv',
                    models_dir: str = 'arimaxmodels',
                    results_dir: str = 'arimaxresults',
                    train_size: float = 0.8,
                    max_workers: int = 4,
                    min_observations: int = 50,
                    sample_stocks: int = None,
                    use_cv: bool = True) -> pd.DataFrame:
    """
    Train ARIMAX models for all stocks in parallel with optional cross-validation.

    Args:
        data_file: Path to the lagged dataset
        models_dir: Directory to save trained models
        results_dir: Directory to save results
        train_size: Proportion of data for training
        max_workers: Number of parallel processes
        min_observations: Minimum observations required per stock
        sample_stocks: If specified, train only on this many stocks (for testing)
        use_cv: Whether to use cross-validation for model selection

    Returns:
        DataFrame with training results for all stocks
    """
    start_time = time.time()

    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    df = prepare_training_data(data_file)

    # Filter stocks
    valid_stocks = filter_stocks_for_training(df, min_observations)

    # Sample stocks if requested (useful for testing)
    if sample_stocks and sample_stocks < len(valid_stocks):
        valid_stocks = valid_stocks[:sample_stocks]
        logger.info(f"Training on sample of {sample_stocks} stocks: {valid_stocks}")

    # Prepare arguments for parallel processing
    training_args = [(ticker, df, train_size, models_dir, use_cv) for ticker in valid_stocks]

    # Train models in parallel
    results = []
    failed_count = 0

    cv_info = "with CV" if use_cv else "with AIC selection"
    logger.info(f"Starting training for {len(valid_stocks)} stocks using {max_workers} workers {cv_info}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training jobs
        future_to_ticker = {
            executor.submit(train_single_stock, args): args[0]
            for args in training_args
        }

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker),
                          desc="Training models", unit="stock"):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results.append(result)

                if result['status'] == 'failed':
                    failed_count += 1

            except Exception as e:
                logger.error(f"Unexpected error for {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'status': 'failed',
                    'error': str(e)
                })
                failed_count += 1

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results summary
    summary_file = os.path.join(results_dir, 'model_summary.csv')
    results_df.to_csv(summary_file, index=False)

    # Calculate summary statistics
    successful_models = len(results_df[results_df['status'] == 'success'])
    total_time = time.time() - start_time

    logger.info(f"Training completed in {total_time:.1f} seconds")
    logger.info(f"Successfully trained: {successful_models}/{len(valid_stocks)} models")
    logger.info(f"Failed: {failed_count}/{len(valid_stocks)} models")
    logger.info(f"Results saved to: {summary_file}")

    # Display summary statistics for successful models
    if successful_models > 0:
        successful_df = results_df[results_df['status'] == 'success']

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total stocks processed: {len(valid_stocks)}")
        print(f"Successful models: {successful_models}")
        print(f"Failed models: {failed_count}")
        print(f"Success rate: {successful_models/len(valid_stocks)*100:.1f}%")
        print(f"Training time: {total_time:.1f} seconds")
        print(f"Model selection: {'Cross-validation' if use_cv else 'AIC criterion'}")

        # Show breakdown by selection method if CV was attempted
        if 'selection_method' in successful_df.columns:
            method_counts = successful_df['selection_method'].value_counts()
            print(f"\nSelection Method Breakdown:")
            for method, count in method_counts.items():
                print(f"  {method}: {count} models ({count/len(successful_df)*100:.1f}%)")

        if 'test_rmse' in successful_df.columns:
            print(f"\nTest Performance (successful models):")
            print(f"Mean RMSE: {successful_df['test_rmse'].mean():.4f}")
            print(f"Median RMSE: {successful_df['test_rmse'].median():.4f}")
            print(f"Best RMSE: {successful_df['test_rmse'].min():.4f} ({successful_df.loc[successful_df['test_rmse'].idxmin(), 'ticker']})")
            print(f"Worst RMSE: {successful_df['test_rmse'].max():.4f} ({successful_df.loc[successful_df['test_rmse'].idxmax(), 'ticker']})")

        if 'directional_accuracy' in successful_df.columns:
            print(f"\nDirectional Accuracy:")
            print(f"Mean: {successful_df['directional_accuracy'].mean():.1f}%")
            print(f"Median: {successful_df['directional_accuracy'].median():.1f}%")
            print(f"Best: {successful_df['directional_accuracy'].max():.1f}% ({successful_df.loc[successful_df['directional_accuracy'].idxmax(), 'ticker']})")

        print("="*60)

    return results_df

def main():
    """Main training script."""
    import argparse

    parser = argparse.ArgumentParser(description='Train ARIMAX models for all stocks')
    parser.add_argument('--data-file', type=str, default='../dataset/stock_dataset_with_lags.csv',
                       help='Path to the lagged dataset file')
    parser.add_argument('--models-dir', type=str, default='arimaxmodels',
                       help='Directory to save trained models')
    parser.add_argument('--results-dir', type=str, default='arimaxresults',
                       help='Directory to save results')
    parser.add_argument('--train-size', type=float, default=0.8,
                       help='Proportion of data for training (default: 0.8)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel processes (default: 4)')
    parser.add_argument('--min-observations', type=int, default=50,
                       help='Minimum observations required per stock (default: 50)')
    parser.add_argument('--sample-stocks', type=int, default=None,
                       help='Train only on this many stocks (for testing)')
    parser.add_argument('--no-cv', action='store_true',
                       help='Disable cross-validation and use AIC for model selection')

    args = parser.parse_args()

    # Suppress warnings during training
    warnings.filterwarnings('ignore')

    try:
        results_df = train_all_models(
            data_file=args.data_file,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            train_size=args.train_size,
            max_workers=args.max_workers,
            min_observations=args.min_observations,
            sample_stocks=args.sample_stocks,
            use_cv=not args.no_cv
        )

        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {os.path.join(args.results_dir, 'model_summary.csv')}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()