import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str = 'stock_dataset.csv') -> pd.DataFrame:
    """Load and validate the stock dataset CSV file."""
    try:
        df = pd.read_csv(file_path)

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Validate required columns (updated for normalized returns dataset)
        required_columns = ['ticker', 'Date', 'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by ticker and Date to ensure proper chronological order
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Loaded dataset with {len(df)} records for {df['ticker'].nunique()} stocks")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

        return df

    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def validate_lag_parameters(n_lags: int, features_to_lag: List[str], df_columns: List[str]) -> None:
    """Validate lag parameters before processing."""
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")

    if n_lags > 50:
        logger.warning(f"Large number of lags ({n_lags}) may create a very wide dataset")

    # Check if features_to_lag exist in the dataset
    missing_features = [feature for feature in features_to_lag if feature not in df_columns]
    if missing_features:
        raise ValueError(f"Features not found in dataset: {missing_features}")

    # Warn about non-numeric features
    reserved_columns = ['ticker', 'Date']
    invalid_features = [feature for feature in features_to_lag if feature in reserved_columns]
    if invalid_features:
        raise ValueError(f"Cannot lag reserved columns: {invalid_features}")

def create_lag_features(input_file: str = 'stock_dataset.csv',
                       n_lags: int = 3,
                       features_to_lag: List[str] = None,
                       output_file: str = 'stock_dataset_with_lags.csv') -> pd.DataFrame:
    """
    Create lagged features for time series analysis.

    Args:
        input_file: Path to input CSV file
        n_lags: Number of time steps to lag
        features_to_lag: List of column names to create lags for
        output_file: Path to save the output CSV file

    Returns:
        pandas.DataFrame: Dataset with original and lagged features
    """
    # Set default features to lag if not specified (updated for normalized returns)
    if features_to_lag is None:
        features_to_lag = ['weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']

    # Load the dataset
    df = load_dataset(input_file)

    # Validate parameters
    validate_lag_parameters(n_lags, features_to_lag, df.columns.tolist())

    logger.info(f"Creating {n_lags} lag features for: {features_to_lag}")

    # Create lagged features
    lagged_dfs = []

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()

        # Create lag features for this ticker
        for feature in features_to_lag:
            for lag in range(1, n_lags + 1):
                lag_column_name = f"{feature}_lag_{lag}"
                ticker_data[lag_column_name] = ticker_data[feature].shift(lag)

        lagged_dfs.append(ticker_data)
        logger.debug(f"Processed lags for {ticker}")

    # Combine all ticker data
    result_df = pd.concat(lagged_dfs, ignore_index=True)

    # Sort by ticker and Date
    result_df = result_df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # Save to file
    result_df.to_csv(output_file, index=False)

    # Log summary
    total_features = len(features_to_lag) * n_lags
    original_rows = len(df)
    rows_with_complete_data = len(result_df.dropna())

    logger.info(f"Created {total_features} lag features")
    logger.info(f"Output saved to {output_file}")
    logger.info(f"Dataset shape: {result_df.shape}")
    logger.info(f"Rows with complete data (no NaN): {rows_with_complete_data}/{original_rows}")

    return result_df

def get_lag_info(df: pd.DataFrame, n_lags: int, features_to_lag: List[str]) -> Dict[str, Any]:
    """Get summary information about the lagged dataset."""
    if df.empty:
        return {"error": "Dataset is empty"}

    # Identify lag columns
    lag_columns = []
    for feature in features_to_lag:
        for lag in range(1, n_lags + 1):
            lag_columns.append(f"{feature}_lag_{lag}")

    # Calculate statistics
    total_rows = len(df)
    complete_rows = len(df.dropna())
    missing_data_pct = ((total_rows - complete_rows) / total_rows) * 100

    # Missing data by stock (first n_lags rows per stock will have NaN)
    stocks_count = df['ticker'].nunique()
    expected_missing = stocks_count * n_lags

    info = {
        "total_rows": total_rows,
        "complete_rows": complete_rows,
        "missing_data_percentage": round(missing_data_pct, 2),
        "expected_missing_rows": expected_missing,
        "lag_features_created": len(lag_columns),
        "lag_columns": lag_columns,
        "stocks_processed": stocks_count,
        "date_range": {
            "start": df['Date'].min().strftime('%Y-%m-%d'),
            "end": df['Date'].max().strftime('%Y-%m-%d')
        },
        "features_lagged": features_to_lag,
        "max_lag_steps": n_lags
    }

    return info

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create lagged features for time series analysis')
    parser.add_argument('--input-file', type=str, default='greenfield/dataset/stock_dataset.csv',
                       help='Path to input CSV file (default: stock_dataset.csv)')
    parser.add_argument('--output-file', type=str, default='greenfield/dataset/stock_dataset_with_lags.csv',
                       help='Path to output CSV file (default: stock_dataset_with_lags.csv)')
    parser.add_argument('--n-lags', type=int, default=3,
                       help='Number of lag steps to create (default: 3)')
    parser.add_argument('--features', type=str, nargs='*',
                       default=['weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility'],
                       help='Features to create lags for (default: all return features)')

    args = parser.parse_args()

    print(f"Creating {args.n_lags} lagged features for normalized returns dataset...")

    # Create dataset with specified lag steps
    lagged_dataset = create_lag_features(
        input_file=args.input_file,
        n_lags=args.n_lags,
        features_to_lag=args.features,
        output_file=args.output_file
    )

    # Display summary information
    lag_info = get_lag_info(
        lagged_dataset,
        n_lags=args.n_lags,
        features_to_lag=args.features
    )

    print("\nLag Features Summary:")
    print(f"Total rows: {lag_info['total_rows']}")
    print(f"Complete rows (no missing data): {lag_info['complete_rows']}")
    print(f"Missing data percentage: {lag_info['missing_data_percentage']}%")
    print(f"Lag features created: {lag_info['lag_features_created']}")
    print(f"Stocks processed: {lag_info['stocks_processed']}")
    print(f"Date range: {lag_info['date_range']['start']} to {lag_info['date_range']['end']}")

    # Show sample of lagged data
    print(f"\nSample of dataset with lag features:")
    sample_cols = ['ticker', 'Date', 'weekly_return', 'weekly_return_lag_1', 'weekly_return_lag_2', 'volume_change', 'volume_change_lag_1']
    available_cols = [col for col in sample_cols if col in lagged_dataset.columns]
    print(lagged_dataset[available_cols].head(10))