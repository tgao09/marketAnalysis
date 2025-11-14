import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str = 'dataset/stock_dataset.csv') -> pd.DataFrame:
    
    try:
        df = pd.read_csv(file_path)

        # clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # validate required columns (updated for normalized returns dataset)
        required_columns = ['ticker', 'Date', 'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # sort by ticker and date to ensure proper chronological order
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
    
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")

    if n_lags > 50:
        logger.warning(f"Large number of lags ({n_lags}) may create a very wide dataset")

    # check if features_to_lag exist in the dataset
    missing_features = [feature for feature in features_to_lag if feature not in df_columns]
    if missing_features:
        raise ValueError(f"Features not found in dataset: {missing_features}")

    # warn about non-numeric features
    reserved_columns = ['ticker', 'Date']
    invalid_features = [feature for feature in features_to_lag if feature in reserved_columns]
    if invalid_features:
        raise ValueError(f"Cannot lag reserved columns: {invalid_features}")

def create_lag_features(input_file: str = 'dataset/stock_dataset.csv',
                       n_lags: int = 3,
                       features_to_lag: List[str] = None,
                       output_file: str = 'dataset/stock_dataset_with_lags.csv') -> pd.DataFrame:
    
    # load the dataset first
    df = load_dataset(input_file)

    # set default features to lag if not specified (updated for market + momentum features)
    # note: sharpe_ratio_3m is excluded - it's a stock quality metric that should not be lagged
    if features_to_lag is None:
        features_to_lag = [
            'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility',
            # market features
            'SPY_return', '^VIX_return', 'XLF_return', 'XLK_return', 'XLE_return',
            'XLV_return', 'XLI_return', 'XLY_return', 'XLP_return', 'XLU_return', 'XLRE_return',
            'SPY_volatility', '^VIX_volatility', 'XLF_volatility', 'XLK_volatility', 'XLE_volatility',
            'XLV_volatility', 'XLI_volatility', 'XLY_volatility', 'XLP_volatility', 'XLU_volatility', 'XLRE_volatility',
            # momentum features
            'momentum_4w', 'momentum_12w', 'momentum_52w', 'price_to_52w_high'
            # sharpe_ratio_3m is intentionally excluded - passed through without lagging
        ]

        # filter to only include features that exist in the dataset
        features_to_lag = [f for f in features_to_lag if f in df.columns]
        logger.info(f"Auto-detected {len(features_to_lag)} features to lag from dataset")

        # check if sharpe_ratio_3m exists and log it's being passed through
        if 'sharpe_ratio_3m' in df.columns:
            logger.info("sharpe_ratio_3m found - will be passed through without lagging")

    # validate parameters
    validate_lag_parameters(n_lags, features_to_lag, df.columns.tolist())

    logger.info(f"Creating {n_lags} lag features for: {features_to_lag}")

    # create lagged features
    lagged_dfs = []

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()

        # create lag features for this ticker
        for feature in features_to_lag:
            for lag in range(1, n_lags + 1):
                lag_column_name = f"{feature}_lag_{lag}"
                ticker_data[lag_column_name] = ticker_data[feature].shift(lag)

        # fill nan values in lagged momentum features (first n_lags rows will be nan due to shifting)
        # base momentum features are already filled in construct_dataset.py
        lagged_momentum_cols = [col for col in ticker_data.columns
                               if '_lag_' in col and ('momentum' in col or 'price_to_52w' in col)]
        if lagged_momentum_cols:
            # forward fill: use the first available value for early lags
            ticker_data[lagged_momentum_cols] = ticker_data[lagged_momentum_cols].bfill()

        lagged_dfs.append(ticker_data)
        logger.debug(f"Processed lags for {ticker}")

    # combine all ticker data
    result_df = pd.concat(lagged_dfs, ignore_index=True)

    # sort by ticker and date
    result_df = result_df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # save to file
    result_df.to_csv(output_file, index=False)

    # log summary
    total_features = len(features_to_lag) * n_lags
    original_rows = len(df)
    rows_with_complete_data = len(result_df.dropna())

    logger.info(f"Created {total_features} lag features")
    logger.info(f"Output saved to {output_file}")
    logger.info(f"Dataset shape: {result_df.shape}")
    logger.info(f"Rows with complete data (no NaN): {rows_with_complete_data}/{original_rows}")

    return result_df

def get_lag_info(df: pd.DataFrame, n_lags: int, features_to_lag: List[str]) -> Dict[str, Any]:
    
    if df.empty:
        return {"error": "Dataset is empty"}

    # identify lag columns
    lag_columns = []
    for feature in features_to_lag:
        for lag in range(1, n_lags + 1):
            lag_columns.append(f"{feature}_lag_{lag}")

    # calculate statistics
    total_rows = len(df)
    complete_rows = len(df.dropna())
    missing_data_pct = ((total_rows - complete_rows) / total_rows) * 100

    # missing data by stock (first n_lags rows per stock will have nan)
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
    parser.add_argument('--input-file', type=str, default='dataset/stock_dataset.csv',
                       help='Path to input CSV file (default: dataset/stock_dataset.csv)')
    parser.add_argument('--output-file', type=str, default='dataset/stock_dataset_with_lags.csv',
                       help='Path to output CSV file (default: dataset/stock_dataset_with_lags.csv)')
    parser.add_argument('--n-lags', type=int, default=3,
                       help='Number of lag steps to create (default: 3)')
    parser.add_argument('--features', type=str, nargs='*',
                       default=None,  # use none to trigger default list with market + momentum features
                       help='Features to create lags for (default: all features including market and momentum)')

    args = parser.parse_args()

    print(f"Creating {args.n_lags} lagged features for normalized returns dataset...")

    # create dataset with specified lag steps
    lagged_dataset = create_lag_features(
        input_file=args.input_file,
        n_lags=args.n_lags,
        features_to_lag=args.features,
        output_file=args.output_file
    )

    # get the actual features that were lagged (from the dataset)
    actual_features = [col.replace('_lag_1', '').replace('_lag_2', '').replace('_lag_3', '')
                      for col in lagged_dataset.columns if '_lag_' in col]
    actual_features = list(set(actual_features))  # remove duplicates

    # display summary information
    lag_info = get_lag_info(
        lagged_dataset,
        n_lags=args.n_lags,
        features_to_lag=actual_features
    )

    print("\nLag Features Summary:")
    print(f"Total rows: {lag_info['total_rows']}")
    print(f"Complete rows (no missing data): {lag_info['complete_rows']}")
    print(f"Missing data percentage: {lag_info['missing_data_percentage']}%")
    print(f"Lag features created: {lag_info['lag_features_created']}")
    print(f"Stocks processed: {lag_info['stocks_processed']}")
    print(f"Date range: {lag_info['date_range']['start']} to {lag_info['date_range']['end']}")

    # show sample of lagged data
    print(f"\nSample of dataset with lag features:")
    sample_cols = ['ticker', 'Date', 'weekly_return', 'weekly_return_lag_1', 'weekly_return_lag_2', 'volume_change', 'volume_change_lag_1']
    available_cols = [col for col in sample_cols if col in lagged_dataset.columns]
    print(lagged_dataset[available_cols].head(10))