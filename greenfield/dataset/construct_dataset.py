import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_stock_tickers(file_path: str = 'training_stocks.txt') -> List[str]:
    """Load stock tickers from comma-separated text file."""
    try:
        # If relative path, make it relative to this script's directory
        if not os.path.isabs(file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, file_path)

        with open(file_path, 'r') as f:
            content = f.read().strip()
            tickers = [ticker.strip().upper() for ticker in content.split(',') if ticker.strip()]
        logger.info(f"Loaded {len(tickers)} stock tickers from {file_path}")
        return tickers
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        raise

def fetch_stock_data(ticker: str, start_Date: str, end_Date: str) -> Optional[pd.DataFrame]:
    """Fetch daily stock data for a single ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_Date, end=end_Date, interval='1d')
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None
            
        # Clean column names and reset index
        data.columns = [col.lower() for col in data.columns]
        data = data.reset_index()
        data['ticker'] = ticker
        
        logger.info(f"Fetched {len(data)} days of data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def aggregate_to_weekly(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Convert daily stock data to weekly aggregation with returns."""
    if daily_data is None or daily_data.empty:
        return pd.DataFrame()

    # Ensure Date column is Datetime
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])

    # Set Date as index for resampling
    daily_data_indexed = daily_data.set_index('Date')

    # Calculate daily returns for volatility calculation
    daily_data_indexed['daily_return'] = daily_data_indexed['close'].pct_change()

    # Weekly aggregation (W-FRI = weekly ending on Friday)
    weekly_agg = daily_data_indexed.resample('W-FRI').agg({
        'open': 'first',      # First day's opening price
        'high': 'max',        # Maximum price during week
        'low': 'min',         # Minimum price during week
        'volume': 'sum',      # Sum of daily volumes
        'close': 'last',      # Last day's closing price
        'daily_return': lambda x: x.std(),  # Standard deviation of daily returns
        'ticker': 'first'     # Keep ticker symbol
    })

    # Calculate returns instead of using raw prices
    weekly_agg['weekly_return'] = (weekly_agg['close'] / weekly_agg['open']) - 1
    weekly_agg['high_return'] = (weekly_agg['high'] / weekly_agg['open']) - 1
    weekly_agg['low_return'] = (weekly_agg['low'] / weekly_agg['open']) - 1

    # Calculate volume change (percentage change from previous week)
    weekly_agg['volume_change'] = weekly_agg['volume'].pct_change()

    # Rename volatility column
    weekly_agg = weekly_agg.rename(columns={'daily_return': 'volatility'})

    # Reset index to get Date as column
    weekly_agg = weekly_agg.reset_index()

    # Remove rows with NaN values (first week may have NaN volatility)
    weekly_agg = weekly_agg.dropna()

    # Select return-based columns: [ticker, Date, weekly_return, high_return, low_return, volume_change, volatility]
    weekly_agg = weekly_agg[['ticker', 'Date', 'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']]

    return weekly_agg

def normalize_features_per_stock(df: pd.DataFrame, features_to_normalize: List[str] = None) -> pd.DataFrame:
    """
    Normalize features per stock using z-score normalization.
    Note: weekly_return is excluded from normalization to preserve dollar return values.

    Args:
        df: DataFrame with stock data
        features_to_normalize: List of column names to normalize

    Returns:
        DataFrame with normalized features (original columns replaced)
    """
    if features_to_normalize is None:
        # Exclude weekly_return from normalization to preserve dollar returns
        features_to_normalize = ['high_return', 'low_return', 'volume_change', 'volatility']

    normalized_df = df.copy()

    for feature in features_to_normalize:
        if feature in df.columns:
            # Z-score normalization per stock: (x - mean) / std
            normalized_df[feature] = df.groupby('ticker')[feature].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            logger.debug(f"Normalized {feature} per stock")
        else:
            logger.warning(f"Feature {feature} not found in dataset")

    return normalized_df

def compile_stock_dataset(tickers_file: str = 'training_stocks.txt',
                         years: int = 3,
                         output_file: str = 'stock_dataset.csv',
                         normalize: bool = True) -> pd.DataFrame:
    """
    Compile weekly stock dataset for multiple tickers over specified years.

    Args:
        tickers_file: Path to file containing comma-separated stock tickers
        years: Number of years of historical data to fetch
        output_file: Path to save the compiled dataset
        normalize: Whether to normalize features per stock (default: True)

    Returns:
        pandas.DataFrame: Compiled weekly stock dataset with returns and normalization
    """
    # Calculate Date range
    end_Date = datetime.now()
    start_Date = end_Date - timedelta(days=years * 365)
    
    start_str = start_Date.strftime('%Y-%m-%d')
    end_str = end_Date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_str} to {end_str}")
    
    # Load tickers
    tickers = load_stock_tickers(tickers_file)
    
    # Compile dataset
    all_weekly_data = []
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        
        # Fetch daily data
        daily_data = fetch_stock_data(ticker, start_str, end_str)
        
        if daily_data is not None:
            # Convert to weekly
            weekly_data = aggregate_to_weekly(daily_data)
            
            if not weekly_data.empty:
                all_weekly_data.append(weekly_data)
                logger.info(f"Added {len(weekly_data)} weekly records for {ticker}")
            else:
                logger.warning(f"No weekly data generated for {ticker}")
        
        # Small delay to be respectful to API
        import time
        time.sleep(0.1)
    
    if not all_weekly_data:
        logger.error("No data collected for any tickers")
        return pd.DataFrame()
    
    # Combine all data
    final_dataset = pd.concat(all_weekly_data, ignore_index=True)

    # Sort by ticker and Date
    final_dataset = final_dataset.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # Apply normalization if requested
    if normalize:
        logger.info("Applying per-stock normalization...")
        final_dataset = normalize_features_per_stock(final_dataset)
        logger.info("Normalization completed")

    # Save to CSV
    final_dataset.to_csv(output_file, index=False)
    dataset_type = "normalized returns" if normalize else "returns"
    logger.info(f"Dataset ({dataset_type}) saved to {output_file} with {len(final_dataset)} weekly records")
    logger.info(f"Dataset covers {final_dataset['ticker'].nunique()} unique stocks")
    logger.info(f"Date range: {final_dataset['Date'].min()} to {final_dataset['Date'].max()}")

    return final_dataset

def get_dataset_info(dataset: pd.DataFrame) -> dict:
    """Get summary information about the compiled dataset."""
    if dataset.empty:
        return {"error": "Dataset is empty"}
    
    info = {
        "total_records": len(dataset),
        "unique_stocks": dataset['ticker'].nunique(),
        "stock_list": sorted(dataset['ticker'].unique().tolist()),
        "Date_range": {
            "start": dataset['Date'].min().strftime('%Y-%m-%d'),
            "end": dataset['Date'].max().strftime('%Y-%m-%d')
        },
        "columns": dataset.columns.tolist(),
        "sample_data": dataset.head().to_dict('records')
    }
    
    return info

if __name__ == "__main__":
    # Example usage with normalized returns
    dataset = compile_stock_dataset(
        tickers_file='training_stocks.txt',
        years=3,
        output_file='greenfield/dataset/stock_dataset.csv',
        normalize=True
    )

    if not dataset.empty:
        info = get_dataset_info(dataset)
        print("\nDataset Summary:")
        print(f"Total records: {info['total_records']}")
        print(f"Unique stocks: {info['unique_stocks']}")
        print(f"Date range: {info['Date_range']['start']} to {info['Date_range']['end']}")
        print(f"Columns: {info['columns']}")

        # Show sample of normalized return data
        print(f"\nSample of normalized return data:")
        sample_cols = ['ticker', 'Date', 'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']
        print(dataset[sample_cols].head(10))

        # Show normalization statistics per stock (first few stocks)
        print(f"\nNormalization check (mean ~= 0, std ~= 1 per stock for normalized features):")
        normalized_cols = ['high_return', 'low_return', 'volume_change', 'volatility']
        unnormalized_cols = ['weekly_return']
        sample_tickers = dataset['ticker'].unique()[:3]
        for ticker in sample_tickers:
            ticker_data = dataset[dataset['ticker'] == ticker]
            print(f"{ticker} normalized: means = {ticker_data[normalized_cols].mean().round(3).to_dict()}")
            print(f"{ticker} normalized: stds  = {ticker_data[normalized_cols].std().round(3).to_dict()}")
            print(f"{ticker} weekly_return (unnormalized): mean = {ticker_data['weekly_return'].mean():.6f}, std = {ticker_data['weekly_return'].std():.6f}")