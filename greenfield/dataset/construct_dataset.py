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
    
    try:
        # if relative path, make it relative to this script's directory
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
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_Date, end=end_Date, interval='1d')
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None
            
        # clean column names and reset index
        data.columns = [col.lower() for col in data.columns]
        data = data.reset_index()
        data['ticker'] = ticker
        
        logger.info(f"Fetched {len(data)} days of data for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_market_data(start_date: str, end_date: str) -> dict:
    
    market_tickers = {
        'SPY': 'S&P 500',
        '^VIX': 'VIX',
        'XLF': 'Financials',
        'XLK': 'Technology',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate'
    }

    market_data = {}
    for ticker, name in market_tickers.items():
        logger.info(f"Fetching market data for {ticker} ({name})")
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            market_data[ticker] = df
        else:
            logger.warning(f"Failed to fetch {ticker}")

    return market_data

def aggregate_to_weekly(daily_data: pd.DataFrame) -> pd.DataFrame:
    
    if daily_data is None or daily_data.empty:
        return pd.DataFrame()

    # ensure date column is datetime
    daily_data['Date'] = pd.to_datetime(daily_data['Date'])

    # set date as index for resampling
    daily_data_indexed = daily_data.set_index('Date')

    # calculate daily returns for volatility calculation
    daily_data_indexed['daily_return'] = daily_data_indexed['close'].pct_change()

    # weekly aggregation (w-fri = weekly ending on friday)
    weekly_agg = daily_data_indexed.resample('W-FRI').agg({
        'open': 'first',      # first day's opening price
        'high': 'max',        # maximum price during week
        'low': 'min',         # minimum price during week
        'volume': 'sum',      # sum of daily volumes
        'close': 'last',      # last day's closing price
        'daily_return': lambda x: x.std(),  # standard deviation of daily returns
        'ticker': 'first'     # keep ticker symbol
    })

    # calculate returns instead of using raw prices
    weekly_agg['weekly_return'] = (weekly_agg['close'] / weekly_agg['open']) - 1
    weekly_agg['high_return'] = (weekly_agg['high'] / weekly_agg['open']) - 1
    weekly_agg['low_return'] = (weekly_agg['low'] / weekly_agg['open']) - 1

    # calculate volume change (percentage change from previous week)
    # handle indices like vix that have zero volume by filling nan with 0
    weekly_agg['volume_change'] = weekly_agg['volume'].pct_change()
    weekly_agg['volume_change'] = weekly_agg['volume_change'].fillna(0)

    # rename volatility column
    weekly_agg = weekly_agg.rename(columns={'daily_return': 'volatility'})

    # reset index to get date as column
    weekly_agg = weekly_agg.reset_index()

    # remove rows with nan values in volatility only (first row may have nan)
    # don't drop rows with nan in volume_change (already filled)
    weekly_agg = weekly_agg.dropna(subset=['volatility', 'weekly_return', 'high_return', 'low_return'])

    # select return-based columns: [ticker, date, weekly_return, high_return, low_return, volume_change, volatility]
    weekly_agg = weekly_agg[['ticker', 'Date', 'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']]

    return weekly_agg

def aggregate_market_data(market_data: dict) -> pd.DataFrame:
    
    market_dfs = []

    for ticker, daily_df in market_data.items():
        weekly = aggregate_to_weekly(daily_df)

        if weekly.empty:
            logger.warning(f"No weekly data for {ticker}")
            continue

        # rename columns to be market-specific (keep only return and volatility)
        weekly = weekly.rename(columns={
            'weekly_return': f'{ticker}_return',
            'volatility': f'{ticker}_volatility'
        })

        # keep only date and relevant columns
        weekly = weekly[['Date', f'{ticker}_return', f'{ticker}_volatility']]
        market_dfs.append(weekly)

    if not market_dfs:
        logger.error("No market data aggregated")
        return pd.DataFrame()

    # merge all market data on date
    market_df = market_dfs[0]
    for df in market_dfs[1:]:
        market_df = market_df.merge(df, on='Date', how='outer')

    # sort by date
    market_df = market_df.sort_values('Date').reset_index(drop=True)

    logger.info(f"Aggregated {len(market_dfs)} market indices to weekly format")
    return market_df

def normalize_features_per_stock(df: pd.DataFrame, features_to_normalize: List[str] = None) -> pd.DataFrame:
    
    if features_to_normalize is None:
        # exclude weekly_return from normalization to preserve dollar returns
        features_to_normalize = ['high_return', 'low_return', 'volume_change', 'volatility']

    normalized_df = df.copy()

    for feature in features_to_normalize:
        if feature in df.columns:
            # z-score normalization per stock: (x - mean) / std
            normalized_df[feature] = df.groupby('ticker')[feature].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            logger.debug(f"Normalized {feature} per stock")
        else:
            logger.warning(f"Feature {feature} not found in dataset")

    return normalized_df

def merge_with_market_data(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("Merging market data with stock data...")

    # ensure date columns are datetime
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    market_df['Date'] = pd.to_datetime(market_df['Date'])

    # merge on date (left join to keep all stock data)
    merged = stock_df.merge(market_df, on='Date', how='left')

    # fill any missing market data with forward fill (market closed days)
    market_columns = [col for col in merged.columns if col not in stock_df.columns]
    merged[market_columns] = merged[market_columns].ffill()

    logger.info(f"Merged dataset shape: {merged.shape}")
    logger.info(f"Added {len(market_columns)} market features")

    return merged

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("Calculating momentum features...")

    momentum_features = []

    for ticker in df['ticker'].unique():
        stock_data = df[df['ticker'] == ticker].copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)

        # calculate cumulative returns for momentum
        stock_data['cum_return'] = (1 + stock_data['weekly_return']).cumprod()

        # 4-week momentum (1 month)
        stock_data['momentum_4w'] = stock_data['cum_return'] / stock_data['cum_return'].shift(4) - 1

        # 12-week momentum (3 months)
        stock_data['momentum_12w'] = stock_data['cum_return'] / stock_data['cum_return'].shift(12) - 1

        # 52-week momentum (1 year)
        stock_data['momentum_52w'] = stock_data['cum_return'] / stock_data['cum_return'].shift(52) - 1

        # price relative to 52-week high
        stock_data['close_rolling'] = stock_data['weekly_return'].cumsum()  # proxy for log price
        stock_data['high_52w'] = stock_data['close_rolling'].rolling(window=52, min_periods=1).max()
        stock_data['price_to_52w_high'] = stock_data['close_rolling'] / stock_data['high_52w']

        # fill nan momentum values with shorter-period fallbacks
        # for weeks 1-4: momentum_4w = 0 (reasonable: no prior data)
        # for weeks 5-12: momentum_12w falls back to momentum_4w
        # for weeks 13-52: momentum_52w falls back to momentum_12w
        stock_data['momentum_4w'] = stock_data['momentum_4w'].fillna(0)
        stock_data['momentum_12w'] = stock_data['momentum_12w'].fillna(stock_data['momentum_4w'])
        stock_data['momentum_52w'] = stock_data['momentum_52w'].fillna(stock_data['momentum_12w'])

        # drop intermediate columns
        stock_data = stock_data.drop(columns=['cum_return', 'close_rolling', 'high_52w'])

        momentum_features.append(stock_data)

    result = pd.concat(momentum_features, ignore_index=True)

    logger.info(f"Added momentum features: momentum_4w, momentum_12w, momentum_52w, price_to_52w_high")

    return result

def calculate_sharpe_ratio(df: pd.DataFrame, window: int = 12, risk_free_rate: float = 0.0) -> pd.DataFrame:
    
    logger.info(f"Calculating {window}-week rolling Sharpe ratio...")

    sharpe_features = []

    for ticker in df['ticker'].unique():
        stock_data = df[df['ticker'] == ticker].copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)

        # calculate rolling mean and std of weekly returns
        rolling_mean = stock_data['weekly_return'].rolling(window=window, min_periods=window).mean()
        rolling_std = stock_data['weekly_return'].rolling(window=window, min_periods=window).std()

        # sharpe ratio: (mean_return - risk_free_rate) / std_return
        # avoid division by zero: if std is 0 or nan, set sharpe to 0
        stock_data['sharpe_ratio_3m'] = np.where(
            (rolling_std > 0) & (~rolling_std.isna()),
            (rolling_mean - risk_free_rate) / rolling_std,
            0.0
        )

        # fill nan values (first 12 weeks) with 0
        stock_data['sharpe_ratio_3m'] = stock_data['sharpe_ratio_3m'].fillna(0.0)

        sharpe_features.append(stock_data)

    result = pd.concat(sharpe_features, ignore_index=True)

    logger.info(f"Added sharpe_ratio_3m feature")

    return result

def compile_stock_dataset(tickers_file: str = 'training_stocks.txt',
                         years: int = 3,
                         output_file: str = 'dataset/stock_dataset.csv',
                         normalize: bool = True) -> pd.DataFrame:
    
    # calculate date range
    end_Date = datetime.now()
    start_Date = end_Date - timedelta(days=years * 365)
    
    start_str = start_Date.strftime('%Y-%m-%d')
    end_str = end_Date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_str} to {end_str}")
    
    # load tickers
    tickers = load_stock_tickers(tickers_file)
    
    # compile dataset
    all_weekly_data = []
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        
        # fetch daily data
        daily_data = fetch_stock_data(ticker, start_str, end_str)
        
        if daily_data is not None:
            # convert to weekly
            weekly_data = aggregate_to_weekly(daily_data)
            
            if not weekly_data.empty:
                all_weekly_data.append(weekly_data)
                logger.info(f"Added {len(weekly_data)} weekly records for {ticker}")
            else:
                logger.warning(f"No weekly data generated for {ticker}")
        
        # small delay to be respectful to api
        import time
        time.sleep(0.1)
    
    if not all_weekly_data:
        logger.error("No data collected for any tickers")
        return pd.DataFrame()
    
    # combine all data
    final_dataset = pd.concat(all_weekly_data, ignore_index=True)

    # sort by ticker and date
    final_dataset = final_dataset.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # fetch and merge market data
    logger.info("Fetching market indices...")
    market_data = fetch_market_data(start_str, end_str)

    if market_data:
        market_weekly = aggregate_market_data(market_data)

        # save market data separately for reference
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            market_output_file = os.path.join(output_dir, 'market_data.csv')
        else:
            market_output_file = 'market_data.csv'

        market_weekly.to_csv(market_output_file, index=False)
        logger.info(f"Market data saved to {market_output_file}")

        # merge with stock data
        final_dataset = merge_with_market_data(final_dataset, market_weekly)
    else:
        logger.warning("No market data fetched, skipping market features")

    # calculate momentum features
    final_dataset = calculate_momentum_features(final_dataset)

    # calculate sharpe ratio features
    final_dataset = calculate_sharpe_ratio(final_dataset)

    # apply normalization if requested
    # disabled: normalization causes scale mismatch between lag features and target
    # all features are already in percentage/return space which is comparable across stocks
    if normalize:
        logger.warning("Normalization is disabled - return-based features are already comparable")
        # final_dataset = normalize_features_per_stock(final_dataset)
        # logger.info("normalization completed")

    # save to csv
    final_dataset.to_csv(output_file, index=False)
    dataset_type = "normalized returns" if normalize else "returns"
    logger.info(f"Dataset ({dataset_type}) saved to {output_file} with {len(final_dataset)} weekly records")
    logger.info(f"Dataset covers {final_dataset['ticker'].nunique()} unique stocks")
    logger.info(f"Date range: {final_dataset['Date'].min()} to {final_dataset['Date'].max()}")
    logger.info(f"Total features: {len(final_dataset.columns)}")

    return final_dataset

def get_dataset_info(dataset: pd.DataFrame) -> dict:
    
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
    # build dataset without normalization (returns are already comparable percentages)
    dataset = compile_stock_dataset(
        tickers_file='training_stocks.txt',
        years=3,
        output_file='dataset/stock_dataset.csv',
        normalize=False
    )

    if not dataset.empty:
        info = get_dataset_info(dataset)
        print("\nDataset Summary:")
        print(f"Total records: {info['total_records']}")
        print(f"Unique stocks: {info['unique_stocks']}")
        print(f"Date range: {info['Date_range']['start']} to {info['Date_range']['end']}")
        print(f"Columns: {info['columns']}")

        # show sample of normalized return data
        print(f"\nSample of normalized return data:")
        sample_cols = ['ticker', 'Date', 'weekly_return', 'high_return', 'low_return', 'volume_change', 'volatility']
        print(dataset[sample_cols].head(10))

        # show normalization statistics per stock (first few stocks)
        print(f"\nNormalization check (mean ~= 0, std ~= 1 per stock for normalized features):")
        normalized_cols = ['high_return', 'low_return', 'volume_change', 'volatility']
        unnormalized_cols = ['weekly_return']
        sample_tickers = dataset['ticker'].unique()[:3]
        for ticker in sample_tickers:
            ticker_data = dataset[dataset['ticker'] == ticker]
            print(f"{ticker} normalized: means = {ticker_data[normalized_cols].mean().round(3).to_dict()}")
            print(f"{ticker} normalized: stds  = {ticker_data[normalized_cols].std().round(3).to_dict()}")
            print(f"{ticker} weekly_return (unnormalized): mean = {ticker_data['weekly_return'].mean():.6f}, std = {ticker_data['weekly_return'].std():.6f}")
