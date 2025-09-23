#!/usr/bin/env python3
"""
Stock Prediction Visualization Script

This script loads ARIMAX prediction results and creates visualizations
showing predicted stock prices. It prompts the user to select which stocks
to analyze and only fetches price data for the selected tickers.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import glob
import yfinance as yf


def get_all_prediction_files(predictions_dir="./arimaxresults", file_type="future"):
    """
    Find all prediction files in the arimaxresults directory

    Args:
        predictions_dir: Directory to search for files
        file_type: 'future' for future_forecasts_*.csv, 'backtest' for backtest_results_*.csv
    """
    if file_type == "future":
        pattern = os.path.join(predictions_dir, "future_forecasts_*.csv")
    elif file_type == "backtest":
        pattern = os.path.join(predictions_dir, "backtest_results_*.csv")
    else:
        # Try both patterns and return all files
        future_pattern = os.path.join(predictions_dir, "future_forecasts_*.csv")
        backtest_pattern = os.path.join(predictions_dir, "backtest_results_*.csv")
        old_pattern = os.path.join(predictions_dir, "predictions_*.csv")

        all_files = glob.glob(future_pattern) + glob.glob(backtest_pattern) + glob.glob(old_pattern)
        return sorted(all_files, key=os.path.getctime, reverse=True)  # Most recent first

    files = glob.glob(pattern)
    return sorted(files, key=os.path.getctime, reverse=True)  # Most recent first


def get_latest_prediction_file(predictions_dir="./arimaxresults", file_type="future"):
    """
    Find the most recent prediction file in the arimaxresults directory
    Kept for backward compatibility
    """
    files = get_all_prediction_files(predictions_dir, file_type)
    return files[0] if files else None


def get_starting_price(ticker, target_date):
    """
    Get the appropriate starting price for predictions.
    For future dates: use current price
    For historical dates: use the week's open price
    """
    try:
        print(f"Fetching starting price for {ticker}...")
        stock = yf.Ticker(ticker)

        # Check if target_date is in the future
        today = datetime.now().date()
        if hasattr(target_date, 'date'):
            target_date_only = target_date.date()
        else:
            target_date_only = pd.to_datetime(target_date).date()

        if target_date_only > today:
            # Future prediction - get current price
            print(f"Target date {target_date_only} is in the future, using current price...")
            # Get recent data (last 5 days)
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                print(f"Using current price for {ticker}: ${current_price:.2f}")
                return current_price
        else:
            # Historical prediction - get week open price
            print(f"Target date {target_date_only} is historical, using week open price...")

        # Get data from 2 weeks before target_date to ensure we have enough data
        start_date = target_date - timedelta(days=14)
        end_date = target_date + timedelta(days=7)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        hist = hist.reset_index()

        # Convert Date column to timezone-naive datetime to match target_date
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)

        # Ensure target_date is also timezone-naive
        if hasattr(target_date, 'tz') and target_date.tz is not None:
            target_date = target_date.tz_localize(None)
        elif hasattr(target_date, 'tzinfo') and target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        # Find the Monday of the target week (assuming week starts on Monday)
        target_weekday = target_date.weekday()  # 0=Monday, 6=Sunday
        monday_of_target_week = target_date - timedelta(days=target_weekday)

        # Look for the first trading day of the target week (Monday or later)
        current_week_data = hist[hist['Date'] >= monday_of_target_week]

        if not current_week_data.empty:
            # Use the first trading day's open price of the target week
            price = current_week_data.iloc[0]['Open']
            print(f"Using week open price for {ticker}: ${price:.2f}")
            return price
        else:
            # Fallback: use last week's close (last available close before target week)
            previous_data = hist[hist['Date'] < monday_of_target_week]
            if not previous_data.empty:
                price = previous_data.iloc[-1]['Close']
                print(f"Using previous week close for {ticker}: ${price:.2f}")
                return price

    except Exception as e:
        print(f"Error fetching starting price for {ticker}: {e}")

    return None


def convert_returns_to_prices(ticker_df, ticker):
    """
    Convert predicted returns to actual stock prices for a single ticker
    """
    # Handle both old and new date column names
    date_col = 'future_date' if 'future_date' in ticker_df.columns else 'prediction_date'
    first_prediction_date = ticker_df[date_col].iloc[0]

    # Get starting price (current price for future dates, week open for historical)
    starting_price = get_starting_price(ticker, first_prediction_date)
    if starting_price is None:
        print(f"Warning: Could not fetch starting price for {ticker}, using $100 as default")
        starting_price = 100.0

    # Convert weekly returns to cumulative stock prices
    # Assumption: week close = next week's open
    ticker_df = ticker_df.copy()
    ticker_df['stock_price'] = 0.0

    for i in range(len(ticker_df)):
        # Check if returns are in decimal (new format) or percentage (old format)
        predicted_return = ticker_df.iloc[i]['predicted_return']
        # If the absolute value is small (< 1), assume it's decimal format
        if abs(predicted_return) < 1:
            return_multiplier = predicted_return  # Already in decimal form
        else:
            return_multiplier = predicted_return / 100  # Convert percentage to decimal

        if i == 0:
            # First prediction: apply return to week's open price
            week_close = starting_price * (1 + return_multiplier)
            ticker_df.iloc[i, ticker_df.columns.get_loc('stock_price')] = week_close
        else:
            # Subsequent predictions: previous week's close = this week's open
            prev_week_close = ticker_df.iloc[i-1]['stock_price']
            current_week_close = prev_week_close * (1 + return_multiplier)
            ticker_df.iloc[i, ticker_df.columns.get_loc('stock_price')] = current_week_close

    return ticker_df


def get_historical_data(ticker, first_prediction_date, periods=28):
    """
    Get historical stock data for the specified ticker.
    For future predictions, get recent historical data up to today.
    For historical predictions, get data up to the prediction start date.
    """
    try:
        print(f"Fetching historical data for {ticker}...")
        stock = yf.Ticker(ticker)

        # Check if predictions are for future dates
        today = datetime.now().date()
        if hasattr(first_prediction_date, 'date'):
            pred_date = first_prediction_date.date()
        else:
            pred_date = pd.to_datetime(first_prediction_date).date()

        if pred_date > today:
            # Future predictions - get recent historical data up to today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods + 10)
            print(f"Getting recent historical data up to today...")
        else:
            # Historical predictions - get data up to prediction start
            end_date = first_prediction_date
            start_date = end_date - timedelta(days=periods + 10)
            print(f"Getting historical data up to {pred_date}...")

        hist = stock.history(start=start_date, end=end_date)

        if not hist.empty:
            hist = hist.reset_index()
            hist['ticker'] = ticker
            # Convert Date to timezone-naive to match prediction dates
            hist['date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            hist['stock_price'] = hist['Close']
            return hist[['ticker', 'date', 'stock_price']].tail(periods)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
    return pd.DataFrame()


def plot_single_ticker(df, ticker, show_historical=True):
    """
    Create visualization for a single ticker
    """
    # Filter data for this ticker
    ticker_df = df[df['ticker'] == ticker].copy()
    if ticker_df.empty:
        print(f"No prediction data found for {ticker}")
        return

    # Convert returns to prices
    ticker_df = convert_returns_to_prices(ticker_df, ticker)
    ticker_df['data_type'] = 'predicted'

    # Get historical data if requested
    historical_data = pd.DataFrame()
    if show_historical:
        first_prediction_date = ticker_df['date'].min()
        historical_data = get_historical_data(ticker, first_prediction_date, periods=28)
        if not historical_data.empty:
            historical_data['data_type'] = 'historical'

    # Combine data
    if not historical_data.empty:
        combined_data = pd.concat([historical_data, ticker_df[['ticker', 'date', 'stock_price', 'data_type']]],
                                 ignore_index=True)
    else:
        combined_data = ticker_df[['ticker', 'date', 'stock_price', 'data_type']].copy()

    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    # Create plot
    plt.figure(figsize=(15, 10))

    # Separate data types
    historical_subset = combined_data[combined_data['data_type'] == 'historical']
    predicted_subset = combined_data[combined_data['data_type'] == 'predicted']

    # Plot historical data
    if len(historical_subset) > 0:
        sns.lineplot(data=historical_subset, x='date', y='stock_price',
                    label=f'{ticker} Historical Price', linewidth=2.5,
                    marker='o', markersize=4, color='blue')

    # Plot predictions
    if len(predicted_subset) > 0:
        sns.lineplot(data=predicted_subset, x='date', y='stock_price',
                    label=f'{ticker} Predicted Price', linewidth=2.5,
                    linestyle='--', marker='s', markersize=5, color='red')

    # Add confidence intervals if available
    if 'ci_lower' in ticker_df.columns and 'ci_upper' in ticker_df.columns:
        # Convert confidence intervals from returns to prices
        date_col = 'future_date' if 'future_date' in ticker_df.columns else 'prediction_date'
        first_prediction_date = ticker_df[date_col].iloc[0]
        week_open_price = get_starting_price(ticker, first_prediction_date)

        if week_open_price is None:
            initial_price_ci = ticker_df['stock_price'].iloc[0] / (1 + ticker_df['predicted_return'].iloc[0] / 100)
        else:
            initial_price_ci = week_open_price

        # Calculate CI prices
        ticker_df_with_ci = ticker_df.copy()
        ticker_df_with_ci['ci_lower_price'] = 0.0
        ticker_df_with_ci['ci_upper_price'] = 0.0

        for i in range(len(ticker_df_with_ci)):
            if i == 0:
                base_price = initial_price_ci
            else:
                base_price_lower = ticker_df_with_ci.iloc[i-1]['ci_lower_price']
                base_price_upper = ticker_df_with_ci.iloc[i-1]['ci_upper_price']
                base_price = (base_price_lower + base_price_upper) / 2

            # Handle both decimal and percentage format for confidence intervals
            ci_lower = ticker_df_with_ci.iloc[i]['ci_lower']
            ci_upper = ticker_df_with_ci.iloc[i]['ci_upper']

            # Check if CI values are in decimal (new format) or percentage (old format)
            if abs(ci_lower) < 1 and abs(ci_upper) < 1:
                ci_lower_mult = ci_lower  # Already in decimal form
                ci_upper_mult = ci_upper  # Already in decimal form
            else:
                ci_lower_mult = ci_lower / 100  # Convert percentage to decimal
                ci_upper_mult = ci_upper / 100  # Convert percentage to decimal

            ticker_df_with_ci.iloc[i, ticker_df_with_ci.columns.get_loc('ci_lower_price')] = base_price * (1 + ci_lower_mult)
            ticker_df_with_ci.iloc[i, ticker_df_with_ci.columns.get_loc('ci_upper_price')] = base_price * (1 + ci_upper_mult)

        # Plot confidence intervals
        plt.fill_between(ticker_df_with_ci['date'],
                        ticker_df_with_ci['ci_lower_price'],
                        ticker_df_with_ci['ci_upper_price'],
                        alpha=0.2, color='red', label='95% Confidence Interval')

    # Determine plot title based on data type
    if 'future_date' in ticker_df.columns:
        plot_title = f'{ticker} Stock Price: Historical Data + Future Forecasts'
    elif 'historical_date' in ticker_df.columns:
        plot_title = f'{ticker} Stock Price: Historical Data + Backtest Results'
    else:
        plot_title = f'{ticker} Stock Price: Historical Data + ARIMAX Predictions'

    plt.title(plot_title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)

    # Add separation line
    if len(historical_subset) > 0 and len(predicted_subset) > 0:
        separation_date = historical_subset['date'].max()
        plt.axvline(x=separation_date, color='green', linestyle=':', alpha=0.8, linewidth=2)
        plt.text(separation_date, plt.ylim()[1]*0.95, 'Prediction Start',
                 rotation=90, ha='right', va='top', fontsize=10, color='green')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n=== {ticker} SUMMARY ===")
    if len(historical_subset) > 0:
        print(f"Historical data: {len(historical_subset)} days")
        print(f"Current price: ${historical_subset['stock_price'].iloc[-1]:.2f}")

    print(f"Predictions: {len(predicted_subset)} weeks")
    if len(predicted_subset) > 0:
        print(f"Final predicted price: ${predicted_subset['stock_price'].iloc[-1]:.2f}")

        if len(historical_subset) > 0:
            last_historical = historical_subset['stock_price'].iloc[-1]
            final_predicted = predicted_subset['stock_price'].iloc[-1]
            total_change_pct = ((final_predicted - last_historical) / last_historical) * 100
            print(f"Total predicted change: {total_change_pct:+.2f}% over {len(predicted_subset)} weeks")

    # Show model info
    if len(ticker_df) > 0:
        model_info = ticker_df.iloc[0]
        print(f"ARIMAX Model Order: {model_info['model_order']}")
        print(f"Model AIC: {model_info['model_aic']:.2f}")


def plot_multiple_tickers(df, tickers):
    """
    Create comparison plot for multiple tickers
    """
    plt.figure(figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))

    for i, ticker in enumerate(tickers):
        ticker_df = df[df['ticker'] == ticker].copy()
        if ticker_df.empty:
            print(f"No prediction data found for {ticker}")
            continue

        # Convert returns to prices (without printing individual progress)
        date_col = 'future_date' if 'future_date' in ticker_df.columns else 'prediction_date'
        first_prediction_date = ticker_df[date_col].iloc[0]
        week_open_price = get_starting_price(ticker, first_prediction_date)

        if week_open_price is None:
            starting_price = 100.0
        else:
            starting_price = week_open_price

        ticker_df['stock_price'] = 0.0
        for j in range(len(ticker_df)):
            # Handle both decimal and percentage format
            predicted_return = ticker_df.iloc[j]['predicted_return']
            if abs(predicted_return) < 1:
                return_multiplier = predicted_return  # Already in decimal form
            else:
                return_multiplier = predicted_return / 100  # Convert percentage to decimal

            if j == 0:
                week_close = starting_price * (1 + return_multiplier)
                ticker_df.iloc[j, ticker_df.columns.get_loc('stock_price')] = week_close
            else:
                prev_week_close = ticker_df.iloc[j-1]['stock_price']
                current_week_close = prev_week_close * (1 + return_multiplier)
                ticker_df.iloc[j, ticker_df.columns.get_loc('stock_price')] = current_week_close

        # Plot this ticker
        plt.plot(ticker_df['date'], ticker_df['stock_price'],
                label=ticker, linewidth=2, marker='o', markersize=3, color=colors[i])

    plt.title('Stock Price Predictions - Selected Tickers', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the prediction visualization
    """
    print("=== Stock Prediction Visualization ===\n")

    # Configuration
    predictions_dir = "arimaxresults"

    # Ask user for file type preference
    print("What type of predictions would you like to visualize?")
    print("1. Future forecasts (future_forecasts_*.csv)")
    print("2. Backtest results (backtest_results_*.csv)")
    print("3. Auto-detect latest file")

    choice = input("Choice (1, 2, or 3, default=1): ").strip()

    if choice == "2":
        file_type = "backtest"
    elif choice == "3":
        file_type = "auto"
    else:
        file_type = "future"

    # Try to find prediction files
    prediction_files = []
    if os.path.exists(predictions_dir):
        prediction_files = get_all_prediction_files(predictions_dir, file_type)

    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}/")
        file_name = input("Enter prediction file name (or full path): ").strip()
        if not os.path.exists(file_name):
            print(f"File {file_name} not found!")
            return
        prediction_files = [file_name]

    # Load and combine all prediction files
    print(f"Loading predictions from {len(prediction_files)} file(s):")
    for file in prediction_files:
        print(f"  - {os.path.basename(file)}")

    dfs = []
    try:
        for prediction_file in prediction_files:
            file_df = pd.read_csv(prediction_file)
            dfs.append(file_df)

        # Combine all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Handle both old and new file formats
        if 'future_date' in df.columns:
            # New future forecast format
            df['future_date'] = pd.to_datetime(df['future_date']).dt.tz_localize(None)
            df['date'] = df['future_date']
            prediction_type = "Future Forecasts"
        elif 'historical_date' in df.columns:
            # New backtest format
            df['historical_date'] = pd.to_datetime(df['historical_date']).dt.tz_localize(None)
            df['date'] = df['historical_date']
            prediction_type = "Backtest Results"
        else:
            # Old format fallback
            df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.tz_localize(None)
            df['date'] = df['prediction_date']
            prediction_type = "Predictions (Legacy)"

        print(f"Loaded {len(df)} {prediction_type.lower()} for {len(df['ticker'].unique())} tickers")
        print(f"Data type: {prediction_type}")

        # Show date range
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Show available tickers
    available_tickers = sorted(df['ticker'].unique())
    print(f"\nAvailable tickers ({len(available_tickers)}):")
    print(", ".join(available_tickers))

    # Get user input
    while True:
        print("\nEnter ticker symbols to analyze (comma-separated):")
        print("Examples: AAPL, AAPL,MSFT,GOOGL, or 'all' for all tickers")
        user_input = input("Tickers: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'all':
            selected_tickers = available_tickers
        else:
            selected_tickers = [t.strip().upper() for t in user_input.split(',')]
            # Validate tickers
            invalid_tickers = [t for t in selected_tickers if t not in available_tickers]
            if invalid_tickers:
                print(f"Invalid tickers: {', '.join(invalid_tickers)}")
                print(f"Available tickers: {', '.join(available_tickers)}")
                continue

        break

    print(f"\nAnalyzing: {', '.join(selected_tickers)}")

    # Ask for plot type
    if len(selected_tickers) == 1:
        show_historical = input("\nShow historical data? (y/n, default=y): ").strip().lower()
        show_historical = show_historical != 'n'
        plot_single_ticker(df, selected_tickers[0], show_historical)
    else:
        print("\nChoose plot type:")
        print("1. Individual plots for each ticker")
        print("2. Combined comparison plot")
        choice = input("Choice (1 or 2, default=1): ").strip()

        if choice == '2':
            plot_multiple_tickers(df, selected_tickers)
        else:
            show_historical = input("\nShow historical data for each ticker? (y/n, default=y): ").strip().lower()
            show_historical = show_historical != 'n'

            for ticker in selected_tickers:
                print(f"\n{'='*50}")
                print(f"Plotting {ticker}")
                print('='*50)
                plot_single_ticker(df, ticker, show_historical)


def plot_ticker_from_cmdline(ticker='AAPL', file_type='future'):
    """Simple command-line interface for testing"""
    try:
        # Get prediction files
        files = get_all_prediction_files('arimaxresults', file_type)
        if not files:
            print(f"No {file_type} prediction files found")
            return

        print(f"Using file: {files[0]}")

        # Load data
        df = pd.read_csv(files[0])
        df['future_date'] = pd.to_datetime(df['future_date'])
        df['date'] = df['future_date']

        # Check if ticker exists
        if ticker not in df['ticker'].unique():
            available = sorted(df['ticker'].unique())
            print(f"Ticker {ticker} not found. Available: {', '.join(available[:10])}...")
            return

        print(f"Plotting {ticker}...")
        plot_single_ticker(df, ticker, show_historical=True)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys

    # Simple command line usage: python visualize_predictions.py AAPL
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        plot_ticker_from_cmdline(ticker)
    else:
        main()