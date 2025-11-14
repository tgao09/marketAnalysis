# !/usr/bin/env python3


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import glob
import yfinance as yf


def get_all_prediction_files(predictions_dir="./arimaxresults", file_type="future"):
    
    if file_type == "future":
        pattern = os.path.join(predictions_dir, "future_forecasts_*.csv")
    elif file_type == "backtest":
        pattern = os.path.join(predictions_dir, "backtest_results_*.csv")
    else:
        # try both patterns and return all files
        future_pattern = os.path.join(predictions_dir, "future_forecasts_*.csv")
        backtest_pattern = os.path.join(predictions_dir, "backtest_results_*.csv")
        old_pattern = os.path.join(predictions_dir, "predictions_*.csv")

        all_files = glob.glob(future_pattern) + glob.glob(backtest_pattern) + glob.glob(old_pattern)
        return sorted(all_files, key=os.path.getctime, reverse=True)  # most recent first

    files = glob.glob(pattern)
    return sorted(files, key=os.path.getctime, reverse=True)  # most recent first


def get_latest_prediction_file(predictions_dir="./arimaxresults", file_type="future"):
    
    files = get_all_prediction_files(predictions_dir, file_type)
    return files[0] if files else None


def get_starting_price(ticker, target_date):
    
    try:
        print(f"Fetching starting price for {ticker}...")
        stock = yf.Ticker(ticker)

        # check if target_date is in the future
        today = datetime.now().date()
        if hasattr(target_date, 'date'):
            target_date_only = target_date.date()
        else:
            target_date_only = pd.to_datetime(target_date).date()

        if target_date_only > today:
            # future prediction - get current price
            print(f"Target date {target_date_only} is in the future, using current price...")
            # get recent data (last 5 days)
            hist = stock.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                print(f"Using current price for {ticker}: ${current_price:.2f}")
                return current_price
        else:
            # historical prediction - get week open price
            print(f"Target date {target_date_only} is historical, using week open price...")

        # get data from 2 weeks before target_date to ensure we have enough data
        start_date = target_date - timedelta(days=14)
        end_date = target_date + timedelta(days=7)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return None

        hist = hist.reset_index()

        # convert date column to timezone-naive datetime to match target_date
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)

        # ensure target_date is also timezone-naive
        if hasattr(target_date, 'tz') and target_date.tz is not None:
            target_date = target_date.tz_localize(None)
        elif hasattr(target_date, 'tzinfo') and target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        # find the monday of the target week (assuming week starts on monday)
        target_weekday = target_date.weekday()  # 0=monday, 6=sunday
        monday_of_target_week = target_date - timedelta(days=target_weekday)

        # look for the first trading day of the target week (monday or later)
        current_week_data = hist[hist['Date'] >= monday_of_target_week]

        if not current_week_data.empty:
            # use the first trading day's open price of the target week
            price = current_week_data.iloc[0]['Open']
            print(f"Using week open price for {ticker}: ${price:.2f}")
            return price
        else:
            # fallback: use last week's close (last available close before target week)
            previous_data = hist[hist['Date'] < monday_of_target_week]
            if not previous_data.empty:
                price = previous_data.iloc[-1]['Close']
                print(f"Using previous week close for {ticker}: ${price:.2f}")
                return price

    except Exception as e:
        print(f"Error fetching starting price for {ticker}: {e}")

    return None


def convert_returns_to_prices(ticker_df, ticker):
    
    # handle both old and new date column names
    date_col = 'future_date' if 'future_date' in ticker_df.columns else 'prediction_date'
    first_prediction_date = ticker_df[date_col].iloc[0]

    # get starting price (current price for future dates, week open for historical)
    starting_price = get_starting_price(ticker, first_prediction_date)
    if starting_price is None:
        print(f"Warning: Could not fetch starting price for {ticker}, using $100 as default")
        starting_price = 100.0

    # convert weekly returns to cumulative stock prices
    # assumption: week close = next week's open
    ticker_df = ticker_df.copy()
    ticker_df['stock_price'] = 0.0

    for i in range(len(ticker_df)):
        # check if returns are in decimal (new format) or percentage (old format)
        predicted_return = ticker_df.iloc[i]['predicted_return']
        # if the absolute value is small (< 1), assume it's decimal format
        if abs(predicted_return) < 1:
            return_multiplier = predicted_return  # already in decimal form
        else:
            return_multiplier = predicted_return / 100  # convert percentage to decimal

        if i == 0:
            # first prediction: apply return to week's open price
            week_close = starting_price * (1 + return_multiplier)
            ticker_df.iloc[i, ticker_df.columns.get_loc('stock_price')] = week_close
        else:
            # subsequent predictions: previous week's close = this week's open
            prev_week_close = ticker_df.iloc[i-1]['stock_price']
            current_week_close = prev_week_close * (1 + return_multiplier)
            ticker_df.iloc[i, ticker_df.columns.get_loc('stock_price')] = current_week_close

    return ticker_df


def get_historical_data(ticker, first_prediction_date, periods=28):
    
    try:
        print(f"Fetching historical data for {ticker}...")
        stock = yf.Ticker(ticker)

        # check if predictions are for future dates
        today = datetime.now().date()
        if hasattr(first_prediction_date, 'date'):
            pred_date = first_prediction_date.date()
        else:
            pred_date = pd.to_datetime(first_prediction_date).date()

        if pred_date > today:
            # future predictions - get recent historical data up to today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods + 10)
            print(f"Getting recent historical data up to today...")
        else:
            # historical predictions - get data up to prediction start
            end_date = first_prediction_date
            start_date = end_date - timedelta(days=periods + 10)
            print(f"Getting historical data up to {pred_date}...")

        hist = stock.history(start=start_date, end=end_date)

        if not hist.empty:
            hist = hist.reset_index()
            hist['ticker'] = ticker
            # convert date to timezone-naive to match prediction dates
            hist['date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            hist['stock_price'] = hist['Close']
            return hist[['ticker', 'date', 'stock_price']].tail(periods)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
    return pd.DataFrame()


def plot_single_ticker(df, ticker, show_historical=True):
    
    # filter data for this ticker
    ticker_df = df[df['ticker'] == ticker].copy()
    if ticker_df.empty:
        print(f"No prediction data found for {ticker}")
        return

    # convert returns to prices
    ticker_df = convert_returns_to_prices(ticker_df, ticker)
    ticker_df['data_type'] = 'predicted'

    # get historical data if requested
    historical_data = pd.DataFrame()
    if show_historical:
        first_prediction_date = ticker_df['date'].min()
        historical_data = get_historical_data(ticker, first_prediction_date, periods=28)
        if not historical_data.empty:
            historical_data['data_type'] = 'historical'

    # combine data
    if not historical_data.empty:
        combined_data = pd.concat([historical_data, ticker_df[['ticker', 'date', 'stock_price', 'data_type']]],
                                 ignore_index=True)
    else:
        combined_data = ticker_df[['ticker', 'date', 'stock_price', 'data_type']].copy()

    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    # create plot
    plt.figure(figsize=(15, 10))

    # separate data types
    historical_subset = combined_data[combined_data['data_type'] == 'historical']
    predicted_subset = combined_data[combined_data['data_type'] == 'predicted']

    # plot historical data
    if len(historical_subset) > 0:
        sns.lineplot(data=historical_subset, x='date', y='stock_price',
                    label=f'{ticker} Historical Price', linewidth=2.5,
                    marker='o', markersize=4, color='blue')

    # plot predictions
    if len(predicted_subset) > 0:
        sns.lineplot(data=predicted_subset, x='date', y='stock_price',
                    label=f'{ticker} Predicted Price', linewidth=2.5,
                    linestyle='--', marker='s', markersize=5, color='red')

    # add confidence intervals if available
    if 'ci_lower' in ticker_df.columns and 'ci_upper' in ticker_df.columns:
        # convert confidence intervals from returns to prices
        date_col = 'future_date' if 'future_date' in ticker_df.columns else 'prediction_date'
        first_prediction_date = ticker_df[date_col].iloc[0]
        week_open_price = get_starting_price(ticker, first_prediction_date)

        if week_open_price is None:
            initial_price_ci = ticker_df['stock_price'].iloc[0] / (1 + ticker_df['predicted_return'].iloc[0] / 100)
        else:
            initial_price_ci = week_open_price

        # calculate ci prices
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

            # handle both decimal and percentage format for confidence intervals
            ci_lower = ticker_df_with_ci.iloc[i]['ci_lower']
            ci_upper = ticker_df_with_ci.iloc[i]['ci_upper']

            # check if ci values are in decimal (new format) or percentage (old format)
            if abs(ci_lower) < 1 and abs(ci_upper) < 1:
                ci_lower_mult = ci_lower  # already in decimal form
                ci_upper_mult = ci_upper  # already in decimal form
            else:
                ci_lower_mult = ci_lower / 100  # convert percentage to decimal
                ci_upper_mult = ci_upper / 100  # convert percentage to decimal

            ticker_df_with_ci.iloc[i, ticker_df_with_ci.columns.get_loc('ci_lower_price')] = base_price * (1 + ci_lower_mult)
            ticker_df_with_ci.iloc[i, ticker_df_with_ci.columns.get_loc('ci_upper_price')] = base_price * (1 + ci_upper_mult)

        # plot confidence intervals
        plt.fill_between(ticker_df_with_ci['date'],
                        ticker_df_with_ci['ci_lower_price'],
                        ticker_df_with_ci['ci_upper_price'],
                        alpha=0.2, color='red', label='95% Confidence Interval')

    # determine plot title based on data type
    if 'future_date' in ticker_df.columns:
        plot_title = f'{ticker} Stock Price: Historical Data + Future Forecasts'
    elif 'historical_date' in ticker_df.columns:
        plot_title = f'{ticker} Stock Price: Historical Data + Backtest Results'
    else:
        plot_title = f'{ticker} Stock Price: Historical Data + ARIMAX Predictions'

    plt.title(plot_title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)

    # add separation line
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

    # print summary
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

    # show model info
    if len(ticker_df) > 0:
        model_info = ticker_df.iloc[0]
        print(f"ARIMAX Model Order: {model_info['model_order']}")
        print(f"Model AIC: {model_info['model_aic']:.2f}")


def plot_multiple_tickers(df, tickers):
    
    plt.figure(figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))

    for i, ticker in enumerate(tickers):
        ticker_df = df[df['ticker'] == ticker].copy()
        if ticker_df.empty:
            print(f"No prediction data found for {ticker}")
            continue

        # convert returns to prices (without printing individual progress)
        date_col = 'future_date' if 'future_date' in ticker_df.columns else 'prediction_date'
        first_prediction_date = ticker_df[date_col].iloc[0]
        week_open_price = get_starting_price(ticker, first_prediction_date)

        if week_open_price is None:
            starting_price = 100.0
        else:
            starting_price = week_open_price

        ticker_df['stock_price'] = 0.0
        for j in range(len(ticker_df)):
            # handle both decimal and percentage format
            predicted_return = ticker_df.iloc[j]['predicted_return']
            if abs(predicted_return) < 1:
                return_multiplier = predicted_return  # already in decimal form
            else:
                return_multiplier = predicted_return / 100  # convert percentage to decimal

            if j == 0:
                week_close = starting_price * (1 + return_multiplier)
                ticker_df.iloc[j, ticker_df.columns.get_loc('stock_price')] = week_close
            else:
                prev_week_close = ticker_df.iloc[j-1]['stock_price']
                current_week_close = prev_week_close * (1 + return_multiplier)
                ticker_df.iloc[j, ticker_df.columns.get_loc('stock_price')] = current_week_close

        # plot this ticker
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
    
    print("=== Stock Prediction Visualization ===\n")

    # configuration
    predictions_dir = "arimaxresults"

    # ask user for file type preference
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

    # try to find prediction files
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

    # load and combine all prediction files
    print(f"Loading predictions from {len(prediction_files)} file(s):")
    for file in prediction_files:
        print(f"  - {os.path.basename(file)}")

    dfs = []
    try:
        for prediction_file in prediction_files:
            file_df = pd.read_csv(prediction_file)
            dfs.append(file_df)

        # combine all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # handle both old and new file formats
        if 'future_date' in df.columns:
            # new future forecast format
            df['future_date'] = pd.to_datetime(df['future_date']).dt.tz_localize(None)
            df['date'] = df['future_date']
            prediction_type = "Future Forecasts"
        elif 'historical_date' in df.columns:
            # new backtest format
            df['historical_date'] = pd.to_datetime(df['historical_date']).dt.tz_localize(None)
            df['date'] = df['historical_date']
            prediction_type = "Backtest Results"
        else:
            # old format fallback
            df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.tz_localize(None)
            df['date'] = df['prediction_date']
            prediction_type = "Predictions (Legacy)"

        print(f"Loaded {len(df)} {prediction_type.lower()} for {len(df['ticker'].unique())} tickers")
        print(f"Data type: {prediction_type}")

        # show date range
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # show available tickers
    available_tickers = sorted(df['ticker'].unique())
    print(f"\nAvailable tickers ({len(available_tickers)}):")
    print(", ".join(available_tickers))

    # get user input
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
            # validate tickers
            invalid_tickers = [t for t in selected_tickers if t not in available_tickers]
            if invalid_tickers:
                print(f"Invalid tickers: {', '.join(invalid_tickers)}")
                print(f"Available tickers: {', '.join(available_tickers)}")
                continue

        break

    print(f"\nAnalyzing: {', '.join(selected_tickers)}")

    # ask for plot type
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
    
    try:
        # get prediction files
        files = get_all_prediction_files('arimaxresults', file_type)
        if not files:
            print(f"No {file_type} prediction files found")
            return

        print(f"Using file: {files[0]}")

        # load data
        df = pd.read_csv(files[0])
        df['future_date'] = pd.to_datetime(df['future_date'])
        df['date'] = df['future_date']

        # check if ticker exists
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

    # simple command line usage: python visualize_predictions.py aapl
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        plot_ticker_from_cmdline(ticker)
    else:
        main()