

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import sys
import os

# add arimax directory to path for importing stockarimax
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from arimax_model import StockARIMAX

warnings.filterwarnings('ignore')


class WalkForwardTester:
    

    def __init__(self,
                 model_dir: str = "arimax/models",
                 data_file: str = "dataset/stock_dataset_with_lags.csv",
                 n_models: int = 10,
                 test_weeks: int = 12,
                 initial_capital: float = 10000.0):
        
        self.model_dir = Path(model_dir)
        self.data_file = data_file
        self.n_models = n_models
        self.test_weeks = test_weeks
        self.initial_capital = initial_capital
        self.results = []
        self.price_cache = {}  # cache for price data

    def load_data(self) -> pd.DataFrame:
        
        df = pd.read_csv(self.data_file)
        # handle both 'date' and 'date' column names
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date'])
        return df

    def get_available_models(self) -> List[str]:
        
        model_files = list(self.model_dir.glob("*_arimax.pkl"))
        return [f.stem.replace("_arimax", "") for f in model_files]

    def select_random_models(self, available_tickers: List[str]) -> List[str]:
        
        if len(available_tickers) <= self.n_models:
            return available_tickers
        return random.sample(available_tickers, self.n_models)

    def load_model(self, ticker: str):
        
        model_path = self.model_dir / f"{ticker}_arimax.pkl"
        # models are saved using joblib, not pickle
        model_data = joblib.load(model_path)

        # reconstruct stockarimax object from saved data
        model = StockARIMAX(ticker=ticker)
        model.fitted_model = model_data.get('fitted_model')
        model.best_order = model_data.get('best_order')
        model.aic_score = model_data.get('aic_score')
        model.feature_columns = model_data.get('feature_columns', [])
        model.is_fitted = True

        # set exog_features for compatibility
        model.exog_features = model.feature_columns

        return model

    def fetch_price_data(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        
        # check cache first
        cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        try:
            # add buffer to ensure we get all needed data
            fetch_start = (start_date - timedelta(days=14)).strftime('%Y-%m-%d')
            fetch_end = (end_date + timedelta(days=7)).strftime('%Y-%m-%d')

            stock = yf.Ticker(ticker)
            price_data = stock.history(start=fetch_start, end=fetch_end, interval='1d')

            if price_data.empty:
                print(f"  Warning: No price data found for {ticker}")
                return None

            # clean and prepare
            price_data = price_data.reset_index()
            price_data.columns = [col.lower() for col in price_data.columns]
            price_data = price_data[['date', 'close']].copy()
            price_data['date'] = pd.to_datetime(price_data['date'])

            # cache the result
            self.price_cache[cache_key] = price_data

            return price_data

        except Exception as e:
            print(f"  Warning: Failed to fetch price data for {ticker}: {e}")
            return None

    def calculate_trading_returns(self,
                                   forecast_df: pd.DataFrame,
                                   ticker: str,
                                   price_data: pd.DataFrame) -> Dict:
        
        if price_data is None or len(forecast_df) == 0:
            return {
                'total_return': np.nan,
                'annualized_return': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'profit_factor': np.nan,
                'final_value': np.nan
            }

        # merge forecasts with prices
        forecast_df = forecast_df.copy()
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])

        trades = []
        capital = self.initial_capital
        equity_curve = [capital]

        for i in range(len(forecast_df)):
            row = forecast_df.iloc[i]

            # skip if forecast is nan
            if pd.isna(row['forecast']):
                equity_curve.append(capital)
                continue

            # get entry and exit prices
            forecast_date = pd.to_datetime(row['date'])

            # entry: closest price on/after forecast date (beginning of week)
            entry_prices = price_data[price_data['date'] >= forecast_date]
            if len(entry_prices) == 0:
                equity_curve.append(capital)
                continue
            entry_price = entry_prices.iloc[0]['close']
            entry_date = entry_prices.iloc[0]['date']

            # exit: one week later (end of week)
            exit_date = entry_date + timedelta(days=7)
            exit_prices = price_data[price_data['date'] >= exit_date]
            if len(exit_prices) == 0:
                equity_curve.append(capital)
                continue
            exit_price = exit_prices.iloc[0]['close']
            actual_exit_date = exit_prices.iloc[0]['date']

            # trade logic: only go long if forecast is positive
            if row['forecast'] > 0:
                # calculate return
                trade_return = (exit_price / entry_price) - 1
                trade_pnl = capital * trade_return
                capital += trade_pnl

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': actual_exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'pnl': trade_pnl,
                    'forecast': row['forecast'],
                    'actual_return': row['actual']
                })
            # else: stay in cash, no change to capital

            equity_curve.append(capital)

        # calculate metrics
        if len(trades) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'final_value': self.initial_capital,
                'num_trades': 0
            }

        trades_df = pd.DataFrame(trades)

        # total return
        total_return = (capital - self.initial_capital) / self.initial_capital

        # annualized return (assuming test_weeks is the period)
        years = self.test_weeks / 52.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return

        # sharpe ratio (weekly returns)
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(52) if equity_returns.std() > 0 else 0.0

        # maximum drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # win rate
        win_rate = (trades_df['return'] > 0).mean()

        # profit factor
        gains = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gains / losses if losses > 0 else np.nan

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_value': capital,
            'num_trades': len(trades)
        }

    def prepare_walk_forward_data(self,
                                   df: pd.DataFrame,
                                   ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')

        if len(ticker_data) < self.test_weeks:
            raise ValueError(f"Insufficient data for {ticker}: {len(ticker_data)} weeks")

        # split: everything except last test_weeks for training context
        # last test_weeks for walk-forward testing
        train_data = ticker_data.iloc[:-self.test_weeks]
        test_data = ticker_data.iloc[-self.test_weeks:]

        return train_data, test_data

    def walk_forward_forecast(self,
                               model_obj,
                               train_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> pd.DataFrame:
        
        forecasts = []

        # get exogenous feature names from model
        exog_features = model_obj.exog_features if hasattr(model_obj, 'exog_features') else []

        # get the fitted model (statsmodels sarimax object)
        fitted_model = model_obj.fitted_model

        for i in range(len(test_data)):
            current_test = test_data.iloc[i]
            y_test = current_test['weekly_return']

            # prepare exogenous variables if model uses them
            exog_forecast = None
            if exog_features:
                # get exogenous values for forecast period and ensure they're numeric
                try:
                    exog_values = current_test[exog_features].values
                    # convert to float to ensure numeric types
                    exog_array = np.array(exog_values, dtype=np.float64).reshape(1, -1)

                    # check for nan or inf values
                    if np.any(~np.isfinite(exog_array)):
                        nan_features = [exog_features[j] for j in range(len(exog_features))
                                       if not np.isfinite(exog_array[0, j])]
                        print(f"  Warning: Non-finite values in features: {nan_features}")
                        # replace nan/inf with 0 (or could skip this forecast)
                        exog_array = np.nan_to_num(exog_array, nan=0.0, posinf=0.0, neginf=0.0)

                    exog_forecast = exog_array
                except (ValueError, TypeError) as e:
                    print(f"  Warning: Could not convert exog features to numeric: {e}")
                    print(f"  Feature types: {current_test[exog_features].apply(type).to_dict()}")
                    exog_forecast = None

            # get 1-step ahead forecast using the fitted model
            try:
                # use get_forecast for proper out-of-sample prediction
                if exog_forecast is not None:
                    forecast_result = fitted_model.get_forecast(steps=1, exog=exog_forecast)
                else:
                    forecast_result = fitted_model.get_forecast(steps=1)

                # extract forecast value and ensure it's a float
                forecast_val = forecast_result.predicted_mean
                if hasattr(forecast_val, 'iloc'):
                    forecast = float(forecast_val.iloc[0])
                elif hasattr(forecast_val, '__getitem__'):
                    forecast = float(forecast_val[0])
                else:
                    forecast = float(forecast_val)

                # ensure actual is also float
                y_test_float = float(y_test)

                forecasts.append({
                    'date': current_test['date'],
                    'actual': y_test_float,
                    'forecast': forecast,
                    'error': forecast - y_test_float,
                    'abs_error': abs(forecast - y_test_float),
                    'squared_error': (forecast - y_test_float) ** 2,
                    'direction_correct': (np.sign(forecast) == np.sign(y_test_float))
                })

                # update the model with observed data for next iteration
                # this allows the model to use actual past values for next forecast
                if i < len(test_data) - 1:  # don't append after last forecast
                    exog_append = exog_forecast if exog_forecast is not None else None
                    fitted_model = fitted_model.append([y_test_float], exog=exog_append, refit=False)

            except Exception as e:
                print(f"  Warning: Forecast failed for week {i}: {e}")
                import traceback
                traceback.print_exc()
                forecasts.append({
                    'date': current_test['date'],
                    'actual': y_test,
                    'forecast': np.nan,
                    'error': np.nan,
                    'abs_error': np.nan,
                    'squared_error': np.nan,
                    'direction_correct': False
                })

        return pd.DataFrame(forecasts)

    def calculate_metrics(self, forecast_df: pd.DataFrame) -> Dict:
        
        # remove any nan forecasts
        valid_forecasts = forecast_df.dropna(subset=['forecast'])

        if len(valid_forecasts) == 0:
            return {
                'n_forecasts': 0,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'direction_accuracy': np.nan,
                'mean_error': np.nan
            }

        rmse = np.sqrt(valid_forecasts['squared_error'].mean())
        mae = valid_forecasts['abs_error'].mean()

        # mape (only for non-zero actuals)
        non_zero = valid_forecasts[valid_forecasts['actual'] != 0]
        if len(non_zero) > 0:
            mape = (non_zero['abs_error'] / non_zero['actual'].abs()).mean() * 100
        else:
            mape = np.nan

        direction_accuracy = valid_forecasts['direction_correct'].mean() * 100
        mean_error = valid_forecasts['error'].mean()

        return {
            'n_forecasts': len(valid_forecasts),
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'mean_error': mean_error
        }

    def test_model(self, ticker: str, df: pd.DataFrame) -> Dict:
        
        print(f"\nTesting {ticker}...")

        try:
            # load model
            model_obj = self.load_model(ticker)

            # prepare data
            train_data, test_data = self.prepare_walk_forward_data(df, ticker)

            print(f"  Train: {len(train_data)} weeks, Test: {len(test_data)} weeks")
            print(f"  Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")

            # perform walk-forward forecasting
            forecast_df = self.walk_forward_forecast(model_obj, train_data, test_data)

            # calculate forecast accuracy metrics
            metrics = self.calculate_metrics(forecast_df)

            # fetch price data for profitability calculations
            test_start = test_data['date'].min()
            test_end = test_data['date'].max()
            price_data = self.fetch_price_data(ticker, test_start, test_end)

            # calculate profitability metrics
            profitability = self.calculate_trading_returns(forecast_df, ticker, price_data)
            metrics.update(profitability)

            # add ticker and test period info
            metrics['ticker'] = ticker
            metrics['test_start'] = test_start
            metrics['test_end'] = test_end
            metrics['train_weeks'] = len(train_data)
            metrics['test_weeks'] = len(test_data)

            print(f"  RMSE: {metrics['rmse']:.4f}, Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
            if not pd.isna(metrics.get('total_return')):
                print(f"  Total Return: {metrics['total_return']*100:.2f}%, Win Rate: {metrics['win_rate']*100:.1f}%")

            return metrics

        except Exception as e:
            print(f"  Error testing {ticker}: {e}")
            return {
                'ticker': ticker,
                'n_forecasts': 0,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'direction_accuracy': np.nan,
                'mean_error': np.nan,
                'total_return': np.nan,
                'annualized_return': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'profit_factor': np.nan,
                'final_value': np.nan,
                'num_trades': np.nan,
                'error': str(e)
            }

    def run(self) -> pd.DataFrame:
        
        print(f"=== Walk-Forward Testing ===")
        print(f"Test Period: {self.test_weeks} weeks")
        print(f"Number of Models: {self.n_models}")

        # load data
        print("\nLoading data...")
        df = self.load_data()
        print(f"Data loaded: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

        # get available models
        available_tickers = self.get_available_models()
        print(f"Available models: {len(available_tickers)}")

        # randomly select models
        selected_tickers = self.select_random_models(available_tickers)
        print(f"Selected models: {selected_tickers}")

        # test each model
        results = []
        for ticker in selected_tickers:
            result = self.test_model(ticker, df)
            results.append(result)

        # convert to dataframe
        results_df = pd.DataFrame(results)

        # calculate summary statistics
        print("\n=== Summary Statistics ===")
        valid_results = results_df[results_df['n_forecasts'] > 0]

        if len(valid_results) > 0:
            print(f"Successfully tested: {len(valid_results)}/{len(selected_tickers)} models")
            print(f"\n--- Forecast Accuracy ---")
            print(f"RMSE: {valid_results['rmse'].mean():.4f} ± {valid_results['rmse'].std():.4f}")
            print(f"MAE: {valid_results['mae'].mean():.4f} ± {valid_results['mae'].std():.4f}")
            print(f"Direction Accuracy: {valid_results['direction_accuracy'].mean():.1f}% ± {valid_results['direction_accuracy'].std():.1f}%")
            print(f"Mean Error (bias): {valid_results['mean_error'].mean():.4f}")

            # profitability statistics
            profit_results = valid_results.dropna(subset=['total_return'])
            if len(profit_results) > 0:
                print(f"\n--- Profitability (Long-Only Strategy) ---")
                print(f"Total Return: {profit_results['total_return'].mean()*100:.2f}% ± {profit_results['total_return'].std()*100:.2f}%")
                print(f"Annualized Return: {profit_results['annualized_return'].mean()*100:.2f}% ± {profit_results['annualized_return'].std()*100:.2f}%")
                print(f"Sharpe Ratio: {profit_results['sharpe_ratio'].mean():.2f} ± {profit_results['sharpe_ratio'].std():.2f}")
                print(f"Max Drawdown: {profit_results['max_drawdown'].mean()*100:.2f}% ± {profit_results['max_drawdown'].std()*100:.2f}%")
                print(f"Win Rate: {profit_results['win_rate'].mean()*100:.1f}% ± {profit_results['win_rate'].std()*100:.1f}%")
                print(f"Profit Factor: {profit_results['profit_factor'].mean():.2f} ± {profit_results['profit_factor'].std():.2f}")
                print(f"Avg Trades per Stock: {profit_results['num_trades'].mean():.1f}")

                # count profitable vs unprofitable
                profitable = (profit_results['total_return'] > 0).sum()
                unprofitable = (profit_results['total_return'] <= 0).sum()
                print(f"\nProfitable Stocks: {profitable}/{len(profit_results)} ({profitable/len(profit_results)*100:.1f}%)")
        else:
            print("No models successfully tested.")

        # save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"arimax/results/walk_forward_test_{timestamp}.csv"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        return results_df


def main():
    
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward testing for ARIMAX models")
    parser.add_argument("--n-models", type=int, default=10,
                       help="Number of models to randomly test (default: 10)")
    parser.add_argument("--test-weeks", type=int, default=12,
                       help="Number of weeks for walk-forward testing (default: 12)")
    parser.add_argument("--initial-capital", type=float, default=10000.0,
                       help="Initial capital for profitability simulation (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # run walk-forward testing
    tester = WalkForwardTester(
        n_models=args.n_models,
        test_weeks=args.test_weeks,
        initial_capital=args.initial_capital
    )
    results = tester.run()


if __name__ == "__main__":
    main()
