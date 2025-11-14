

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xgboost_model import StockXGBoost, PurgedWalkForwardCV
from feature_engineering import TechnicalFeatureEngineer
from model_diagnostics import ModelDiagnostics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostWalkForwardTester:
    

    def __init__(self, models_dir: str = 'xgboost/xgboostmodels',
                 data_file: str = 'xgboost/features_engineered.csv',
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.10,
                 recompute_technical: bool = True):
        
        self.models_dir = Path(models_dir)
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.recompute_technical = recompute_technical
        self.feature_engineer = TechnicalFeatureEngineer() if recompute_technical else None

    def recompute_forward_returns(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        
        logger.info(f"Recomputing forward returns for {horizon}-week horizon...")

        result_dfs = []
        target_col = f'forward_return_{horizon}w'

        for ticker in df['ticker'].unique():
            stock_df = df[df['ticker'] == ticker].copy()
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)

            forward_returns = []
            for i in range(len(stock_df)):
                # get next 'horizon' weeks of returns (i+1 to i+horizon)
                future_window = stock_df['weekly_return'].iloc[i+1:i+1+horizon]

                if len(future_window) == horizon:
                    # compute cumulative return
                    cum_return = (1 + future_window).prod() - 1
                    forward_returns.append(cum_return)
                else:
                    # not enough future data
                    forward_returns.append(np.nan)

            stock_df[target_col] = forward_returns
            result_dfs.append(stock_df)

        result = pd.concat(result_dfs, ignore_index=True)
        result = result.sort_values(['ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Recomputed forward returns: {(~result[target_col].isna()).sum()} valid samples")
        return result

    def recompute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if not self.recompute_technical or self.feature_engineer is None:
            return df

        logger.info("Recomputing technical indicators for realistic backtesting...")

        # calculate technical features (rsi, macd, bollinger bands, etc.)
        result = self.feature_engineer.calculate_technical_features(df)

        # calculate cross-sectional features (rankings, z-scores)
        result = self.feature_engineer.calculate_cross_sectional_features(result)

        # calculate interaction features
        result = self.feature_engineer.calculate_interaction_features(result)

        # calculate time features
        result = self.feature_engineer.calculate_time_features(result)

        logger.info("Technical indicators recomputed successfully")
        return result

    def load_models(self) -> Dict[int, StockXGBoost]:
        
        models = {}

        for model_file in self.models_dir.glob('xgboost_*.pkl'):
            # extract horizon
            parts = model_file.stem.split('_')
            if len(parts) >= 2:
                horizon_str = parts[1].replace('w', '')
                try:
                    horizon = int(horizon_str)
                    models[horizon] = StockXGBoost.load_model(str(model_file))
                    logger.info(f"Loaded {horizon}-week model")
                except ValueError:
                    continue

        return models

    def simulate_trading(self, predictions_df: pd.DataFrame,
                        prices_df: pd.DataFrame,
                        horizon: int = 1) -> Dict:
        
        capital = self.initial_capital
        equity_curve = [capital]
        positions = {}  # ticker -> shares
        trades = []

        # sort predictions by date
        predictions_df = predictions_df.sort_values('Date')

        for date in predictions_df['Date'].unique():
            date_predictions = predictions_df[predictions_df['Date'] == date]

            # get top long opportunities (positive predicted returns)
            longs = date_predictions[date_predictions['predicted_return'] > 0].nlargest(5, 'predicted_return')

            # close existing positions
            for ticker in list(positions.keys()):
                if ticker not in longs['ticker'].values:
                    # get exit price
                    price_data = prices_df[(prices_df['ticker'] == ticker) & (prices_df['Date'] >= date)]
                    if len(price_data) > 0:
                        exit_price = price_data.iloc[0]['close']
                        shares = positions[ticker]
                        proceeds = shares * exit_price * (1 - self.transaction_cost)
                        capital += proceeds
                        del positions[ticker]

            # open new positions
            if len(longs) > 0:
                position_size = capital * self.max_position_size
                for _, row in longs.iterrows():
                    ticker = row['ticker']
                    # get entry price
                    price_data = prices_df[(prices_df['ticker'] == ticker) & (prices_df['Date'] >= date)]
                    if len(price_data) > 0:
                        entry_price = price_data.iloc[0]['close']
                        cost = position_size * (1 + self.transaction_cost)
                        if cost <= capital:
                            shares = position_size / entry_price
                            positions[ticker] = shares
                            capital -= cost

            # mark to market
            portfolio_value = capital
            for ticker, shares in positions.items():
                price_data = prices_df[(prices_df['ticker'] == ticker) & (prices_df['Date'] >= date)]
                if len(price_data) > 0:
                    current_price = price_data.iloc[0]['close']
                    portfolio_value += shares * current_price

            equity_curve.append(portfolio_value)

        # calculate metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        sharpe = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': equity_curve[-1],
            'num_periods': len(equity_curve)
        }

    def run_test(self, horizon: int = 1, test_weeks: int = 52,
                 num_positions: int = 5, allow_short: bool = False) -> Dict:
        
        logger.info(f"Testing {horizon}-week model...")

        # load data
        df = pd.read_csv(self.data_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

        # split: last test_weeks for testing
        df = df.sort_values(['ticker', 'Date'])
        test_start_date = df['Date'].max() - timedelta(weeks=test_weeks)

        train_df = df[df['Date'] < test_start_date]
        test_df = df[df['Date'] >= test_start_date]

        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        # critical: recompute features to prevent data leakage and simulate real trading
        # step 1: recompute technical indicators using only historical data per fold
        if self.recompute_technical:
            # recompute technical indicators on train set only (simulates real-time)
            train_df_recomputed = self.recompute_technical_indicators(train_df)
            # for test set, compute using train + test up to each point (expanding window)
            test_df_recomputed = self.recompute_technical_indicators(df[df['Date'] < df['Date'].max()])
            # filter to test dates
            test_df = test_df_recomputed[test_df_recomputed['Date'] >= test_start_date].copy()
            train_df = train_df_recomputed.copy()
        else:
            logger.info("Using pre-computed technical indicators (not recommended for backtesting)")

        # step 2: recompute forward returns to prevent leakage
        logger.info("Recomputing forward returns to prevent data leakage...")
        df_combined = pd.concat([train_df, test_df], ignore_index=True).sort_values(['ticker', 'Date'])
        df_combined = self.recompute_forward_returns(df_combined, horizon)

        # re-split after recomputing
        train_df = df_combined[df_combined['Date'] < test_start_date]
        test_df = df_combined[df_combined['Date'] >= test_start_date]

        # load model
        models = self.load_models()
        if horizon not in models:
            raise ValueError(f"No model found for {horizon}-week horizon")

        model = models[horizon]

        # generate predictions on test set
        test_predictions = model.predict(test_df)

        # create predictions dataframe
        predictions_df = test_df[['ticker', 'Date']].copy()
        predictions_df['predicted_return'] = test_predictions

        # calculate prediction accuracy
        target_col = f'forward_return_{horizon}w'
        results = {
            'horizon': horizon,
            'test_weeks': test_weeks,
            'num_positions': num_positions,
            'strategy': 'long-short' if allow_short else 'long-only'
        }

        if target_col in test_df.columns:
            actuals = test_df[target_col].values
            valid_idx = ~pd.isna(actuals)

            rmse = np.sqrt(np.mean((actuals[valid_idx] - test_predictions[valid_idx])**2))
            direction_acc = np.mean(np.sign(actuals[valid_idx]) == np.sign(test_predictions[valid_idx]))

            results['test_rmse'] = rmse
            results['direction_accuracy'] = direction_acc * 100
            results['test_samples'] = int(valid_idx.sum())

            logger.info(f"Test RMSE: {rmse:.4f}, Direction Accuracy: {direction_acc*100:.1f}%")

            # new: add distribution diagnostics to detect sign bias
            dist_metrics = ModelDiagnostics.calculate_prediction_distribution_metrics(
                test_predictions[valid_idx],
                actuals[valid_idx]
            )
            has_sign_bias = ModelDiagnostics.check_sign_bias(test_predictions[valid_idx], threshold=0.85)

            # add diagnostics to results
            results.update({
                'pct_positive': dist_metrics['pct_positive'],
                'pct_negative': dist_metrics['pct_negative'],
                'pred_mean': dist_metrics['pred_mean'],
                'pred_std': dist_metrics['pred_std'],
                'true_mean': dist_metrics['true_mean'],
                'true_std': dist_metrics['true_std'],
                'pred_range': dist_metrics['pred_range'],
                'true_range': dist_metrics['true_range'],
                'std_ratio': dist_metrics['std_ratio'],
                'range_ratio': dist_metrics['range_ratio'],
                'prediction_diversity': dist_metrics['prediction_diversity'],
                'has_sign_bias': has_sign_bias,
            })

            # log warnings if issues detected
            if has_sign_bias:
                logger.warning(f"⚠️ SIGN BIAS DETECTED: {dist_metrics['pct_positive']:.1f}% positive predictions")
            if dist_metrics['prediction_diversity'] < 100:
                logger.warning(f"⚠️ LOW DIVERSITY: Only {dist_metrics['prediction_diversity']} unique predictions")
            if dist_metrics['std_ratio'] > 3.0 or dist_metrics['std_ratio'] < 0.3:
                logger.warning(f"⚠️ STD MISMATCH: pred_std/true_std = {dist_metrics['std_ratio']:.2f}")

            # validation check
            is_valid, failures = ModelDiagnostics.validate_model_quality(
                test_predictions[valid_idx],
                actuals[valid_idx]
            )
            results['validation_passed'] = is_valid
            results['validation_failures'] = failures

            if not is_valid:
                logger.error(f"❌ MODEL VALIDATION FAILED:")
                for failure in failures:
                    logger.error(f"   • {failure}")

        # calculate profitability using actual forward returns
        if target_col in test_df.columns:
            logger.info("Calculating portfolio profitability from forward returns...")

            portfolio_returns = []
            dates = sorted(predictions_df['Date'].unique())

            for date in dates:
                # get predictions for this date
                date_preds = predictions_df[predictions_df['Date'] == date].copy()

                # select top stocks based on strategy
                if allow_short:
                    # long/short: top n longs + top n shorts
                    longs = date_preds.nlargest(num_positions, 'predicted_return')
                    shorts = date_preds.nsmallest(num_positions, 'predicted_return')
                    selected_stocks = pd.concat([longs, shorts])
                else:
                    # long-only: top n predicted positive returns
                    selected_stocks = date_preds[date_preds['predicted_return'] > 0].nlargest(num_positions, 'predicted_return')

                if len(selected_stocks) == 0:
                    continue

                # get actual forward returns for selected stocks
                for _, row in selected_stocks.iterrows():
                    ticker = row['ticker']
                    predicted = row['predicted_return']

                    # find actual return in test_df
                    actual_row = test_df[(test_df['ticker'] == ticker) &
                                        (test_df['Date'] == date)]

                    if len(actual_row) > 0 and target_col in actual_row.columns:
                        actual_return = actual_row[target_col].iloc[0]
                        if not pd.isna(actual_return):
                            # for shorts, invert the actual return (profit from decline)
                            if allow_short and predicted < 0:
                                actual_return = -actual_return

                            portfolio_returns.append({
                                'date': date,
                                'ticker': ticker,
                                'predicted': predicted,
                                'actual': actual_return,
                                'position_type': 'short' if (allow_short and predicted < 0) else 'long'
                            })

            if portfolio_returns:
                returns_df = pd.DataFrame(portfolio_returns)

                # calculate aggregate portfolio performance
                # equal weight: average return across holdings per period
                period_returns = returns_df.groupby('date')['actual'].mean()

                # apply transaction costs (assume rebalance each period)
                period_returns = period_returns - (self.transaction_cost * 2)  # buy + sell

                # calculate cumulative performance
                cumulative_return = (1 + period_returns).prod() - 1

                # annualize
                years = test_weeks / 52.0
                annualized_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else cumulative_return

                # sharpe ratio
                sharpe = period_returns.mean() / period_returns.std() * np.sqrt(52 / horizon) if period_returns.std() > 0 else 0

                # max drawdown
                cumulative_curve = (1 + period_returns).cumprod()
                cummax = cumulative_curve.cummax()
                drawdown = (cumulative_curve - cummax) / cummax
                max_drawdown = drawdown.min()

                # win rate
                win_rate = (period_returns > 0).mean()

                # calculate position-level stats
                profitability = {
                    'cumulative_return': cumulative_return,
                    'annualized_return': annualized_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'num_periods': len(period_returns),
                    'avg_period_return': period_returns.mean(),
                    'avg_position_return': returns_df['actual'].mean()
                }

                # add long/short breakdown if applicable
                if allow_short and 'position_type' in returns_df.columns:
                    long_positions = returns_df[returns_df['position_type'] == 'long']
                    short_positions = returns_df[returns_df['position_type'] == 'short']

                    profitability['num_long_positions'] = len(long_positions)
                    profitability['num_short_positions'] = len(short_positions)
                    profitability['avg_long_return'] = long_positions['actual'].mean() if len(long_positions) > 0 else 0
                    profitability['avg_short_return'] = short_positions['actual'].mean() if len(short_positions) > 0 else 0
                    profitability['long_win_rate'] = (long_positions['actual'] > 0).mean() if len(long_positions) > 0 else 0
                    profitability['short_win_rate'] = (short_positions['actual'] > 0).mean() if len(short_positions) > 0 else 0

                results['profitability'] = profitability

                logger.info(f"Profitability: {cumulative_return*100:.2f}% total, "
                          f"{annualized_return*100:.2f}% annualized, "
                          f"Sharpe={sharpe:.2f}, Win rate={win_rate*100:.1f}%")

        return results


def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='Walk-forward test XGBoost models')
    parser.add_argument('--horizon', type=int, default=1, help='Model horizon to test')
    parser.add_argument('--test-weeks', type=int, default=52, help='Test period in weeks')
    parser.add_argument('--num-positions', type=int, default=5, help='Number of positions to hold (per side for long/short)')
    parser.add_argument('--allow-short', action='store_true', help='Enable long/short strategy (default: long-only)')
    parser.add_argument('--no-recompute-technical', action='store_true', help='Skip technical indicator recomputation (faster but less realistic)')

    args = parser.parse_args()

    tester = XGBoostWalkForwardTester(recompute_technical=not args.no_recompute_technical)
    results = tester.run_test(args.horizon, args.test_weeks, args.num_positions, args.allow_short)

    print(f"\n{'='*80}")
    print("WALK-FORWARD TEST RESULTS")
    print(f"{'='*80}")
    print(f"Horizon: {results['horizon']} weeks")
    print(f"Test Period: {results['test_weeks']} weeks")
    print(f"Strategy: {results['strategy']}")
    if results['strategy'] == 'long-short':
        print(f"Portfolio Size: {results['num_positions']} longs + {results['num_positions']} shorts")
    else:
        print(f"Portfolio Size: {results['num_positions']} stocks (top predicted positive)")
    print()

    if 'test_rmse' in results and results['test_rmse']:
        print(f"--- Prediction Accuracy ---")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
        print(f"Direction Accuracy: {results['direction_accuracy']:.1f}%")
        print(f"Test Samples: {results['test_samples']}")
        print()

    if 'profitability' in results:
        prof = results['profitability']
        print(f"--- Portfolio Performance ---")
        print(f"Cumulative Return: {prof['cumulative_return']*100:>7.2f}%")
        print(f"Annualized Return: {prof['annualized_return']*100:>7.2f}%")
        print(f"Sharpe Ratio:      {prof['sharpe_ratio']:>7.2f}")
        print(f"Max Drawdown:      {prof['max_drawdown']*100:>7.2f}%")
        print(f"Win Rate:          {prof['win_rate']*100:>7.1f}%")
        print(f"Avg Period Return: {prof['avg_period_return']*100:>7.2f}%")
        print(f"Trading Periods:   {prof['num_periods']}")
        print()

        # long/short breakdown
        if 'num_long_positions' in prof:
            print(f"--- Long/Short Breakdown ---")
            print(f"Long Positions:    {prof['num_long_positions']}")
            print(f"  Avg Return:      {prof['avg_long_return']*100:>7.2f}%")
            print(f"  Win Rate:        {prof['long_win_rate']*100:>7.1f}%")
            print(f"Short Positions:   {prof['num_short_positions']}")
            print(f"  Avg Return:      {prof['avg_short_return']*100:>7.2f}%")
            print(f"  Win Rate:        {prof['short_win_rate']*100:>7.1f}%")
            print()

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
