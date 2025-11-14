

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalFeatureEngineer:
    

    def __init__(self, lookback_short: int = 5, lookback_medium: int = 20,
                 lookback_long: int = 50):
        
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long

    def calculate_rsi(self, returns: pd.Series, period: int = 14) -> pd.Series:
        
        # separate gains and losses
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)

        # calculate exponential moving averages
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()

        # avoid division by zero
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(self, returns: pd.Series,
                                  period: int = 20,
                                  num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        
        middle = returns.rolling(window=period, min_periods=period).mean()
        std = returns.rolling(window=period, min_periods=period).std()

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # calculate current position within bands (0-1 scale)
        # 0 = at lower band, 1 = at upper band
        band_width = upper - lower
        bb_position = (returns - lower) / (band_width + 1e-10)
        bb_position = bb_position.clip(0, 1)  # constrain to [0, 1]

        return upper, middle, lower, bb_position

    def calculate_macd(self, cumulative_returns: pd.Series,
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        
        ema_fast = cumulative_returns.ewm(span=fast, adjust=False).mean()
        ema_slow = cumulative_returns.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_atr(self, high_return: pd.Series, low_return: pd.Series,
                     weekly_return: pd.Series, period: int = 14) -> pd.Series:
        
        # true range components
        high_low = high_return - low_return
        high_close = np.abs(high_return - weekly_return.shift(1))
        low_close = np.abs(low_return - weekly_return.shift(1))

        # true range is max of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # atr is ema of true range
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_money_flow_index(self, high_return: pd.Series, low_return: pd.Series,
                                   close_return: pd.Series, volume_change: pd.Series,
                                   period: int = 14) -> pd.Series:
        
        # typical price (average of high, low, close)
        typical_price = (high_return + low_return + close_return) / 3

        # money flow = typical price × volume (use volume_change as proxy)
        # add 1 to volume_change to avoid negative values
        money_flow = typical_price * (1 + volume_change)

        # separate positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        # sum over period
        positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()

        # money flow ratio
        mf_ratio = positive_mf / (negative_mf + 1e-10)

        # mfi
        mfi = 100 - (100 / (1 + mf_ratio))

        return mfi

    def calculate_rate_of_change(self, returns: pd.Series, period: int) -> pd.Series:
        
        cumulative_returns = (1 + returns).cumprod()
        roc = (cumulative_returns / cumulative_returns.shift(period)) - 1
        return roc

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Calculating technical indicators...")

        result_dfs = []

        for ticker in df['ticker'].unique():
            stock_df = df[df['ticker'] == ticker].copy()
            stock_df = stock_df.sort_values('Date')

            # cumulative returns for macd (log price proxy)
            stock_df['cum_return'] = (1 + stock_df['weekly_return']).cumprod()
            stock_df['log_price_proxy'] = np.log(stock_df['cum_return'] + 1)

            # === momentum indicators ===
            # rsi on multiple timeframes
            stock_df['rsi_14'] = self.calculate_rsi(stock_df['weekly_return'], 14)
            stock_df['rsi_5'] = self.calculate_rsi(stock_df['weekly_return'], 5)

            # rate of change
            stock_df['roc_5'] = self.calculate_rate_of_change(stock_df['weekly_return'], 5)
            stock_df['roc_20'] = self.calculate_rate_of_change(stock_df['weekly_return'], 20)

            # === volatility indicators ===
            # bollinger bands
            bb_upper, bb_middle, bb_lower, bb_pos = self.calculate_bollinger_bands(
                stock_df['weekly_return'], period=20
            )
            stock_df['bb_position'] = bb_pos
            stock_df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)

            # average true range
            if all(col in stock_df.columns for col in ['high_return', 'low_return']):
                stock_df['atr_14'] = self.calculate_atr(
                    stock_df['high_return'], stock_df['low_return'],
                    stock_df['weekly_return'], period=14
                )

            # === trend indicators ===
            # macd
            macd_line, signal_line, macd_hist = self.calculate_macd(
                stock_df['log_price_proxy'], fast=12, slow=26, signal=9
            )
            stock_df['macd'] = macd_line
            stock_df['macd_signal'] = signal_line
            stock_df['macd_histogram'] = macd_hist

            # moving averages convergence/divergence
            stock_df['ema_5'] = stock_df['weekly_return'].ewm(span=5, adjust=False).mean()
            stock_df['ema_20'] = stock_df['weekly_return'].ewm(span=20, adjust=False).mean()
            stock_df['ema_cross'] = stock_df['ema_5'] - stock_df['ema_20']

            # === volume indicators ===
            if 'volume_change' in stock_df.columns:
                # money flow index
                if all(col in stock_df.columns for col in ['high_return', 'low_return']):
                    stock_df['mfi_14'] = self.calculate_money_flow_index(
                        stock_df['high_return'], stock_df['low_return'],
                        stock_df['weekly_return'], stock_df['volume_change'], period=14
                    )

                # volume momentum
                stock_df['volume_sma_5'] = stock_df['volume_change'].rolling(5, min_periods=1).mean()
                stock_df['volume_sma_20'] = stock_df['volume_change'].rolling(20, min_periods=1).mean()

            # === volatility regime ===
            # compare current volatility to historical average
            stock_df['vol_ratio'] = stock_df['volatility'] / (
                stock_df['volatility'].rolling(50, min_periods=10).mean() + 1e-10
            )

            # clean up intermediate columns
            stock_df = stock_df.drop(columns=['cum_return', 'log_price_proxy'], errors='ignore')

            result_dfs.append(stock_df)

        result = pd.concat(result_dfs, ignore_index=True)
        result = result.sort_values(['ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Added {len([c for c in result.columns if c not in df.columns])} technical features")

        return result

    def calculate_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Calculating cross-sectional features...")

        result = df.copy()

        # group by date for cross-sectional calculations
        for date in result['Date'].unique():
            date_mask = result['Date'] == date
            date_df = result[date_mask].copy()

            # === return rankings ===
            # rank returns vs peers (0-1 scale, 1 = best performer)
            result.loc[date_mask, 'return_rank'] = date_df['weekly_return'].rank(pct=True)

            # rank momentum vs peers
            if 'momentum_4w' in date_df.columns:
                result.loc[date_mask, 'momentum_4w_rank'] = date_df['momentum_4w'].rank(pct=True)
            if 'momentum_12w' in date_df.columns:
                result.loc[date_mask, 'momentum_12w_rank'] = date_df['momentum_12w'].rank(pct=True)

            # === volatility rankings ===
            if 'volatility' in date_df.columns:
                result.loc[date_mask, 'volatility_rank'] = date_df['volatility'].rank(pct=True)

            # === volume rankings ===
            if 'volume_change' in date_df.columns:
                result.loc[date_mask, 'volume_rank'] = date_df['volume_change'].rank(pct=True)

            # === sharpe ratio rankings ===
            if 'sharpe_ratio_3m' in date_df.columns:
                valid_sharpe = date_df[date_df['sharpe_ratio_3m'].notna()]
                if len(valid_sharpe) > 0:
                    result.loc[date_mask & result['sharpe_ratio_3m'].notna(), 'sharpe_rank'] = \
                        valid_sharpe['sharpe_ratio_3m'].rank(pct=True)

            # === z-scores (standardized cross-sectional) ===
            # return z-score vs peers
            return_mean = date_df['weekly_return'].mean()
            return_std = date_df['weekly_return'].std()
            if return_std > 0:
                result.loc[date_mask, 'return_zscore'] = \
                    (date_df['weekly_return'] - return_mean) / return_std

            # volatility z-score vs peers
            if 'volatility' in date_df.columns:
                vol_mean = date_df['volatility'].mean()
                vol_std = date_df['volatility'].std()
                if vol_std > 0:
                    result.loc[date_mask, 'volatility_zscore'] = \
                        (date_df['volatility'] - vol_mean) / vol_std

        logger.info(f"Added {len([c for c in result.columns if c not in df.columns])} cross-sectional features")

        return result

    def calculate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Calculating interaction features...")

        result = df.copy()

        # === return × volatility interactions ===
        if 'volatility' in result.columns:
            # risk-adjusted return
            result['return_per_volatility'] = result['weekly_return'] / (result['volatility'] + 1e-10)

            # return magnitude × volatility
            result['abs_return_x_vol'] = np.abs(result['weekly_return']) * result['volatility']

        # === momentum × volume interactions ===
        if 'momentum_4w' in result.columns and 'volume_change' in result.columns:
            # momentum confirmation by volume
            result['momentum_x_volume'] = result['momentum_4w'] * result['volume_change']

        # === rsi × momentum ===
        if 'rsi_14' in result.columns and 'momentum_12w' in result.columns:
            # overbought momentum vs oversold momentum
            result['rsi_x_momentum'] = (result['rsi_14'] - 50) * result['momentum_12w']

        # === volatility regime × return ===
        if 'vol_ratio' in result.columns:
            # different return behavior in high vs low vol regimes
            result['return_x_vol_regime'] = result['weekly_return'] * result['vol_ratio']

        # === rank interactions ===
        if 'return_rank' in result.columns and 'volume_rank' in result.columns:
            # strong return + strong volume = momentum confirmation
            result['return_rank_x_volume_rank'] = result['return_rank'] * result['volume_rank']

        logger.info(f"Added {len([c for c in result.columns if c not in df.columns])} interaction features")

        return result

    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Calculating time features...")

        result = df.copy()
        # date is already datetime from previous steps, no need to convert again

        # === calendar features ===
        result['week_of_month'] = result['Date'].dt.day // 7 + 1
        result['month'] = result['Date'].dt.month
        result['quarter'] = result['Date'].dt.quarter
        result['week_of_year'] = result['Date'].dt.isocalendar().week

        # === cyclical encoding (avoid linear bias) ===
        # month as sine/cosine for cyclical pattern
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

        # quarter as sine/cosine
        result['quarter_sin'] = np.sin(2 * np.pi * result['quarter'] / 4)
        result['quarter_cos'] = np.cos(2 * np.pi * result['quarter'] / 4)

        logger.info(f"Added {len([c for c in result.columns if c not in df.columns])} time features")

        return result

    def create_forward_returns(self, df: pd.DataFrame,
                              horizons: List[int] = [1, 2, 4, 8]) -> pd.DataFrame:
        
        logger.info(f"Creating forward returns for horizons: {horizons}")

        result_dfs = []

        for ticker in df['ticker'].unique():
            stock_df = df[df['ticker'] == ticker].copy()
            stock_df = stock_df.sort_values('Date').reset_index(drop=True)

            for horizon in horizons:
                # calculate forward cumulative return using direct indexing
                # this ensures no lookahead: row i uses returns from i+1 to i+horizon
                forward_returns = []

                for i in range(len(stock_df)):
                    # get next 'horizon' weeks of returns
                    future_window = stock_df['weekly_return'].iloc[i+1:i+1+horizon]

                    if len(future_window) == horizon:
                        # compute cumulative return: (1+r1)*(1+r2)*...*(1+rn) - 1
                        cum_return = (1 + future_window).prod() - 1
                        forward_returns.append(cum_return)
                    else:
                        # not enough future data (near end of series)
                        forward_returns.append(np.nan)

                stock_df[f'forward_return_{horizon}w'] = forward_returns

                # binary classification target (up/down)
                stock_df[f'forward_direction_{horizon}w'] = (
                    stock_df[f'forward_return_{horizon}w'] > 0
                ).astype(int)

            result_dfs.append(stock_df)

        result = pd.concat(result_dfs, ignore_index=True)
        result = result.sort_values(['ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Added {len(horizons)} forward return targets")

        return result

    def engineer_all_features(self, df: pd.DataFrame,
                             forward_horizons: List[int] = [1, 2, 4, 8]) -> pd.DataFrame:
        
        logger.info("Starting complete feature engineering pipeline...")

        # calculate features in order
        result = self.calculate_technical_features(df)
        result = self.calculate_cross_sectional_features(result)
        result = self.calculate_interaction_features(result)
        result = self.calculate_time_features(result)

        # create forward return targets
        result = self.create_forward_returns(result, forward_horizons)

        logger.info(f"Feature engineering complete: {result.shape[1]} total columns")

        return result


def load_and_engineer_features(input_file: str = 'dataset/stock_dataset_with_lags.csv',
                               output_file: str = 'xgboost/features_engineered.csv',
                               forward_horizons: List[int] = [1, 2, 4, 8]) -> pd.DataFrame:
    
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    # handle timezone-aware datetimes by converting to utc then removing timezone
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    logger.info(f"Loaded {len(df)} rows for {df['ticker'].nunique()} stocks")

    # initialize engineer
    engineer = TechnicalFeatureEngineer()

    # engineer features
    result = engineer.engineer_all_features(df, forward_horizons)

    # save
    result.to_csv(output_file, index=False)
    logger.info(f"Engineered features saved to {output_file}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Engineer features for XGBoost models')
    parser.add_argument('--input-file', type=str,
                       default='dataset/stock_dataset_with_lags.csv',
                       help='Input dataset with lags')
    parser.add_argument('--output-file', type=str,
                       default='xgboost/features_engineered.csv',
                       help='Output file for engineered features')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Forward return horizons to create')

    args = parser.parse_args()

    result_df = load_and_engineer_features(
        input_file=args.input_file,
        output_file=args.output_file,
        forward_horizons=args.horizons
    )

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total features: {result_df.shape[1]}")
    print(f"Total rows: {len(result_df)}")
    print(f"Stocks: {result_df['ticker'].nunique()}")
    print(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")

    # show feature categories
    technical_features = [c for c in result_df.columns if any(x in c for x in ['rsi', 'macd', 'bb', 'atr', 'mfi', 'ema', 'roc'])]
    cross_sectional = [c for c in result_df.columns if any(x in c for x in ['rank', 'zscore'])]
    interaction = [c for c in result_df.columns if '_x_' in c or 'per_' in c]
    time_features = [c for c in result_df.columns if any(x in c for x in ['week', 'month', 'quarter', '_sin', '_cos'])]
    forward_returns = [c for c in result_df.columns if 'forward_' in c]

    print(f"\nFeature Categories:")
    print(f"  Technical indicators: {len(technical_features)}")
    print(f"  Cross-sectional: {len(cross_sectional)}")
    print(f"  Interactions: {len(interaction)}")
    print(f"  Time features: {len(time_features)}")
    print(f"  Forward targets: {len(forward_returns)}")
    print(f"{'='*60}\n")
