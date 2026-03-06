

import pandas as pd
import numpy as np
import argparse
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostScreener:
    

    def __init__(self, min_return: float = 0.0, max_return: float = 0.5,
                 min_sharpe: float = None, min_weekly_return: float = 0.0):
        
        self.min_return = min_return
        self.max_return = max_return
        self.min_sharpe = min_sharpe
        self.min_weekly_return = min_weekly_return

    def calculate_ensemble_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        
        signals = []

        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].copy()

            # get forecasts by horizon
            horizons = ticker_df.set_index('horizon_weeks')['predicted_return'].to_dict()

            # weekly trading strategy: prioritize 1-week forecast
            # 1-week: 80% (primary signal), 2-week: 15%, 4-week: 5% (directional check)
            weights = {1: 0.80, 2: 0.15, 4: 0.05}

            ensemble_return = 0
            total_weight = 0

            for horizon, weight in weights.items():
                if horizon in horizons:
                    ensemble_return += horizons[horizon] * weight
                    total_weight += weight

            # normalize by actual weights used
            if total_weight > 0:
                ensemble_return /= total_weight

            # signal strength: consistency across horizons
            if len(horizons) > 1:
                returns_list = list(horizons.values())
                direction_agreement = np.mean([np.sign(r) == np.sign(ensemble_return) for r in returns_list])
            else:
                direction_agreement = 1.0

            # get latest sharpe ratio
            sharpe = ticker_df['sharpe_ratio_3m'].iloc[0] if 'sharpe_ratio_3m' in ticker_df.columns else None

            signals.append({
                'ticker': ticker,
                'ensemble_return': ensemble_return,
                'direction_agreement': direction_agreement,
                'signal_strength': abs(ensemble_return) * direction_agreement,
                'weekly_return': horizons.get(1, 0),  # primary: 1-week forecast
                'biweekly_return': horizons.get(2, 0),  # secondary: directional confirmation
                'sharpe_ratio_3m': sharpe,
                'num_horizons': len(horizons)
            })

        return pd.DataFrame(signals)

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info(f"Starting with {len(df)} stocks")

        # remove extreme outliers only
        df = df[abs(df['ensemble_return']) <= self.max_return]
        logger.info(f"After outlier filter: {len(df)}")

        # minimum return magnitude (only if specified)
        if self.min_return > 0:
            df = df[abs(df['ensemble_return']) >= self.min_return]
            logger.info(f"After minimum return filter: {len(df)}")

        # weekly trading filter: 1-week forecast must align with ensemble direction
        # since 1-week is 80% of ensemble, this is effectively checking directional consistency
        if self.min_weekly_return > 0:
            # strict: 1-week forecast must exceed minimum threshold
            long_candidates = df[(df['ensemble_return'] > 0) &
                                (df['weekly_return'] >= self.min_weekly_return)]
            short_candidates = df[(df['ensemble_return'] < 0) &
                                 (df['weekly_return'] <= -self.min_weekly_return)]
        else:
            # directional filter: 1-week must agree with ensemble
            long_candidates = df[(df['ensemble_return'] > 0) & (df['weekly_return'] > 0)]
            short_candidates = df[(df['ensemble_return'] < 0) & (df['weekly_return'] < 0)]

        df = pd.concat([long_candidates, short_candidates], ignore_index=True)
        logger.info(f"After weekly directional filter: {len(df)}")

        # sharpe ratio filter (only if specified)
        if self.min_sharpe is not None and 'sharpe_ratio_3m' in df.columns:
            df = df[df['sharpe_ratio_3m'].fillna(-999) >= self.min_sharpe]
            logger.info(f"After Sharpe filter: {len(df)}")

        return df

    def rank_opportunities(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        
        # rank by signal strength
        df = df.sort_values('signal_strength', ascending=False)

        # separate longs and shorts
        longs = df[df['ensemble_return'] > 0].head(n // 2)
        shorts = df[df['ensemble_return'] < 0].head(n // 2)

        top = pd.concat([longs, shorts], ignore_index=True)

        return top.sort_values('signal_strength', ascending=False)

    def screen(self, forecast_file: str, n: int = 20) -> pd.DataFrame:
        
        df = pd.read_csv(forecast_file)

        # calculate ensemble signals
        signals = self.calculate_ensemble_signal(df)

        # apply filters
        filtered = self.apply_filters(signals)

        # rank
        top = self.rank_opportunities(filtered, n)

        return top


def main():
    parser = argparse.ArgumentParser(description='Screen XGBoost forecasts for weekly trading')
    parser.add_argument('forecast_file', help='Path to forecasts CSV')
    parser.add_argument('-n', '--num', type=int, default=20, help='Number of top opportunities')
    parser.add_argument('-m', '--min-return', type=float, default=0.003, help='Minimum return (default 0.3% for weekly trading)')
    parser.add_argument('--min-sharpe', type=float, default=None, help='Minimum Sharpe ratio')
    parser.add_argument('--min-weekly-return', type=float, default=0.0, help='Minimum 1-week return threshold')

    args = parser.parse_args()

    screener = XGBoostScreener(
        min_return=args.min_return,
        min_sharpe=args.min_sharpe,
        min_weekly_return=args.min_weekly_return
    )

    top = screener.screen(args.forecast_file, args.num)

    print(f"\n{'='*80}")
    print("TOP TRADING OPPORTUNITIES - Weekly Trading (Open Mon, Close Fri)")
    print(f"{'='*80}\n")

    for _, row in top.iterrows():
        direction = "LONG " if row['ensemble_return'] > 0 else "SHORT"
        print(f"{direction} | {row['ticker']:>6} | 1W: {row['weekly_return']*100:>6.2f}% | "
              f"Ensemble: {row['ensemble_return']*100:>6.2f}% | Strength: {row['signal_strength']:.4f}")

    print(f"\n{'='*80}")
    print(f"Total: {len(top)} opportunities ({(top['ensemble_return'] > 0).sum()} longs, {(top['ensemble_return'] < 0).sum()} shorts)")
    print(f"Min return filter: {args.min_return*100:.2f}% | Sharpe filter: {args.min_sharpe if args.min_sharpe else 'None'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
