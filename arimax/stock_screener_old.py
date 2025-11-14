# !/usr/bin/env python3


import pandas as pd
import numpy as np
import argparse
import sys
from typing import Dict, Any, Tuple
import os

class ARIMAXScreener:
    

    def __init__(self,
                 min_return: float = 0.01,
                 max_return: float = 0.5,
                 max_uncertainty: float = 0.1,
                 min_sharpe: float = 0.0,
                 use_valid_only: bool = True,
                 forecast_date: str = None):
        
        self.min_return = min_return
        self.max_return = max_return
        self.max_uncertainty = max_uncertainty
        self.min_sharpe = min_sharpe
        self.use_valid_only = use_valid_only
        self.forecast_date = pd.to_datetime(forecast_date) if forecast_date else None

    def load_forecasts(self, filepath: str) -> pd.DataFrame:
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Forecast file not found: {filepath}")

        df = pd.read_csv(filepath)

        # validate required columns
        required = ['ticker', 'future_date', 'predicted_return', 'ci_lower', 'ci_upper']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # convert date column
        df['future_date'] = pd.to_datetime(df['future_date'])

        # filter to valid forecasts
        if self.use_valid_only and 'forecast_valid' in df.columns:
            initial = len(df)
            df = df[df['forecast_valid'] == True].copy()
            print(f"Valid forecasts: {len(df)}/{initial} ({100*len(df)/initial:.1f}%)")

        # filter by date if specified
        if self.forecast_date is not None:
            initial = len(df)
            df = df[df['future_date'] == self.forecast_date].copy()
            print(f"Filtered to {self.forecast_date.strftime('%Y-%m-%d')}: {len(df)}/{initial} forecasts")

        # remove nan values
        df = df.dropna(subset=['predicted_return', 'ci_lower', 'ci_upper'])

        if df.empty:
            raise ValueError("No valid forecast data after filtering")

        return df

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()

        # expected return (already in dataframe)
        df['expected_return'] = df['predicted_return']

        # uncertainty = confidence interval width
        df['uncertainty'] = df['ci_upper'] - df['ci_lower']
        df['uncertainty'] = np.maximum(df['uncertainty'], 0.0001)  # prevent division by zero

        # sharpe-like ratio: return per unit of uncertainty
        df['sharpe_ratio'] = df['expected_return'] / df['uncertainty']

        # conviction: how many standard errors away from zero
        # high conviction = prediction far from zero relative to uncertainty
        df['conviction'] = np.abs(df['expected_return']) / df['uncertainty']

        # directional strength: is the entire ci on one side of zero?
        df['directional_strength'] = np.where(
            df['expected_return'] > 0,
            np.where(df['ci_lower'] > 0, 1.0, np.maximum(0, df['expected_return'] / df['uncertainty'])),
            np.where(df['ci_upper'] < 0, 1.0, np.maximum(0, -df['expected_return'] / df['uncertainty']))
        )

        # overall score: combination of magnitude and conviction
        # prioritizes large moves with low uncertainty
        df['opportunity_score'] = (
            np.abs(df['expected_return']) *     # magnitude of opportunity
            df['conviction'] *                    # conviction strength
            (1 + df['directional_strength'])     # bonus for clear direction
        )

        return df

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        

        print(f"\nStarting with {len(df)} forecasts...")

        # 1. filter unrealistic outliers
        realistic = np.abs(df['expected_return']) <= self.max_return
        outliers = len(df) - realistic.sum()
        if outliers > 0:
            print(f"Removed {outliers} outliers (>{self.max_return*100:.0f}% predicted return)")

        # 2. minimum magnitude filter
        min_magnitude = np.abs(df['expected_return']) >= self.min_return
        print(f"After magnitude filter (>={self.min_return*100:.1f}%): {min_magnitude.sum()}")

        # 3. maximum uncertainty filter
        low_uncertainty = df['uncertainty'] <= self.max_uncertainty
        print(f"After uncertainty filter (<={self.max_uncertainty*100:.1f}% CI width): {low_uncertainty.sum()}")

        # 4. minimum sharpe filter
        good_sharpe = df['sharpe_ratio'] >= self.min_sharpe
        print(f"After Sharpe filter (>={self.min_sharpe:.2f}): {good_sharpe.sum()}")

        # combine all filters
        all_filters = realistic & min_magnitude & low_uncertainty & good_sharpe
        filtered = df[all_filters].copy()

        print(f"Final qualified opportunities: {len(filtered)}\n")
        return filtered

    def get_top_opportunities(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        
        if df.empty:
            return df

        # get earliest forecast for each ticker (most immediate opportunity)
        earliest = df.loc[df.groupby('ticker')['future_date'].idxmin()]

        # rank by opportunity score
        top = earliest.nlargest(n, 'opportunity_score')

        return top

    def generate_summary(self, opportunities: pd.DataFrame) -> Dict[str, Any]:
        
        if opportunities.empty:
            return {'message': 'No opportunities meet screening criteria'}

        longs = opportunities[opportunities['expected_return'] > 0]
        shorts = opportunities[opportunities['expected_return'] < 0]

        summary = {
            'total': len(opportunities),
            'long': len(longs),
            'short': len(shorts),
            'avg_return': opportunities['expected_return'].mean(),
            'avg_sharpe': opportunities['sharpe_ratio'].mean(),
            'avg_conviction': opportunities['conviction'].mean(),
            'avg_uncertainty': opportunities['uncertainty'].mean(),
            'return_range': {
                'min': opportunities['expected_return'].min(),
                'max': opportunities['expected_return'].max()
            },
            'next_date': opportunities['future_date'].min().strftime('%Y-%m-%d')
        }

        if len(longs) > 0:
            best_long = longs['opportunity_score'].idxmax()
            summary['best_long'] = {
                'ticker': longs.loc[best_long, 'ticker'],
                'return': longs.loc[best_long, 'expected_return'],
                'sharpe': longs.loc[best_long, 'sharpe_ratio'],
                'conviction': longs.loc[best_long, 'conviction']
            }

        if len(shorts) > 0:
            best_short = shorts['opportunity_score'].idxmax()
            summary['best_short'] = {
                'ticker': shorts.loc[best_short, 'ticker'],
                'return': shorts.loc[best_short, 'expected_return'],
                'sharpe': shorts.loc[best_short, 'sharpe_ratio'],
                'conviction': shorts.loc[best_short, 'conviction']
            }

        return summary

    def screen(self, filepath: str, n: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        df = self.load_forecasts(filepath)
        df = self.calculate_metrics(df)
        filtered = self.apply_filters(df)
        top = self.get_top_opportunities(filtered, n)
        summary = self.generate_summary(top)

        return top, summary


def format_output(opportunities: pd.DataFrame) -> None:
    
    if opportunities.empty:
        print("No opportunities meet the screening criteria.")
        return

    print(f"\n{'='*100}")
    print(f"{'TOP TRADING OPPORTUNITIES':^100}")
    print(f"{'='*100}")

    for idx, row in opportunities.iterrows():
        direction = "LONG " if row['expected_return'] > 0 else "SHORT"

        print(f"\n{direction} | {row['ticker']:>6} | {row['future_date'].strftime('%Y-%m-%d')}")
        print(f"       | Expected Return:    {row['expected_return']*100:>7.2f}%")
        print(f"       | Confidence Int:     [{row['ci_lower']*100:>6.2f}%, {row['ci_upper']*100:>6.2f}%]")
        print(f"       | Uncertainty:        {row['uncertainty']*100:>7.2f}%")
        print(f"       | Sharpe Ratio:       {row['sharpe_ratio']:>7.2f}")
        print(f"       | Conviction:         {row['conviction']:>7.2f}x")
        print(f"       | Opportunity Score:  {row['opportunity_score']:>7.4f}")

        if 'model_order' in row:
            print(f"       | Model:              ARIMA{row['model_order']}")


def main():
    parser = argparse.ArgumentParser(
        description='Screen ARIMAX forecasts for optimal trading opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic screening with defaults
  python stock_screener.py future_forecasts_20251013_091557.csv

  # Top 5 opportunities with minimum 2% return
  python stock_screener.py future_forecasts_20251013_091557.csv -n 5 -m 0.02

  # Conservative: max 5% uncertainty, min Sharpe 2.0
  python stock_screener.py future_forecasts_20251013_091557.csv -u 0.05 -s 2.0

  # Filter to specific date
  python stock_screener.py future_forecasts_20251013_091557.csv -d 2025-10-17

  # Export results to CSV
  python stock_screener.py future_forecasts_20251013_091557.csv --export trades.csv
        """
    )

    parser.add_argument('filepath', help='Path to forecast CSV file')
    parser.add_argument('-n', '--num', type=int, default=10,
                       help='Number of top opportunities (default: 10)')
    parser.add_argument('-m', '--min-return', type=float, default=0.01,
                       help='Minimum absolute return (default: 0.01 = 1%%)')
    parser.add_argument('-r', '--max-return', type=float, default=0.5,
                       help='Maximum return to filter outliers (default: 0.5 = 50%%)')
    parser.add_argument('-u', '--max-uncertainty', type=float, default=0.1,
                       help='Maximum CI width (default: 0.1 = 10%%)')
    parser.add_argument('-s', '--min-sharpe', type=float, default=0.0,
                       help='Minimum Sharpe ratio (default: 0.0)')
    parser.add_argument('-d', '--date', type=str,
                       help='Filter to specific date (YYYY-MM-DD)')
    parser.add_argument('--include-invalid', action='store_true',
                       help='Include forecasts marked as invalid')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary statistics')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')

    args = parser.parse_args()

    try:
        # initialize screener
        screener = ARIMAXScreener(
            min_return=args.min_return,
            max_return=args.max_return,
            max_uncertainty=args.max_uncertainty,
            min_sharpe=args.min_sharpe,
            use_valid_only=not args.include_invalid,
            forecast_date=args.date
        )

        # run screening
        top, summary = screener.screen(args.filepath, args.num)

        # display results
        if not args.summary_only:
            format_output(top)

        # display summary
        print(f"\n{'='*100}")
        print(f"{'SUMMARY':^100}")
        print(f"{'='*100}")

        if 'message' in summary:
            print(summary['message'])
        else:
            print(f"Total Opportunities:      {summary['total']}")
            print(f"  Long Positions:         {summary['long']}")
            print(f"  Short Positions:        {summary['short']}")
            print(f"Average Expected Return:  {summary['avg_return']*100:>6.2f}%")
            print(f"Average Sharpe Ratio:     {summary['avg_sharpe']:>6.2f}")
            print(f"Average Conviction:       {summary['avg_conviction']:>6.2f}x")
            print(f"Average Uncertainty:      {summary['avg_uncertainty']*100:>6.2f}%")
            print(f"Return Range:             {summary['return_range']['min']*100:>6.2f}% to {summary['return_range']['max']*100:>6.2f}%")
            print(f"Next Trading Date:        {summary['next_date']}")

            if 'best_long' in summary:
                b = summary['best_long']
                print(f"\nBest Long:  {b['ticker']} (Return: {b['return']*100:.2f}%, "
                      f"Sharpe: {b['sharpe']:.2f}, Conviction: {b['conviction']:.2f}x)")

            if 'best_short' in summary:
                b = summary['best_short']
                print(f"Best Short: {b['ticker']} (Return: {b['return']*100:.2f}%, "
                      f"Sharpe: {b['sharpe']:.2f}, Conviction: {b['conviction']:.2f}x)")

        # export if requested
        if args.export and not top.empty:
            export_cols = ['ticker', 'future_date', 'expected_return', 'ci_lower', 'ci_upper',
                          'uncertainty', 'sharpe_ratio', 'conviction', 'opportunity_score']
            # only include columns that exist
            export_cols = [c for c in export_cols if c in top.columns]
            top[export_cols].to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

        print(f"{'='*100}\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
