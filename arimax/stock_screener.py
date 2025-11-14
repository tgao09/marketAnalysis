# !/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
from typing import Dict, Any, Tuple
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedARIMAXScreener:
    

    def __init__(self,
                 min_return: float = 0.01,
                 max_return: float = 0.5,
                 max_ci_width: float = 0.10,
                 min_directional_accuracy: float = 0.50,
                 max_rmse: float = 0.10,
                 min_sharpe_ratio: float = None,
                 profitability_weight: float = 1.0,
                 reliability_weight: float = 1.0,
                 risk_weight: float = 1.0,
                 sharpe_weight: float = 1.0,
                 use_valid_only: bool = True,
                 forecast_date: str = None):
        
        self.min_return = min_return
        self.max_return = max_return
        self.max_ci_width = max_ci_width
        self.min_directional_accuracy = min_directional_accuracy
        self.max_rmse = max_rmse
        self.min_sharpe_ratio = min_sharpe_ratio
        self.profitability_weight = profitability_weight
        self.reliability_weight = reliability_weight
        self.risk_weight = risk_weight
        self.sharpe_weight = sharpe_weight
        self.use_valid_only = use_valid_only
        self.forecast_date = pd.to_datetime(forecast_date) if forecast_date else None

    def load_data(self, forecast_file: str, model_summary_file: str, dataset_file: str = 'dataset/stock_dataset.csv') -> pd.DataFrame:
        
        # load forecasts
        if not os.path.exists(forecast_file):
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")

        forecasts = pd.read_csv(forecast_file)
        forecasts['future_date'] = pd.to_datetime(forecasts['future_date'])

        # load model summary
        if not os.path.exists(model_summary_file):
            raise FileNotFoundError(f"Model summary file not found: {model_summary_file}")

        model_summary = pd.read_csv(model_summary_file)

        # check if sharpe_ratio_3m already in forecasts (from forecast_arimax.py)
        sharpe_in_forecasts = 'sharpe_ratio_3m' in forecasts.columns

        # load sharpe ratios from dataset only if not in forecasts
        if not sharpe_in_forecasts and os.path.exists(dataset_file):
            logger.info(f"Loading Sharpe ratios from {dataset_file}")
            dataset = pd.read_csv(dataset_file)
            dataset['Date'] = pd.to_datetime(dataset['Date'], utc=True)

            # get latest sharpe_ratio_3m for each ticker
            if 'sharpe_ratio_3m' in dataset.columns:
                latest_sharpe = dataset.groupby('ticker').last()[['sharpe_ratio_3m']].reset_index()
                logger.info(f"Loaded Sharpe ratios for {len(latest_sharpe)} stocks")
            else:
                logger.warning(f"sharpe_ratio_3m column not found in {dataset_file}")
                latest_sharpe = pd.DataFrame(columns=['ticker', 'sharpe_ratio_3m'])
        elif sharpe_in_forecasts:
            logger.info(f"Using sharpe_ratio_3m from forecast file ({forecasts['sharpe_ratio_3m'].notna().sum()} stocks)")
            latest_sharpe = pd.DataFrame()  # empty - already in forecasts
        else:
            logger.warning(f"Dataset file not found: {dataset_file} - Sharpe ratios will not be available")
            latest_sharpe = pd.DataFrame(columns=['ticker', 'sharpe_ratio_3m'])

        # validate columns
        required_forecast = ['ticker', 'future_date', 'predicted_return', 'ci_lower', 'ci_upper']
        required_summary = ['ticker', 'test_rmse', 'directional_accuracy']

        missing_forecast = [c for c in required_forecast if c not in forecasts.columns]
        missing_summary = [c for c in required_summary if c not in model_summary.columns]

        if missing_forecast:
            raise ValueError(f"Missing forecast columns: {missing_forecast}")
        if missing_summary:
            raise ValueError(f"Missing model summary columns: {missing_summary}")

        # merge on ticker
        df = forecasts.merge(
            model_summary[['ticker', 'test_rmse', 'test_mae', 'directional_accuracy', 'order', 'aic']],
            on='ticker',
            how='left'
        )

        # merge sharpe ratios only if loaded from dataset
        if not latest_sharpe.empty:
            df = df.merge(latest_sharpe, on='ticker', how='left')
            logger.info(f"Merged Sharpe ratios from dataset - {df['sharpe_ratio_3m'].notna().sum()} stocks have Sharpe data")

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

        # remove rows with missing critical data
        df = df.dropna(subset=['predicted_return', 'ci_lower', 'ci_upper', 'test_rmse', 'directional_accuracy'])

        if df.empty:
            raise ValueError("No valid data after merging and filtering")

        print(f"Loaded {len(df)} forecasts for {df['ticker'].nunique()} stocks")
        return df

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()

        # expected return (already in dataframe)
        df['expected_return'] = df['predicted_return']

        # uncertainty = confidence interval width
        df['uncertainty'] = df['ci_upper'] - df['ci_lower']
        df['uncertainty'] = np.maximum(df['uncertainty'], 0.0001)  # prevent division by zero

        # precision ratio: prediction magnitude relative to uncertainty
        df['precision_ratio'] = np.abs(df['expected_return']) / df['uncertainty']

        # risk-adjusted return (preserves sign for direction)
        df['risk_adj_return'] = df['expected_return'] / df['uncertainty']

        # directional confidence: proportion of ci on the correct side of zero
        # 1.0 when ci entirely supports direction, 0.0 when entirely opposes
        # for longs: what % of ci is above zero?
        # for shorts: what % of ci is below zero?

        def calc_directional_confidence(row):
            if row['expected_return'] > 0:  # long position
                if row['ci_lower'] > 0:
                    return 1.0  # entire ci is positive
                elif row['ci_upper'] < 0:
                    return 0.0  # entire ci is negative (wrong direction)
                else:
                    # ci crosses zero - proportion above zero
                    return row['ci_upper'] / row['uncertainty']
            else:  # short position
                if row['ci_upper'] < 0:
                    return 1.0  # entire ci is negative
                elif row['ci_lower'] > 0:
                    return 0.0  # entire ci is positive (wrong direction)
                else:
                    # ci crosses zero - proportion below zero
                    return -row['ci_lower'] / row['uncertainty']

        df['directional_confidence'] = df.apply(calc_directional_confidence, axis=1)

        # ===== profitability score =====
        # combines magnitude, precision, and directional confidence
        df['profitability_score'] = (
            np.abs(df['expected_return']) *           # return magnitude
            (1 + df['precision_ratio']) *              # precision bonus
            (1 + df['directional_confidence'])         # direction confidence bonus
        )

        # ===== reliability score (from historical performance) =====
        df['hist_accuracy'] = df['directional_accuracy'] / 100.0
        df['rmse_score'] = 1 / (1 + df['test_rmse'])

        # quality tiers based on historical accuracy
        df['quality_tier'] = pd.cut(
            df['directional_accuracy'],
            bins=[0, 55, 65, 100],
            labels=['C', 'B', 'A'],
            include_lowest=True
        )

        df['reliability_score'] = df['hist_accuracy'] * df['rmse_score']

        # normalize reliability to 0-1
        r_min, r_max = df['reliability_score'].min(), df['reliability_score'].max()
        if r_max > r_min:
            df['reliability_score'] = (df['reliability_score'] - r_min) / (r_max - r_min)

        # ===== risk score (combined forecast + historical) =====
        # lower combined uncertainty = higher risk score
        df['combined_uncertainty'] = (df['uncertainty'] + df['test_rmse']) / 2.0
        df['risk_score'] = 1 / (1 + df['combined_uncertainty'])

        # normalize risk to 0-1
        rs_min, rs_max = df['risk_score'].min(), df['risk_score'].max()
        if rs_max > rs_min:
            df['risk_score'] = (df['risk_score'] - rs_min) / (rs_max - rs_min)

        # ===== sharpe ratio (3-month rolling) =====
        # use raw sharpe ratio directly - no transformation needed
        # sharpe ratio is already interpretable: higher is better regardless of long/short
        # for shorts, you'd invert the predicted return, not the sharpe ratio
        if 'sharpe_ratio_3m' in df.columns:
            df['sharpe_score'] = df['sharpe_ratio_3m']
            logger.info(f"Sharpe ratio range: {df['sharpe_score'].min():.3f} to {df['sharpe_score'].max():.3f}")
        else:
            # if sharpe_ratio_3m not available, default to 0 (neutral)
            df['sharpe_score'] = 0.0
            logger.warning("sharpe_ratio_3m not found in data - using neutral Sharpe of 0.0")

        # ===== composite opportunity score =====
        # profitability^w1 x reliability^w2 x risk^w3 x sharpe^w4
        # note: sharpe can be negative, so we don't use it in the multiplicative score
        # instead, use it as a filter or display metric
        df['opportunity_score'] = (
            (df['profitability_score'] ** self.profitability_weight) *
            (df['reliability_score'] ** self.reliability_weight) *
            (df['risk_score'] ** self.risk_weight)
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
        low_uncertainty = df['uncertainty'] <= self.max_ci_width
        print(f"After uncertainty filter (<={self.max_ci_width*100:.1f}% CI width): {low_uncertainty.sum()}")

        # 4. minimum directional accuracy filter
        good_accuracy = df['directional_accuracy'] >= (self.min_directional_accuracy * 100)
        print(f"After accuracy filter (>={self.min_directional_accuracy*100:.0f}%): {good_accuracy.sum()}")

        # 5. maximum rmse filter
        low_error = df['test_rmse'] <= self.max_rmse
        print(f"After RMSE filter (<={self.max_rmse*100:.1f}%): {low_error.sum()}")

        # 6. minimum sharpe ratio filter (if specified)
        if self.min_sharpe_ratio is not None and 'sharpe_ratio_3m' in df.columns:
            good_sharpe = df['sharpe_ratio_3m'] >= self.min_sharpe_ratio
            print(f"After Sharpe filter (>={self.min_sharpe_ratio:.2f}): {good_sharpe.sum()}")
        else:
            good_sharpe = pd.Series([True] * len(df), index=df.index)

        # combine all filters
        all_filters = realistic & min_magnitude & low_uncertainty & good_accuracy & low_error & good_sharpe

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
            'avg_accuracy': opportunities['directional_accuracy'].mean(),
            'avg_rmse': opportunities['test_rmse'].mean(),
            'avg_ci_width': opportunities['uncertainty'].mean(),
            'avg_opportunity_score': opportunities['opportunity_score'].mean(),
            'quality_distribution': opportunities['quality_tier'].value_counts().to_dict(),
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
                'accuracy': longs.loc[best_long, 'directional_accuracy'],
                'score': longs.loc[best_long, 'opportunity_score'],
                'tier': str(longs.loc[best_long, 'quality_tier'])
            }

        if len(shorts) > 0:
            best_short = shorts['opportunity_score'].idxmax()
            summary['best_short'] = {
                'ticker': shorts.loc[best_short, 'ticker'],
                'return': shorts.loc[best_short, 'expected_return'],
                'accuracy': shorts.loc[best_short, 'directional_accuracy'],
                'score': shorts.loc[best_short, 'opportunity_score'],
                'tier': str(shorts.loc[best_short, 'quality_tier'])
            }

        return summary

    def screen(self, forecast_file: str, model_summary_file: str, n: int = 10, dataset_file: str = 'dataset/stock_dataset.csv') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        
        df = self.load_data(forecast_file, model_summary_file, dataset_file)
        df = self.calculate_metrics(df)
        filtered = self.apply_filters(df)
        top = self.get_top_opportunities(filtered, n)
        summary = self.generate_summary(top)

        return top, summary


def format_output(opportunities: pd.DataFrame) -> None:
    
    if opportunities.empty:
        print("No opportunities meet the screening criteria.")
        return

    print(f"\n{'='*120}")
    print(f"{'TOP TRADING OPPORTUNITIES':^120}")
    print(f"{'='*120}")

    for idx, row in opportunities.iterrows():
        direction = "LONG " if row['expected_return'] > 0 else "SHORT"
        tier = str(row['quality_tier'])

        print(f"\n{direction} | {row['ticker']:>6} [Tier {tier}] | {row['future_date'].strftime('%Y-%m-%d')}")
        print(f"  Forecast:")
        print(f"    Expected Return:        {row['expected_return']*100:>7.2f}%")
        print(f"    Confidence Interval:    [{row['ci_lower']*100:>6.2f}%, {row['ci_upper']*100:>6.2f}%]")
        print(f"    CI Width:               {row['uncertainty']*100:>7.2f}%")
        print(f"    Precision Ratio:        {row['precision_ratio']:>7.2f}")
        print(f"    Directional Confidence: {row['directional_confidence']*100:>7.1f}%")
        print(f"  Historical Performance:")
        print(f"    Directional Accuracy:   {row['directional_accuracy']:>7.1f}%")
        print(f"    Test RMSE:              {row['test_rmse']*100:>7.2f}%")
        if 'sharpe_ratio_3m' in row and pd.notna(row['sharpe_ratio_3m']):
            print(f"    Sharpe Ratio (3m):      {row['sharpe_ratio_3m']:>7.3f}")
        print(f"  Scores:")
        print(f"    Profitability:          {row['profitability_score']:>7.4f}")
        print(f"    Reliability:            {row['reliability_score']:>7.4f}")
        print(f"    Risk:                   {row['risk_score']:>7.4f}")
        print(f"    OPPORTUNITY SCORE:      {row['opportunity_score']:>7.4f}")

        if 'model_order' in row or 'order' in row:
            order = row.get('model_order', row.get('order', 'N/A'))
            print(f"  Model:                    ARIMA{order}")


def main():
    parser = argparse.ArgumentParser(
        description='Screen ARIMAX forecasts for optimal trading opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic screening with defaults
  python stock_screener.py future_forecasts.csv model_summary.csv

  # Top 5 opportunities with minimum 2% return
  python stock_screener.py future_forecasts.csv model_summary.csv -n 5 -m 0.02

  # Conservative: max 5% uncertainty, min 60% accuracy
  python stock_screener.py future_forecasts.csv model_summary.csv -u 0.05 -a 0.60

  # Custom scoring weights (emphasize reliability 2x)
  python stock_screener.py future_forecasts.csv model_summary.csv --reliability-weight 2.0

  # Filter to specific date
  python stock_screener.py future_forecasts.csv model_summary.csv -d 2025-10-24

  # Export results to CSV
  python stock_screener.py future_forecasts.csv model_summary.csv --export trades.csv
        """
    )

    parser.add_argument('forecast_file', help='Path to future_forecasts CSV file')
    parser.add_argument('model_summary_file', nargs='?', default='arimax/arimaxresults/model_summary.csv',
                       help='Path to model_summary CSV file (default: arimax/arimaxresults/model_summary.csv)')
    parser.add_argument('-n', '--num', type=int, default=10,
                       help='Number of top opportunities (default: 10)')
    parser.add_argument('-m', '--min-return', type=float, default=0.01,
                       help='Minimum absolute return (default: 0.01 = 1%%)')
    parser.add_argument('-r', '--max-return', type=float, default=0.5,
                       help='Maximum return to filter outliers (default: 0.5 = 50%%)')
    parser.add_argument('-u', '--max-ci-width', type=float, default=0.10,
                       help='Maximum CI width (default: 0.10 = 10%%)')
    parser.add_argument('-a', '--min-accuracy', type=float, default=0.50,
                       help='Minimum historical directional accuracy (default: 0.50 = 50%%)')
    parser.add_argument('--max-rmse', type=float, default=0.10,
                       help='Maximum historical RMSE (default: 0.10 = 10%%)')
    parser.add_argument('--min-sharpe', type=float, default=None,
                       help='Minimum 3-month Sharpe ratio (default: None = no filter)')
    parser.add_argument('--profitability-weight', type=float, default=1.0,
                       help='Weight for profitability score (default: 1.0)')
    parser.add_argument('--reliability-weight', type=float, default=1.0,
                       help='Weight for reliability score (default: 1.0)')
    parser.add_argument('--risk-weight', type=float, default=1.0,
                       help='Weight for risk score (default: 1.0)')
    parser.add_argument('--sharpe-weight', type=float, default=1.0,
                       help='Weight for Sharpe quality score (default: 1.0)')
    parser.add_argument('--dataset-file', type=str, default='dataset/stock_dataset.csv',
                       help='Path to dataset CSV with Sharpe ratios (default: dataset/stock_dataset.csv)')
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
        screener = EnhancedARIMAXScreener(
            min_return=args.min_return,
            max_return=args.max_return,
            max_ci_width=args.max_ci_width,
            min_directional_accuracy=args.min_accuracy,
            max_rmse=args.max_rmse,
            min_sharpe_ratio=args.min_sharpe,
            profitability_weight=args.profitability_weight,
            reliability_weight=args.reliability_weight,
            risk_weight=args.risk_weight,
            sharpe_weight=args.sharpe_weight,
            use_valid_only=not args.include_invalid,
            forecast_date=args.date
        )

        # run screening
        top, summary = screener.screen(args.forecast_file, args.model_summary_file, args.num, args.dataset_file)

        # display results
        if not args.summary_only:
            format_output(top)

        # display summary
        print(f"\n{'='*120}")
        print(f"{'SUMMARY':^120}")
        print(f"{'='*120}")

        if 'message' in summary:
            print(summary['message'])
        else:
            print(f"Total Opportunities:           {summary['total']}")
            print(f"  Long Positions:              {summary['long']}")
            print(f"  Short Positions:             {summary['short']}")
            print(f"Average Expected Return:       {summary['avg_return']*100:>6.2f}%")
            print(f"Average Historical Accuracy:   {summary['avg_accuracy']:>6.1f}%")
            print(f"Average Historical RMSE:       {summary['avg_rmse']*100:>6.2f}%")
            print(f"Average CI Width:              {summary['avg_ci_width']*100:>6.2f}%")
            print(f"Average Opportunity Score:     {summary['avg_opportunity_score']:>6.4f}")
            print(f"Return Range:                  {summary['return_range']['min']*100:>6.2f}% to {summary['return_range']['max']*100:>6.2f}%")
            print(f"Next Trading Date:             {summary['next_date']}")

            if 'quality_distribution' in summary:
                print(f"\nQuality Distribution:")
                for tier in ['A', 'B', 'C']:
                    count = summary['quality_distribution'].get(tier, 0)
                    if count > 0:
                        print(f"  Tier {tier}: {count} stocks")

            if 'best_long' in summary:
                b = summary['best_long']
                print(f"\nBest Long:  {b['ticker']} [Tier {b['tier']}] (Return: {b['return']*100:.2f}%, "
                      f"Accuracy: {b['accuracy']:.1f}%, Score: {b['score']:.4f})")

            if 'best_short' in summary:
                b = summary['best_short']
                print(f"Best Short: {b['ticker']} [Tier {b['tier']}] (Return: {b['return']*100:.2f}%, "
                      f"Accuracy: {b['accuracy']:.1f}%, Score: {b['score']:.4f})")

        # export if requested
        if args.export and not top.empty:
            export_cols = ['ticker', 'future_date', 'expected_return', 'ci_lower', 'ci_upper',
                          'uncertainty', 'precision_ratio', 'directional_confidence',
                          'directional_accuracy', 'test_rmse', 'quality_tier',
                          'profitability_score', 'reliability_score', 'risk_score', 'opportunity_score']
            # only include columns that exist
            export_cols = [c for c in export_cols if c in top.columns]
            top[export_cols].to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

        print(f"{'='*120}\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
