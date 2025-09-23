#!/usr/bin/env python3
"""
Stock Screener for ARIMAX Predictions

Screens prediction datasets to identify the best trading opportunities
based on signal strength, confidence intervals, and risk-adjusted returns.
Compatible with future_forecasts_*.csv files from the ARIMAX pipeline.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from typing import List, Dict, Any, Tuple
import os

class StockScreener:
    """
    Screen ARIMAX predictions for optimal trading opportunities.
    """

    def __init__(self, min_magnitude: float = 0.01, max_ci_width: float = 0.5,
                 min_confidence_level: float = 0.0, use_valid_only: bool = True):
        """
        Initialize screener with filtering criteria.

        Args:
            min_magnitude: Minimum absolute predicted return (default: 0.01 = 1%)
            max_ci_width: Maximum confidence interval width (default: 0.5 = 50%)
            min_confidence_level: Minimum confidence level (prediction outside CI of zero)
            use_valid_only: Only use forecasts marked as valid (default: True)
        """
        self.min_magnitude = min_magnitude
        self.max_ci_width = max_ci_width
        self.min_confidence_level = min_confidence_level
        self.use_valid_only = use_valid_only

    def load_predictions(self, filepath: str) -> pd.DataFrame:
        """Load and validate prediction dataset."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prediction file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Check for required columns
        required_columns = ['ticker', 'future_date', 'predicted_return', 'ci_lower', 'ci_upper']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert future_date to datetime
        df['future_date'] = pd.to_datetime(df['future_date'])

        # Filter to valid forecasts if requested and column exists
        if self.use_valid_only and 'forecast_valid' in df.columns:
            initial_count = len(df)
            df = df[df['forecast_valid'] == True].copy()
            valid_count = len(df)
            print(f"Filtered to valid forecasts: {valid_count}/{initial_count} ({valid_count/initial_count*100:.1f}%)")

        # Remove any rows with NaN values in critical columns
        df = df.dropna(subset=['predicted_return', 'ci_lower', 'ci_upper'])

        if df.empty:
            raise ValueError("No valid prediction data found after filtering")

        return df

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate screening metrics for each prediction."""
        df = df.copy()

        # Confidence interval width
        df['ci_width'] = df['ci_upper'] - df['ci_lower']

        # Ensure CI width is positive
        df['ci_width'] = np.maximum(df['ci_width'], 0.001)  # Minimum width to avoid division by zero

        # Signal-to-noise ratio
        df['signal_to_noise'] = np.abs(df['predicted_return']) / df['ci_width']

        # Directional confidence - how confident we are in the direction
        # If CI includes zero, confidence is low
        df['direction_confidence'] = np.where(
            df['predicted_return'] > 0,
            np.where(df['ci_lower'] > 0, 1.0, np.maximum(0, df['predicted_return'] / df['ci_width'])),
            np.where(df['ci_upper'] < 0, 1.0, np.maximum(0, -df['predicted_return'] / df['ci_width']))
        )

        # Risk-adjusted magnitude
        df['risk_adjusted_return'] = df['predicted_return'] / df['ci_width']

        # Overall screening score (combination of magnitude, signal-to-noise, and direction confidence)
        df['screening_score'] = (
            np.abs(df['predicted_return']) *  # Magnitude
            df['signal_to_noise'] *           # Signal clarity
            (1 + df['direction_confidence'])  # Direction confidence boost
        )

        return df

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply screening filters to identify qualified positions."""

        print(f"Starting with {len(df)} predictions...")

        # Apply magnitude filter
        magnitude_filter = np.abs(df['predicted_return']) >= self.min_magnitude
        print(f"After magnitude filter (>= {self.min_magnitude}): {magnitude_filter.sum()}")

        # Apply CI width filter
        ci_width_filter = df['ci_width'] <= self.max_ci_width
        print(f"After CI width filter (<= {self.max_ci_width}): {ci_width_filter.sum()}")

        # Apply directional confidence filter (prediction doesn't include zero in CI)
        direction_filter = (
            ((df['predicted_return'] > 0) & (df['ci_lower'] > self.min_confidence_level)) |
            ((df['predicted_return'] < 0) & (df['ci_upper'] < -self.min_confidence_level))
        )
        print(f"After direction filter: {direction_filter.sum()}")

        # Combine all filters
        all_filters = magnitude_filter & ci_width_filter & direction_filter
        filtered_df = df[all_filters].copy()

        print(f"Final filtered count: {len(filtered_df)}")
        return filtered_df

    def get_top_positions(self, df: pd.DataFrame, n_positions: int = 10) -> pd.DataFrame:
        """Get top N positions ranked by screening score."""
        if df.empty:
            return df

        # Get the earliest prediction date for each stock (most immediate opportunity)
        earliest_predictions = df.loc[df.groupby('ticker')['future_date'].idxmin()]

        # Sort by screening score (descending)
        top_positions = earliest_predictions.nlargest(n_positions, 'screening_score')

        return top_positions

    def generate_position_summary(self, positions: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for selected positions."""
        if positions.empty:
            return {'message': 'No positions meet screening criteria'}

        long_positions = positions[positions['predicted_return'] > 0]
        short_positions = positions[positions['predicted_return'] < 0]

        summary = {
            'total_positions': len(positions),
            'long_positions': len(long_positions),
            'short_positions': len(short_positions),
            'avg_predicted_return': positions['predicted_return'].mean(),
            'avg_signal_to_noise': positions['signal_to_noise'].mean(),
            'avg_direction_confidence': positions['direction_confidence'].mean(),
            'return_range': {
                'min': positions['predicted_return'].min(),
                'max': positions['predicted_return'].max()
            },
            'next_trading_date': positions['future_date'].min().strftime('%Y-%m-%d')
        }

        if len(long_positions) > 0:
            best_long_idx = long_positions['screening_score'].idxmax()
            summary['top_long_pick'] = {
                'ticker': long_positions.loc[best_long_idx, 'ticker'],
                'predicted_return': long_positions.loc[best_long_idx, 'predicted_return'],
                'signal_to_noise': long_positions.loc[best_long_idx, 'signal_to_noise'],
                'direction_confidence': long_positions.loc[best_long_idx, 'direction_confidence']
            }

        if len(short_positions) > 0:
            best_short_idx = short_positions['screening_score'].idxmax()
            summary['top_short_pick'] = {
                'ticker': short_positions.loc[best_short_idx, 'ticker'],
                'predicted_return': short_positions.loc[best_short_idx, 'predicted_return'],
                'signal_to_noise': short_positions.loc[best_short_idx, 'signal_to_noise'],
                'direction_confidence': short_positions.loc[best_short_idx, 'direction_confidence']
            }

        return summary

    def screen_predictions(self, filepath: str, n_positions: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete screening workflow.

        Args:
            filepath: Path to predictions CSV file
            n_positions: Number of top positions to return

        Returns:
            Tuple of (top_positions_df, summary_dict)
        """
        # Load and prepare data
        df = self.load_predictions(filepath)
        df = self.calculate_metrics(df)

        # Apply filters
        filtered_df = self.apply_filters(df)

        # Get top positions
        top_positions = self.get_top_positions(filtered_df, n_positions)

        # Generate summary
        summary = self.generate_position_summary(top_positions)

        return top_positions, summary

def format_position_output(positions: pd.DataFrame) -> None:
    """Format and display position recommendations."""
    if positions.empty:
        print("No positions meet the screening criteria.")
        return

    print(f"\n{'='*90}")
    print("TOP TRADING OPPORTUNITIES")
    print(f"{'='*90}")

    for idx, row in positions.iterrows():
        direction = "LONG" if row['predicted_return'] > 0 else "SHORT"
        return_pct = row['predicted_return'] * 100
        confidence_pct = row['direction_confidence'] * 100

        print(f"\n{direction:>5} | {row['ticker']:>6} | Date: {row['future_date'].strftime('%Y-%m-%d')}")
        print(f"      | Predicted Return: {return_pct:>8.2f}%")
        print(f"      | Confidence Int:   [{row['ci_lower']*100:>7.2f}%, {row['ci_upper']*100:>7.2f}%]")
        print(f"      | Signal/Noise:     {row['signal_to_noise']:>8.2f}")
        print(f"      | Direction Conf:   {confidence_pct:>8.1f}%")
        print(f"      | Overall Score:    {row['screening_score']:>8.4f}")

def main():
    """Main screening application."""
    parser = argparse.ArgumentParser(
        description='Screen ARIMAX predictions for optimal trading opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic screening with default parameters
  python stock_screener.py future_forecasts_20250920_164543.csv

  # Get top 5 positions with custom magnitude threshold
  python stock_screener.py future_forecasts_20250920_164543.csv -n 5 -m 0.02

  # More conservative screening (tighter CI, higher confidence)
  python stock_screener.py future_forecasts_20250920_164543.csv -w 0.3 -c 0.01

  # Show only summary statistics
  python stock_screener.py future_forecasts_20250920_164543.csv --summary-only
        """
    )

    parser.add_argument('filepath', type=str, help='Path to predictions CSV file')
    parser.add_argument('-n', '--positions', type=int, default=10,
                       help='Number of top positions to return (default: 10)')
    parser.add_argument('-m', '--min-magnitude', type=float, default=0.01,
                       help='Minimum absolute predicted return (default: 0.01 = 1%%)')
    parser.add_argument('-w', '--max-ci-width', type=float, default=0.5,
                       help='Maximum confidence interval width (default: 0.5 = 50%%)')
    parser.add_argument('-c', '--min-confidence', type=float, default=0.0,
                       help='Minimum confidence level (default: 0.0)')
    parser.add_argument('--include-invalid', action='store_true',
                       help='Include forecasts marked as invalid')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary statistics')
    parser.add_argument('--export', type=str, default=None,
                       help='Export results to CSV file')

    args = parser.parse_args()

    try:
        # Initialize screener
        screener = StockScreener(
            min_magnitude=args.min_magnitude,
            max_ci_width=args.max_ci_width,
            min_confidence_level=args.min_confidence,
            use_valid_only=not args.include_invalid
        )

        # Screen predictions
        top_positions, summary = screener.screen_predictions(args.filepath, args.positions)

        # Display results
        if not args.summary_only:
            format_position_output(top_positions)

        # Display summary
        print(f"\n{'='*90}")
        print("SCREENING SUMMARY")
        print(f"{'='*90}")

        if 'message' in summary:
            print(summary['message'])
        else:
            print(f"Total Qualified Positions: {summary['total_positions']}")
            print(f"Long Positions: {summary['long_positions']}")
            print(f"Short Positions: {summary['short_positions']}")
            print(f"Average Predicted Return: {summary['avg_predicted_return']*100:.2f}%")
            print(f"Average Signal-to-Noise: {summary['avg_signal_to_noise']:.2f}")
            print(f"Average Direction Confidence: {summary['avg_direction_confidence']*100:.1f}%")
            print(f"Return Range: {summary['return_range']['min']*100:.2f}% to {summary['return_range']['max']*100:.2f}%")
            print(f"Next Trading Date: {summary['next_trading_date']}")

            if 'top_long_pick' in summary:
                pick = summary['top_long_pick']
                print(f"Best Long Pick: {pick['ticker']} ({pick['predicted_return']*100:.2f}%, "
                      f"S/N: {pick['signal_to_noise']:.2f}, Conf: {pick['direction_confidence']*100:.1f}%)")

            if 'top_short_pick' in summary:
                pick = summary['top_short_pick']
                print(f"Best Short Pick: {pick['ticker']} ({pick['predicted_return']*100:.2f}%, "
                      f"S/N: {pick['signal_to_noise']:.2f}, Conf: {pick['direction_confidence']*100:.1f}%)")

        # Export if requested
        if args.export and not top_positions.empty:
            # Select relevant columns for export
            export_columns = ['ticker', 'future_date', 'predicted_return', 'ci_lower', 'ci_upper',
                            'signal_to_noise', 'direction_confidence', 'screening_score']
            export_df = top_positions[export_columns].copy()
            export_df.to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

        print(f"{'='*90}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()