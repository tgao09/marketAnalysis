

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from typing import List, Tuple, Dict
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"Loaded {len(df)} records for {df['ticker'].nunique()} stocks")
    return df


def prepare_feature_data(df: pd.DataFrame, ticker: str = None) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    
    # filter to specific ticker if provided
    if ticker:
        df = df[df['ticker'] == ticker].copy()

    # exclude columns that the arimax model excludes
    exclude_columns = [
        'ticker', 'Date', 'weekly_return',
        # exclude target's own lags (data leakage)
        'weekly_return_lag_1', 'weekly_return_lag_2', 'weekly_return_lag_3',
        'weekly_return_lag_4', 'weekly_return_lag_5',
        # exclude contemporaneous features (data leakage)
        'high_return', 'low_return', 'volume_change', 'volatility'
    ]

    # target variable
    target = df['weekly_return']

    # get exogenous columns
    exog_columns = [col for col in df.columns if col not in exclude_columns]
    exog = df[exog_columns]

    # drop columns that are entirely nan (e.g., missing tickers like ^vix)
    exog = exog.dropna(axis=1, how='all')
    feature_names = exog.columns.tolist()

    # remove rows with nan values
    valid_indices = (~target.isna()) & (~exog.isna().any(axis=1))
    target = target[valid_indices].reset_index(drop=True)
    exog = exog[valid_indices].reset_index(drop=True)

    # remove rows with inf values
    inf_mask = np.isinf(exog).any(axis=1)
    if inf_mask.any():
        logger.warning(f"Removing {inf_mask.sum()} rows with infinite values")
        target = target[~inf_mask].reset_index(drop=True)
        exog = exog[~inf_mask].reset_index(drop=True)

    # clip extreme values (beyond 3 standard deviations)
    for col in exog.columns:
        mean = exog[col].mean()
        std = exog[col].std()
        if std > 0:
            lower = mean - 5 * std
            upper = mean + 5 * std
            exog[col] = exog[col].clip(lower, upper)

    logger.info(f"Prepared {len(target)} observations with {len(feature_names)} features")

    return target, exog, feature_names


def compute_random_forest_importance(target: pd.Series, exog: pd.DataFrame,
                                     n_estimators: int = 100) -> pd.DataFrame:
    
    logger.info("Computing Random Forest feature importance...")

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(exog, target)

    importance_df = pd.DataFrame({
        'feature': exog.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)

    return importance_df


def compute_mutual_information(target: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("Computing Mutual Information scores...")

    mi_scores = mutual_info_regression(exog, target, random_state=42)

    mi_df = pd.DataFrame({
        'feature': exog.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    return mi_df


def compute_correlation(target: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("Computing correlations...")

    correlations = exog.corrwith(target).abs()

    corr_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    }).sort_values('correlation', ascending=False)

    return corr_df


def select_features(importance_df: pd.DataFrame, mi_df: pd.DataFrame,
                   corr_df: pd.DataFrame, top_n: int = 30,
                   method: str = 'combined') -> List[str]:
    
    if method == 'rf':
        selected = importance_df.head(top_n)['feature'].tolist()
    elif method == 'mi':
        selected = mi_df.head(top_n)['feature'].tolist()
    elif method == 'correlation':
        selected = corr_df.head(top_n)['feature'].tolist()
    elif method == 'combined':
        # combine all three methods with equal weighting
        merged = importance_df.merge(mi_df, on='feature').merge(corr_df, on='feature')

        # normalize scores to 0-1 range
        merged['rf_norm'] = (merged['rf_importance'] - merged['rf_importance'].min()) / \
                           (merged['rf_importance'].max() - merged['rf_importance'].min())
        merged['mi_norm'] = (merged['mutual_info'] - merged['mutual_info'].min()) / \
                           (merged['mutual_info'].max() - merged['mutual_info'].min())
        merged['corr_norm'] = (merged['correlation'] - merged['correlation'].min()) / \
                             (merged['correlation'].max() - merged['correlation'].min())

        # combined score (average of normalized scores)
        merged['combined_score'] = (merged['rf_norm'] + merged['mi_norm'] + merged['corr_norm']) / 3
        merged = merged.sort_values('combined_score', ascending=False)

        selected = merged.head(top_n)['feature'].tolist()
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Selected {len(selected)} features using {method} method")
    return selected


def analyze_feature_categories(feature_names: List[str]) -> Dict[str, int]:
    
    categories = {
        'market_indicators': 0,
        'stock_technical': 0,
        'momentum': 0,
        'lag_1': 0,
        'lag_2': 0,
        'lag_3': 0
    }

    for feature in feature_names:
        # market indicators (spy, vix, sector etfs)
        if any(market in feature for market in ['SPY', 'VIX', 'XL']):
            categories['market_indicators'] += 1
        # momentum features
        elif 'momentum' in feature or 'price_to_52w' in feature:
            categories['momentum'] += 1
        # stock technical features
        else:
            categories['stock_technical'] += 1

        # lag timing
        if '_lag_1' in feature:
            categories['lag_1'] += 1
        elif '_lag_2' in feature:
            categories['lag_2'] += 1
        elif '_lag_3' in feature:
            categories['lag_3'] += 1

    return categories


def main():
    parser = argparse.ArgumentParser(description='Feature selection for ARIMAX models')
    parser.add_argument('--input-file', type=str,
                       default='dataset/stock_dataset_with_lags.csv',
                       help='Path to dataset with lagged features')
    parser.add_argument('--output-file', type=str,
                       default='dataset/selected_features.txt',
                       help='Output file for selected features')
    parser.add_argument('--top-n', type=int, default=30,
                       help='Number of features to select (default: 30)')
    parser.add_argument('--method', type=str, default='combined',
                       choices=['rf', 'mi', 'correlation', 'combined'],
                       help='Feature selection method')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Analyze specific ticker (default: all stocks combined)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of Random Forest estimators')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed feature importance report')

    args = parser.parse_args()

    # load data
    df = load_data(args.input_file)

    # prepare features
    target, exog, feature_names = prepare_feature_data(df, args.ticker)

    logger.info(f"\nAnalyzing {len(feature_names)} features")
    logger.info(f"Using {len(target)} observations")

    # compute feature importance using multiple methods
    rf_importance = compute_random_forest_importance(target, exog, args.n_estimators)
    mi_scores = compute_mutual_information(target, exog)
    correlations = compute_correlation(target, exog)

    # select features
    selected_features = select_features(
        rf_importance, mi_scores, correlations,
        top_n=args.top_n, method=args.method
    )

    # analyze categories
    categories = analyze_feature_categories(selected_features)

    # print summary
    print("\n" + "="*60)
    print("FEATURE SELECTION SUMMARY")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Original features: {len(feature_names)}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Reduction: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")
    print("\nFeature Categories:")
    for category, count in categories.items():
        print(f"  {category}: {count}")

    print(f"\nTop 20 selected features:")
    for i, feature in enumerate(selected_features[:20], 1):
        print(f"  {i:2d}. {feature}")

    # save selected features
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(selected_features))

    logger.info(f"\nSelected features saved to: {output_path}")

    # save detailed report if requested
    if args.save_report:
        report_path = output_path.parent / 'feature_importance_report.csv'

        # merge all scores
        report_df = rf_importance.merge(mi_scores, on='feature').merge(correlations, on='feature')
        report_df['selected'] = report_df['feature'].isin(selected_features)
        report_df = report_df.sort_values('rf_importance', ascending=False)

        report_df.to_csv(report_path, index=False)
        logger.info(f"Detailed report saved to: {report_path}")

        # print top features by each method
        print("\nTop 10 features by Random Forest importance:")
        for i, row in rf_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['rf_importance']:.4f}")

        print("\nTop 10 features by Mutual Information:")
        for i, row in mi_scores.head(10).iterrows():
            print(f"  {row['feature']}: {row['mutual_info']:.4f}")

        print("\nTop 10 features by Correlation:")
        for i, row in correlations.head(10).iterrows():
            print(f"  {row['feature']}: {row['correlation']:.4f}")

    print("="*60)


if __name__ == "__main__":
    main()