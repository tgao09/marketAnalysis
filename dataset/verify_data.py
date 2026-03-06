import pandas as pd
import numpy as np
import os

print("DATA VERIFICATION")
print("="*60)

# determine file path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'stock_dataset_with_lags.csv')

# load dataset
df = pd.read_csv(data_file)

print(f"\nDataset shape: {df.shape}")
print(f"Stocks: {df['ticker'].nunique()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

print("\n\nColumn Categories:")

# categorize columns
base_cols = ['ticker', 'Date', 'weekly_return']
original_features = ['high_return', 'low_return', 'volume_change', 'volatility']
market_features = [col for col in df.columns if any(x in col for x in ['SPY', 'VIX', 'XL']) and 'lag' not in col]
momentum_features = [col for col in df.columns if ('momentum' in col or 'price_to_52w' in col) and 'lag' not in col]
lagged_features = [col for col in df.columns if 'lag' in col]

print(f"  Base: {len(base_cols)} - {base_cols}")
print(f"  Original: {len(original_features)} - {original_features}")
print(f"  Market: {len(market_features)} features")
if market_features:
    print(f"    Examples: {market_features[:5]}")
print(f"  Momentum: {len(momentum_features)} - {momentum_features}")
print(f"  Lagged: {len(lagged_features)} features")

print("\n\nMissing Value Analysis:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(f"Columns with missing values (top 10):")
    print(missing.head(10))
    print(f"\nTotal columns with missing values: {len(missing)}")
else:
    print("No missing values!")

print("\n\nSample Statistics:")
if 'SPY_return' in df.columns:
    print("\nSPY_return:")
    print(df['SPY_return'].describe())

if 'momentum_12w' in df.columns:
    print("\nmomentum_12w:")
    print(df['momentum_12w'].describe())

print("\n\nFeatures that will be used in model (after arimax_model.py excludes contemporaneous):")
# simulate what arimax_model.py will use
exclude_columns = [
    'ticker', 'Date', 'weekly_return',
    'weekly_return_lag_1', 'weekly_return_lag_2', 'weekly_return_lag_3',
    'weekly_return_lag_4', 'weekly_return_lag_5',
    # exclude contemporaneous stock features
    'high_return', 'low_return', 'volume_change', 'volatility'
]

# also exclude contemporaneous market/momentum (models use only lagged versions)
contemporaneous_market = [col for col in df.columns if any(x in col for x in ['SPY', 'VIX', 'XL']) and 'lag' not in col]
contemporaneous_momentum = [col for col in df.columns if ('momentum' in col or 'price_to_52w' in col) and 'lag' not in col]
exclude_columns.extend(contemporaneous_market)
exclude_columns.extend(contemporaneous_momentum)

model_features = [col for col in df.columns if col not in exclude_columns]
print(f"\nTotal model features: {len(model_features)}")
print("\nFeature breakdown:")
original_lags = [f for f in model_features if any(x in f for x in ['high_return_lag', 'low_return_lag', 'volume_change_lag', 'volatility_lag'])]
market_lags = [f for f in model_features if any(x in f for x in ['SPY', 'VIX', 'XL']) and 'lag' in f]
momentum_lags = [f for f in model_features if ('momentum' in f or 'price_to_52w' in f) and 'lag' in f]

print(f"  Original lags (high/low/volume/volatility): {len(original_lags)}")
print(f"  Market lags: {len(market_lags)}")
print(f"  Momentum lags: {len(momentum_lags)}")

print("\n\nExpected feature counts:")
print(f"  Before improvements: 12 lagged features")
print(f"  After improvements: {len(model_features)} lagged features")
print(f"  Improvement: +{len(model_features) - 12} features")

print("\n✅ Verification complete!")
print(f"\nReady to train models with {len(model_features)} features")
