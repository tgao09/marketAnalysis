# XGBoost Trading System

Gradient boosting-based stock return prediction system with multi-horizon forecasting, purged cross-validation, and ensemble signals.

## Architecture Overview

**Key Features:**
- Multi-horizon prediction (1w, 2w, 4w, 8w)
- Cross-sectional learning across all stocks
- Purged walk-forward CV (prevents lookahead bias)
- Technical + cross-sectional + interaction features
- Ensemble signals for robust predictions
- Transaction-cost-aware backtesting

**Differences from ARIMAX:**
- **Cross-sectional** (trains on all stocks) vs per-stock models
- **Gradient boosting** vs time series regression
- **Multi-horizon** forecasts vs single-step ahead
- **Non-linear** relationships vs linear + autoregressive
- **Technical indicators** vs pure lagged returns

## Workflow

### 1. Feature Engineering

Generate technical indicators, cross-sectional features, and forward return targets:

```bash
# From greenfield/ directory
python xgboost/feature_engineering.py \
    --input-file dataset/stock_dataset_with_lags.csv \
    --output-file xgboost/features_engineered.csv \
    --horizons 1 2 4 8
```

**Features Generated:**
- **Technical**: RSI, MACD, Bollinger Bands, ATR, MFI, Rate of Change
- **Cross-sectional**: Return ranks, volatility ranks, z-scores vs peers
- **Interactions**: Return × volatility, momentum × volume
- **Time**: Week/month/quarter cyclical encoding

### 2. Model Training

Train XGBoost models for multiple horizons with purged CV:

```bash
python xgboost/train_xgboost.py \
    --data-file xgboost/features_engineered.csv \
    --horizons 1 2 4 8
```

**Training Features:**
- Purged walk-forward CV (removes buffer between train/test)
- Embargo period (prevents reverse leakage from forward-looking labels)
- Time-decayed sample weights (recent data weighted higher)
- Early stopping to prevent overfitting
- Feature importance tracking

**Hyperparameters by Horizon:**
- **1-week**: More complex (max_depth=6), captures short-term patterns
- **8-week**: Heavily regularized (max_depth=3), filters noise

### 3. Forecasting

Generate multi-horizon predictions:

```bash
python xgboost/forecast_xgboost.py \
    --data-file xgboost/features_engineered.csv
```

Output: `xgboostresults/forecasts_YYYYMMDD_HHMMSS.csv`

### 4. Stock Screening

Screen opportunities using ensemble signals:

```bash
python xgboost/stock_screener.py \
    xgboostresults/forecasts_YYYYMMDD_HHMMSS.csv \
    -n 20 \
    --min-return 0.01 \
    --min-sharpe 0.0
```

**Ensemble Strategy:**
- Short-term (1w): 30% weight
- Medium (2w): 20% weight
- Long-term (4w, 8w): 25% each
- Signal strength = magnitude × direction agreement across horizons

### 5. Walk-Forward Testing

Backtest with realistic assumptions:

```bash
python xgboost/walk_forward_test.py \
    --horizon 1 \
    --test-weeks 52
```

**Realistic Assumptions:**
- Transaction costs (default: 0.1%)
- Position size limits (default: 10% per position)
- Weekly rebalancing
- Long-only strategy (top 5 predicted positive returns)

## Expected Performance

**After proper validation (no lookahead bias):**
- **Direction accuracy**: 55-65% (realistic for financial forecasting)
- **RMSE**: Varies by horizon (1w: ~0.02, 8w: ~0.04)
- **Sharpe ratio**: Model-dependent, expect modest improvement over buy-and-hold

**Feature Importance:**
- Technical indicators (RSI, momentum) typically most important
- Cross-sectional ranks capture relative strength
- Market features (SPY, sector ETFs) provide regime context

## Directory Structure

```
greenfield/xgboost/
├── feature_engineering.py       # Technical + cross-sectional features
├── xgboost_model.py            # Core model with purged CV
├── train_xgboost.py            # Training pipeline
├── forecast_xgboost.py         # Prediction generation
├── stock_screener.py           # Multi-horizon screening
├── walk_forward_test.py        # Backtesting
├── requirements_xgboost.txt    # Additional dependencies
├── xgboostmodels/             # Saved models (gitignored)
└── xgboostresults/            # Forecasts and results
```

## Dependencies

Install additional requirements:

```bash
pip install -r xgboost/requirements_xgboost.txt
```

**Core:**
- `xgboost==2.1.3` - Gradient boosting
- `ta==0.11.0` - Technical indicators library
- `optuna==4.1.0` - Hyperparameter optimization (optional)
- `shap==0.47.0` - Model interpretability (optional)

## Comparison: XGBoost vs ARIMAX

| Aspect | XGBoost | ARIMAX |
|--------|---------|--------|
| **Training** | Cross-sectional (all stocks) | Per-stock models |
| **Features** | Technical + cross-sectional | Lagged returns + market data |
| **Relationships** | Non-linear | Linear + autoregressive |
| **Horizons** | Multi-horizon (1w, 2w, 4w, 8w) | Single-step (1w) |
| **Validation** | Purged walk-forward CV | Expanding window CV |
| **Interpretability** | Feature importance, SHAP | Coefficients, order parameters |
| **Use Case** | Pattern recognition, regime shifts | Time series trends, seasonality |

**When to use XGBoost:**
- Rich feature set (technical, fundamental)
- Cross-sectional patterns (relative strength)
- Non-linear relationships
- Multiple prediction horizons needed

**When to use ARIMAX:**
- Pure time series forecasting
- Interpretable linear relationships
- Per-stock customization important
- Econometric rigor required

## Best Practices

1. **Prevent Lookahead Bias**:
   - Use purged CV with embargo periods
   - Only use lagged features (no contemporaneous)
   - Validate cross-sectional features are computed per-period

2. **Feature Engineering**:
   - Test features individually before combining
   - Monitor feature importance across folds
   - Remove highly correlated features (collinearity)

3. **Model Selection**:
   - Tune regularization by horizon (longer = more regularization)
   - Use early stopping on validation set
   - Ensemble multiple models with different feature sets

4. **Evaluation**:
   - Track both RMSE and directional accuracy
   - Simulate realistic trading (costs, slippage)
   - Test on multiple out-of-sample periods

## Notes

- **Data Requirements**: Needs engineered features from `feature_engineering.py`
- **Computation**: XGBoost training is CPU-intensive, use `n_jobs=-1` for parallelization
- **Memory**: Cross-sectional model loads all stocks simultaneously
- **Retraining**: Recommended monthly with new data

## Troubleshooting

**Issue**: Model predicts near-zero for all stocks
**Solution**: Check feature scaling, ensure sufficient training data, reduce regularization

**Issue**: High CV RMSE but good train RMSE
**Solution**: Increase regularization (reg_alpha, reg_lambda), reduce max_depth

**Issue**: Features have many NaN values
**Solution**: Check lag_features.py output, fill momentum features correctly

**Issue**: Direction accuracy ~50% (random)
**Solution**: Model not learning signal - check for lookahead bias, add more informative features
