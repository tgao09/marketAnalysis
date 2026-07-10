# Universal backtester v1

Package: `common.backtesting`.

V1 is single-symbol, forecast-only walk-forward evaluation. No model files changed.

## Ownership and leakage boundary

Package owns target creation, fold creation, target-aware purge, embargo, hiding realized test targets, fresh-model enforcement, and prediction validation. Scripts/models only choose config and build causal features.

For a target horizon `h`, test rows `[s, e)` use only complete labels. Training anchor `i` is eligible only when:

```text
i + h + extra_purge_bars < s
```

So target-horizon purge is automatic; `extra_purge_bars` is optional extra separation. `embargo_rows` removes rows after earlier OOS windows from later retraining folds. It does not replace same-fold purge in strict forward-only backtests.

Engine prevents split/label leakage. It cannot stop a model from engineering future-aware features. Adapter feature code must be causal. Future strict sequential prediction remains separate work.

## Data and target contract

`YFinanceSource` fetches one adjusted-price ticker with `auto_adjust=True`, then canonicalizes column names to lowercase snake case. Input must have a unique `DatetimeIndex` and positive finite `close`; MultiIndex/multi-symbol frames are rejected.

```python
from common.backtesting import (
    BacktestConfig,
    BacktestEngine,
    LogReturnTarget,
    WalkForwardConfig,
    YFinanceSource,
)

config = BacktestConfig(
    target=LogReturnTarget(horizon_bars=5),
    walk_forward=WalkForwardConfig(
        min_train_rows=252,
        test_rows=21,
        embargo_rows=5,
    ),
    target_column="target_log_5bar",
    prediction_column="forecast_log_return",
)
source = YFinanceSource("SPY", period="10y", interval="1d")
```

Target is `log(close[t + h] / close[t])`. `interval="1d", horizon_bars=5` means five observed trading bars. Holidays and missing bars do not silently turn it into five calendar days. Intraday interval works same way. `YFinanceSource` metadata retains request settings, fetch time, yfinance version, and data hash.

yfinance data can be revised, has intraday retention limits, and is not point-in-time/survivorship-safe. Preserve raw snapshots for reproducible research.

## Model contract

Per fold, engine creates a fresh model and calls:

```python
model.fit(train, context)
returned_test = model.predict(test, context)
```

- `train`: raw normalized bars plus exactly configured target column.
- `test`: raw normalized bars only; no target, realized outcome, or prediction column.
- `context.warmup`: raw history strictly before test start, including purge gap; use for causal lookback/sequence state.
- `predict`: return test DataFrame with `context.prediction_column` and exact test index. Indexed `Series` is accepted for thin adapters.

Engine scores its own untouched test copy, then returns raw OOS rows plus target, configured prediction column, `target_end`, fold, error, and direction flag. Metrics: MAE, RMSE, correlation, directional hit rate. No PnL until execution timing, position sizing, fees, and slippage semantics are defined.

```python
class ConstantModel:
    def fit(self, train, context):
        self.value = train[context.target_column].mean()
        return self

    def predict(self, test, context):
        result = test.copy()
        result[context.prediction_column] = self.value
        return result

result = BacktestEngine(config).run_yfinance(
    source,
    lambda context: ConstantModel(),
)
```

## Deferred integration scope

Legacy `common.walk_forward` remains unchanged. It splits already-built frames and exposes target-labelled test data, so it is not integration path for new package.

| Model | Current path | Later integration scope | V1 status |
|---|---|---|---|
| GBM return | `gbm_return/train.py`, `backtest_walk_forward.py` | Fold-local auxiliary panel, features, clipping/recency weights, LightGBM adapter | First candidate |
| GP return | `gp_return/train.py`, `backtest_walk_forward.py` | Fold-local scaler/PCA, fresh GP, auxiliary panel adapter | Second candidate |
| LSTM return | `lstm_return/train.py`, `backtest_walk_forward.py` | Sequence adapter with `seq_len - 1` warmup and causal HMM transformer | Later |
| HMM-GP return | `hmm_gp_return/train.py`, `backtest_walk_forward.py` | Nested causal base forecasts and matured meta-label adapter | Defer |
| GP volatility | `gp_vol/train.py`, `backtest_options_proxy.py` | Add realized-volatility/noise target provider first | Defer |
| HMM regime | `hmm_regime/train.py`, `backtest_walk_forward.py` | Causal state/probability transformer, not scalar-return model | Defer |
| Chronos swing | `chronos2_swing/features.py`, `experiment.py`, `backtest.py` | Panel/entity folds, ranking, sequential Chronos context | Defer |

Integration order: build one `FoldModel` adapter; move target/split ownership into package; keep model-specific causal features in adapter; add frozen-fixture parity/leakage tests; then delete model-local splitter only after parity passes. Do not refactor model folders before this work.

Known deferred leakage fixes: GP/GBM/LSTM quarter-day feature derives quarter length from full fetched index; Chronos `regime_risk` uses full-series percentile rank; Chronos calendar `BDay` split and observed-row targets mismatch; standalone HMM state selection uses forward targets without label purge. HMM-GP cutoff is already causal but needs adapter-level verification.
