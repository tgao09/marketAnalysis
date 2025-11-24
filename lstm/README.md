## LSTM Weekly Return Forecaster

This module provides a masked LSTM pipeline that mirrors the XGBoost workflows. It ingests the same engineered dataset (e.g. `dataset/stock_dataset_with_lags.csv`), respects the ticker universe in `dataset/training_stocks.txt`, and predicts the next week's return for every supported stock. Masking is handled through packed padded sequences so that differences in available history per stock never leak into training.

### Key Components

- `lstm/train_lstm.py`: trains a cross-sectional LSTM forecaster using the most recent `sequence_length` weeks per ticker. The script:
  - builds a one-week-ahead target via a per-ticker shift on `weekly_return`
  - forward/backward fills features per ticker, scales them (StandardScaler), and stores the scaler
  - converts each ticker history into left-padded sequences with explicit masking
  - trains a `StockLSTM` (see `lstm/model.py`) using Smooth L1 loss and saves model weights, scaler, metadata, and a training history CSV
- `lstm/forecast_lstm.py`: loads the artifacts, reconstructs masked sequences from the latest data, and generates next-week forecasts relative to the time of execution (forecast window start = run timestamp, target week = run timestamp + 7 days). Results are saved under `lstm/results/forecasts`.
- `lstm/data.py`: helper utilities for masking, scaler metadata, and ticker loading.

### Dependencies

The scripts rely on pandas, numpy, scikit-learn, joblib, and PyTorch (recommended: `pip install torch --index-url https://download.pytorch.org/whl/cpu`). Install GPU wheels if you plan to train on CUDA.

### Training Example

```bash
python lstm/train_lstm.py \
  --data-file dataset/stock_dataset_with_lags.csv \
  --tickers-file dataset/training_stocks.txt \
  --sequence-length 32 \
  --epochs 30 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --early-stop-patience 8 \
  --models-dir lstm/models \
  --results-dir lstm/results/training
```

Outputs include:

- `lstm/models/lstm_next_week_<timestamp>.pt` – best model weights
- `lstm/models/feature_scaler_<timestamp>.pkl` – fitted StandardScaler
- `lstm/models/lstm_metadata_<timestamp>.json` – configuration + file paths needed for inference
- `lstm/results/training/training_history_<timestamp>.csv` – per-epoch losses

Regularization knobs:

- `--dropout` (default 0.3) controls the dropout applied inside the LSTM head.
- `--weight-decay` adds L2 regularization to Adam; default `1e-4`.
- `--early-stop-patience` together with `--min-delta` stops training once validation loss stalls, preventing overfitting on later epochs.

### Forecasting Example

```bash
python lstm/forecast_lstm.py \
  --data-file dataset/stock_dataset_with_lags.csv \
  --tickers-file dataset/training_stocks.txt \
  --metadata-file lstm/models/lstm_metadata_<timestamp>.json \
  --results-dir lstm/results/forecasts
```

The forecast CSV contains one row per ticker with:

- `ticker`
- `predicted_return`
- `latest_data_date` (most recent observation used)
- `sequence_length_used` (in case some tickers lack a full 32-week window)
- `generated_at_utc`, `forecast_start_date`, and `forecast_target_week` (script run date + 7 days)

Use the metadata file to route the correct model/scaler pair whenever multiple training runs exist.
