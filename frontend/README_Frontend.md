# ARIMAX Stock Forecasting Frontend

A Streamlit web application for interactive stock price predictions using trained ARIMAX models.

## Features

- **Interactive Ticker Selection**: Choose from trained ARIMAX models or manually enter ticker symbols
- **Real-time Data Updates**: Update stock datasets with latest market data
- **Prediction Generation**: Generate new forecasts using the latest data
- **Interactive Visualizations**: Plotly charts showing historical data + future predictions with confidence intervals
- **Prediction Controls**: Adjust forecast periods (1-12 weeks) and toggle confidence intervals
- **Model Information**: View model details including AIC scores and validation status

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_frontend.txt
```

### 2. Launch the Frontend

```bash
python run_frontend.py
```

Or directly with Streamlit:

```bash
cd frontend
streamlit run app.py
```

### 3. Open in Browser

Navigate to: http://localhost:8501

## Usage

### Main Interface

1. **Control Panel (Sidebar)**:
   - Click "🔄 Update Dataset" to refresh stock data
   - Click "🎯 Generate Predictions" to create new forecasts
   - Adjust prediction periods (1-12 weeks)
   - Toggle confidence intervals and historical data

2. **Ticker Selection**:
   - Use dropdown to select from available trained models
   - Or manually type a ticker symbol (e.g., AAPL, MSFT, GOOGL)

3. **Visualization**:
   - Interactive Plotly chart with zoom, pan, and hover
   - Blue line: Historical stock prices
   - Red dashed line: Future predictions
   - Shaded area: 95% confidence intervals
   - Green vertical line: Prediction start point

4. **Prediction Summary**:
   - Table with predicted returns, prices, and confidence intervals
   - Model information including order and AIC score

### Data Pipeline Integration

The frontend integrates with your existing ARIMAX pipeline:

- **Dataset Updates**: Calls `greenfield/dataset/construct_dataset.py` + `lag_features.py`
- **Prediction Generation**: Executes `greenfield/arimax/forecast_arimax.py`
- **Model Loading**: Reads from `greenfield/arimax/models/`
- **Results Display**: Shows data from `greenfield/arimax/results/`

## Architecture

```
frontend/
├── app.py                 # Main Streamlit application
├── data_controller.py     # Dataset updates & prediction generation
├── model_handler.py       # ARIMAX model loading & management
├── visualizer.py          # Plotly chart creation
├── debug_app.py          # Debug version for troubleshooting
└── simple_app.py         # Minimal test version

tests/
└── test_frontend.py      # Playwright UI tests
```

## Testing

The frontend includes comprehensive Playwright tests:

```bash
# Install test dependencies
playwright install chromium

# Run all tests
cd tests
python -m pytest test_frontend.py -v

# Run specific test
python -m pytest test_frontend.py::test_app_loads -v
```

**Test Coverage**:
- ✅ App loading and title verification
- ✅ Sidebar controls and buttons
- ✅ Ticker input functionality
- ✅ Prediction settings (sliders, checkboxes)
- ✅ Dataset update and prediction generation buttons
- ✅ Error handling and invalid inputs
- ✅ Responsive design (desktop, tablet, mobile)
- ✅ Chart rendering and performance
- ✅ Cross-browser compatibility

## Troubleshooting

### No Models Available

If you see "No trained models found":

1. Ensure trained models exist in `greenfield/arimax/models/`
2. Models should be named like `{TICKER}_arimax.pkl`
3. Run model training first if needed

### No Predictions Available

If prediction generation fails:

1. Check that dataset exists: `greenfield/dataset/stock_dataset_with_lags.csv`
2. Run "Update Dataset" first to refresh data
3. Ensure models are compatible with current dataset format

### Import Errors

If you see import errors:

1. Ensure you're running from the project root directory
2. Check that all dependencies are installed: `pip install -r requirements_frontend.txt`
3. Verify the virtual environment is activated

### Performance Issues

If the app is slow:

1. Reduce the number of historical data points in visualization
2. Limit prediction periods to smaller ranges
3. Check network connectivity for stock data fetching

## Development

### Adding New Features

1. **New Visualizations**: Add functions to `visualizer.py`
2. **Data Sources**: Extend `model_handler.py`
3. **UI Components**: Modify `app.py`
4. **Backend Integration**: Update `data_controller.py`

### Testing Changes

Always run the Playwright test suite after making changes:

```bash
cd tests
python -m pytest test_frontend.py -v
```

### Code Structure

The frontend follows a modular architecture:
- **app.py**: Main UI and user interaction logic
- **data_controller.py**: Backend integration and data pipeline control
- **model_handler.py**: Model loading, caching, and prediction management
- **visualizer.py**: Chart creation and data visualization (self-contained)

This separation allows for easy testing, maintenance, and extension of functionality.