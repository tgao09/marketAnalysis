#!/usr/bin/env python3
"""
Stock Forecasting Frontend
Streamlit web application for interactive stock predictions using ARIMAX or XGBoost models
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add paths for importing modules relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / 'arimax'))
sys.path.append(str(PROJECT_ROOT / 'dataset'))
sys.path.append(str(PROJECT_ROOT / 'xgboost'))
sys.path.append(str(FRONTEND_DIR))

try:
    from data_controller import DataController
    from model_handler import ModelHandler
    from visualizer import create_prediction_chart, create_summary_table
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_controller' not in st.session_state:
    st.session_state.data_controller = DataController()
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'arimax'
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = ModelHandler(model_type=st.session_state.model_type)
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = None

def main():
    """Main Streamlit application"""

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.radio(
            "Model Family",
            options=["ARIMAX", "XGBoost"],
            index=0 if st.session_state.model_type == 'arimax' else 1,
            help="Choose between per-ticker ARIMAX models or tuned cross-sectional XGBoost models"
        )

    if selected_model.lower() != st.session_state.model_type:
        st.session_state.model_type = selected_model.lower()
        st.session_state.model_handler = ModelHandler(model_type=st.session_state.model_type)

    # Header
    st.title("📈 Stock Forecasting Dashboard")
    model_description = "ARIMAX time series" if st.session_state.model_type == 'arimax' else "tuned XGBoost cross-sectional"
    st.markdown(f"Interactive stock price predictions using **{model_description}** models")

    # Prediction Settings (moved up to define periods before it's used)
    st.subheader("🎛️ Prediction Settings")

    col1, col2, col3 = st.columns([2, 3, 3])

    with col1:
        periods = st.slider(
            "Prediction Periods (weeks)",
            min_value=1, max_value=12, value=4,
            help="Number of weeks to forecast into the future"
        )

    with col2:
        show_confidence = st.checkbox(
            "Show Confidence Intervals",
            value=True,
            help="Display 95% confidence bands around predictions"
        )

    with col3:
        show_historical = st.checkbox(
            "Show Historical Data",
            value=True,
            help="Include recent historical prices for context"
        )

    # Main content area
    st.subheader("🔍 Stock Ticker Selection")

    # Get available tickers
    available_tickers = st.session_state.model_handler.get_available_tickers()

    # Create columns for ticker selection and generate button
    col1, col2 = st.columns([4, 2])

    with col1:
        if available_tickers:
            selected_ticker = st.selectbox(
                "Select a stock ticker:",
                options=[""] + sorted(available_tickers),
                help="Choose from available trained models"
            )
        else:
            st.warning("⚠️ No trained predictions found. Please generate forecasts first.")
            selected_ticker = ""

    with col2:
        # Align button with selectbox by adding some vertical spacing
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        button_help = (
            f"Generate {selected_model.upper()} forecast for {selected_ticker}" if selected_ticker
            else (
                "Generate tuned XGBoost forecasts" if st.session_state.model_type == 'xgboost'
                else "Select a ticker first"
            )
        )
        button_disabled = (
            st.session_state.model_type == 'arimax' and not selected_ticker
        )
        if st.button("🎯 Generate Forecast", disabled=button_disabled,
                     help=button_help, use_container_width=True):
            if st.session_state.model_type == 'arimax' and not selected_ticker:
                st.error("Please select a ticker before generating forecasts.")
            else:
                spinner_text = (
                    f"Generating forecast for {selected_ticker}..."
                    if st.session_state.model_type == 'arimax'
                    else "Generating XGBoost forecasts..."
                )
                with st.spinner(spinner_text):
                    try:
                        if st.session_state.model_type == 'arimax':
                            result = st.session_state.data_controller.generate_ticker_forecast(selected_ticker, periods)
                        else:
                            # Generate cross-sectional forecasts using tuned XGBoost models
                            result = st.session_state.data_controller.generate_xgboost_forecasts(periods)
                        if result['success']:
                            st.session_state.last_prediction_time = datetime.now()
                            # Reload model handler to pick up new predictions
                            st.session_state.model_handler = ModelHandler(model_type=st.session_state.model_type)
                            if st.session_state.model_type == 'arimax':
                                success_message = (
                                    f"✅ Forecast generated for {selected_ticker} - {result.get('periods', 'unknown')} periods"
                                )
                            else:
                                horizons = ', '.join(str(h) for h in result.get('horizons', []))
                                if not horizons:
                                    horizons = 'all tuned horizons'
                                success_message = (
                                    f"✅ XGBoost forecasts updated (horizons: {horizons})"
                                )
                            st.success(success_message)
                        else:
                            st.error(f"❌ Forecast generation failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Forecast generation failed: {str(e)}")

    # Set ticker variable
    ticker = selected_ticker

    # Data Management Section
    st.subheader("📊 Data Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Add vertical centering for the button
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Update Dataset", help="Refresh stock_dataset_with_lags.csv with latest data"):
            # Create placeholders for progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message, progress_value):
                status_text.text(message)
                if progress_value is not None:
                    progress_bar.progress(progress_value)

            try:
                result = st.session_state.data_controller.update_dataset_with_progress(update_progress)
                if result['success']:
                    st.session_state.last_update_time = datetime.now()
                    progress_bar.progress(1.0)
                    status_text.success("✅ Dataset updated successfully!")
                    st.info(f"Updated {result.get('records', 'unknown')} records for {result.get('stocks', 'unknown')} stocks")
                else:
                    progress_bar.empty()
                    status_text.error(f"❌ Dataset update failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                progress_bar.empty()
                status_text.error(f"❌ Dataset update failed: {str(e)}")

    with col2:
        # Show system status
        if available_tickers:
            st.metric("Available Models", len(available_tickers))
        st.metric("Model Family", selected_model.upper())

    with col3:
        # Show last update time
        if st.session_state.last_update_time:
            st.info(f"Dataset updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M')}")

        if st.session_state.last_prediction_time:
            st.info(f"Predictions generated: {st.session_state.last_prediction_time.strftime('%Y-%m-%d %H:%M')}")


    # Prediction and visualization section
    if ticker:
        if ticker in available_tickers:
            st.subheader(f"📈 Predictions for {ticker}")

            try:
                # Generate predictions
                with st.spinner(f"Loading predictions for {ticker}..."):
                    predictions = st.session_state.model_handler.get_predictions(
                        ticker, periods, show_confidence
                    )

                if predictions is not None and not predictions.empty:
                    # Create visualization
                    fig = create_prediction_chart(
                        predictions, ticker, show_confidence, show_historical,
                        model_type=st.session_state.model_type
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Summary table
                    st.subheader(f"📊 Prediction Summary for {ticker}")
                    summary_table = create_summary_table(predictions)
                    st.dataframe(summary_table, use_container_width=True)

                    # Model information
                    if len(predictions) > 0:
                        model_info = predictions.iloc[0]
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if st.session_state.model_type == 'arimax':
                                st.metric("Model Order", model_info.get('model_order', 'Unknown'))
                            else:
                                horizon = model_info.get('horizon_weeks') or periods
                                horizon_display = f"{int(horizon)}w" if horizon else 'N/A'
                                st.metric("Forecast Horizon", horizon_display)
                        with col2:
                            if st.session_state.model_type == 'arimax':
                                model_aic = model_info.get('model_aic')
                                st.metric("Model AIC", f"{model_aic:.2f}" if model_aic is not None else "N/A")
                            else:
                                latest_date = model_info.get('latest_date')
                                if hasattr(latest_date, 'strftime'):
                                    latest_display = latest_date.strftime('%Y-%m-%d')
                                else:
                                    latest_display = latest_date if latest_date else 'N/A'
                                st.metric("Latest Data", latest_display)
                        with col3:
                            if st.session_state.model_type == 'arimax':
                                forecast_valid = model_info.get('forecast_valid', False)
                                st.metric("Forecast Valid", "✅ Yes" if forecast_valid else "❌ No")
                            else:
                                predicted_return = model_info.get('predicted_return')
                                return_display = (
                                    f"{predicted_return * 100:.2f}%" if predicted_return is not None else 'N/A'
                                )
                                st.metric("Predicted Return", return_display)

                else:
                    st.error(f"❌ No predictions available for {ticker}")

            except Exception as e:
                st.error(f"❌ Error loading predictions for {ticker}: {str(e)}")
                logger.error(f"Prediction error for {ticker}: {e}")

        else:
            st.warning(f"⚠️ No trained model found for {ticker}")
            if available_tickers:
                st.info("Available tickers: " + ", ".join(sorted(available_tickers)))

    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **Stock Forecasting Dashboard** | "
        "Built with Streamlit | "
        f"Models: {len(available_tickers) if available_tickers else 0} | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()
