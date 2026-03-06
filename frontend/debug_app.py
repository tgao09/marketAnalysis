#!/usr/bin/env python3
"""
Debug version of ARIMAX Frontend to isolate issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ARIMAX Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application"""

    # Header
    st.title("📈 ARIMAX Stock Forecasting")
    st.markdown("Interactive stock price predictions using ARIMAX time series models")

    # Test basic functionality
    st.write("Debug: Basic Streamlit components are working")

    # Test sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        st.write("Debug: Sidebar is working")

        if st.button("Test Button"):
            st.success("Button click works!")

    # Test columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Main Content")
        st.write("Debug: Columns are working")

    with col2:
        st.subheader("Side Panel")
        st.write("Debug: Side panel working")

    # Test that dependencies are available
    st.subheader("Dependency Check")

    try:
        import plotly.graph_objects as go
        st.success("✅ Plotly imported successfully")
    except ImportError as e:
        st.error(f"❌ Plotly import failed: {e}")

    try:
        import yfinance as yf
        st.success("✅ yfinance imported successfully")
    except ImportError as e:
        st.error(f"❌ yfinance import failed: {e}")

    # Test project paths
    st.subheader("Path Check")
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    arimax_dir = project_root / 'arimax'

    st.write(f"Current directory: {current_dir}")
    st.write(f"Project root: {project_root}")
    st.write(f"ARIMAX directory: {arimax_dir}")
    st.write(f"ARIMAX directory exists: {os.path.exists(arimax_dir)}")

    # Try importing our modules
    st.subheader("Module Import Check")

    # Add paths
    sys.path.append(str(arimax_dir))
    sys.path.append(str(current_dir))

    try:
        from data_controller import DataController
        st.success("✅ DataController imported successfully")
    except ImportError as e:
        st.error(f"❌ DataController import failed: {e}")

    try:
        from model_handler import ModelHandler
        st.success("✅ ModelHandler imported successfully")
    except ImportError as e:
        st.error(f"❌ ModelHandler import failed: {e}")

    try:
        from visualizer import create_prediction_chart
        st.success("✅ Visualizer imported successfully")
    except ImportError as e:
        st.error(f"❌ Visualizer import failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown("🤖 **Debug Mode** | ARIMAX Stock Forecasting")

if __name__ == "__main__":
    main()
