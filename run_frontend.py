#!/usr/bin/env python3
"""
Launch script for ARIMAX Frontend
Starts the Streamlit web application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit frontend application"""

    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')

    # Command to run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]

    print("Starting ARIMAX Stock Forecasting Frontend...")
    print("Open your browser to: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()

    try:
        # Run Streamlit
        subprocess.run(cmd, cwd=frontend_dir)
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")

if __name__ == "__main__":
    main()