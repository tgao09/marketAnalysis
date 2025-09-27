#!/usr/bin/env python3
"""
Data Controller for ARIMAX Frontend
Handles dataset updates and prediction generation
"""

import os
import sys
import subprocess
import logging
from typing import Dict, Any
from datetime import datetime

# Add paths for importing modules
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(project_root, 'greenfield', 'arimax'))
sys.path.append(os.path.join(project_root, 'greenfield', 'dataset'))

logger = logging.getLogger(__name__)

class DataController:
    """
    Handles data pipeline operations for the ARIMAX frontend
    """

    def __init__(self):
        """Initialize the data controller with project paths"""
        self.project_root = os.path.join(os.path.dirname(__file__), '..')
        self.dataset_dir = os.path.join(self.project_root, 'greenfield', 'dataset')
        self.arimax_dir = os.path.join(self.project_root, 'greenfield', 'arimax')
        self.results_dir = os.path.join(self.arimax_dir, 'arimaxresults')

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def update_dataset(self) -> Dict[str, Any]:
        """
        Update the stock dataset with latest data

        Returns:
            Dict containing success status and details
        """
        try:
            logger.info("Starting dataset update...")

            # Path to the dataset construction script
            construct_script = os.path.join(self.dataset_dir, 'construct_dataset.py')
            lag_features_script = os.path.join(self.dataset_dir, 'lag_features.py')

            if not os.path.exists(construct_script):
                return {
                    'success': False,
                    'error': f'Dataset construction script not found: {construct_script}'
                }

            # Step 1: Update base dataset
            logger.info("Running dataset construction...")
            result1 = subprocess.run(
                [sys.executable, construct_script],
                cwd=self.dataset_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result1.returncode != 0:
                return {
                    'success': False,
                    'error': f'Dataset construction failed: {result1.stderr}'
                }

            # Step 2: Add lag features if script exists
            if os.path.exists(lag_features_script):
                logger.info("Adding lag features...")
                result2 = subprocess.run(
                    [sys.executable, lag_features_script],
                    cwd=self.dataset_dir,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minute timeout
                )

                if result2.returncode != 0:
                    logger.warning(f"Lag features script failed: {result2.stderr}")
                    # Don't fail the entire process for lag features

            # Check if the dataset file was created/updated
            dataset_file = os.path.join(self.dataset_dir, 'stock_dataset_with_lags.csv')
            if not os.path.exists(dataset_file):
                dataset_file = os.path.join(self.dataset_dir, 'stock_dataset.csv')

            if os.path.exists(dataset_file):
                # Get dataset info
                try:
                    import pandas as pd
                    df = pd.read_csv(dataset_file)
                    records = len(df)
                    stocks = df['ticker'].nunique() if 'ticker' in df.columns else 0

                    logger.info(f"Dataset updated successfully: {records} records, {stocks} stocks")
                    return {
                        'success': True,
                        'records': records,
                        'stocks': stocks,
                        'file': dataset_file,
                        'timestamp': datetime.now().isoformat(),
                        'stdout': result1.stdout  # Include stdout for progress parsing
                    }
                except Exception as e:
                    logger.warning(f"Could not read dataset info: {e}")
                    return {
                        'success': True,
                        'message': 'Dataset updated but could not read details',
                        'timestamp': datetime.now().isoformat(),
                        'stdout': result1.stdout
                    }
            else:
                return {
                    'success': False,
                    'error': 'Dataset file was not created'
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Dataset update timed out'
            }
        except Exception as e:
            logger.error(f"Dataset update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def update_dataset_with_progress(self, progress_callback=None):
        """
        Update the stock dataset with real-time progress tracking

        Args:
            progress_callback: Function to call with progress updates

        Returns:
            Dict containing success status and details
        """
        try:
            if progress_callback:
                progress_callback("Starting dataset update...", 0.0)

            # Path to the dataset construction script
            construct_script = os.path.join(self.dataset_dir, 'construct_dataset.py')
            lag_features_script = os.path.join(self.dataset_dir, 'lag_features.py')

            if not os.path.exists(construct_script):
                return {
                    'success': False,
                    'error': f'Dataset construction script not found: {construct_script}'
                }

            # Step 1: Update base dataset with real-time output
            if progress_callback:
                progress_callback("Running dataset construction...", 0.1)

            import subprocess
            process = subprocess.Popen(
                [sys.executable, construct_script],
                cwd=self.dataset_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Read training_stocks.txt to get total stock count for progress
            try:
                with open(os.path.join(self.dataset_dir, 'training_stocks.txt'), 'r') as f:
                    total_stocks = len([ticker.strip() for ticker in f.read().split(',') if ticker.strip()])
            except:
                total_stocks = 342  # fallback estimate

            processed_stocks = 0
            output_lines = []

            for line in process.stdout:
                output_lines.append(line)
                if "Processing" in line and "..." in line:
                    processed_stocks += 1
                    progress = 0.1 + (processed_stocks / total_stocks) * 0.7  # 10% to 80%
                    ticker = line.split("Processing ")[1].split("...")[0] if "Processing " in line else ""
                    if progress_callback:
                        progress_callback(f"Processing {ticker}... ({processed_stocks}/{total_stocks})", progress)
                elif "Added" in line and "weekly records" in line:
                    if progress_callback:
                        # Extract ticker and record count from log line
                        parts = line.split()
                        records = parts[1] if len(parts) > 1 else "unknown"
                        ticker = parts[-1] if len(parts) > 0 else ""
                        progress_callback(f"Added {records} records for {ticker}", None)

            process.wait()

            if process.returncode != 0:
                return {
                    'success': False,
                    'error': f'Dataset construction failed: {"".join(output_lines)}'
                }

            if progress_callback:
                progress_callback("Adding lag features...", 0.85)

            # Step 2: Add lag features if script exists
            if os.path.exists(lag_features_script):
                result2 = subprocess.run(
                    [sys.executable, lag_features_script],
                    cwd=self.dataset_dir,
                    capture_output=True,
                    text=True,
                    timeout=180
                )

                if result2.returncode != 0:
                    logger.warning(f"Lag features script failed: {result2.stderr}")

            if progress_callback:
                progress_callback("Finalizing dataset...", 0.95)

            # Check if the dataset file was created/updated
            dataset_file = os.path.join(self.dataset_dir, 'stock_dataset_with_lags.csv')
            if not os.path.exists(dataset_file):
                dataset_file = os.path.join(self.dataset_dir, 'stock_dataset.csv')

            if os.path.exists(dataset_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(dataset_file)
                    records = len(df)
                    stocks = df['ticker'].nunique() if 'ticker' in df.columns else 0

                    if progress_callback:
                        progress_callback(f"Complete! {records} records, {stocks} stocks", 1.0)

                    logger.info(f"Dataset updated successfully: {records} records, {stocks} stocks")
                    return {
                        'success': True,
                        'records': records,
                        'stocks': stocks,
                        'file': dataset_file,
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    if progress_callback:
                        progress_callback("Dataset updated successfully", 1.0)
                    return {
                        'success': True,
                        'message': 'Dataset updated but could not read details',
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'success': False,
                    'error': 'Dataset file was not created'
                }

        except Exception as e:
            logger.error(f"Dataset update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def generate_predictions(self) -> Dict[str, Any]:
        """
        Generate new ARIMAX predictions using the updated dataset

        Returns:
            Dict containing success status and details
        """
        try:
            logger.info("Starting prediction generation...")

            # Path to the forecast script
            forecast_script = os.path.join(self.arimax_dir, 'forecast_arimax.py')

            if not os.path.exists(forecast_script):
                return {
                    'success': False,
                    'error': f'Forecast script not found: {forecast_script}'
                }

            # Check if dataset exists
            dataset_file = os.path.join(self.dataset_dir, 'stock_dataset_with_lags.csv')
            if not os.path.exists(dataset_file):
                dataset_file = os.path.join(self.dataset_dir, 'stock_dataset.csv')
                if not os.path.exists(dataset_file):
                    return {
                        'success': False,
                        'error': 'No dataset file found. Please update dataset first.'
                    }

            # Run the forecast script
            logger.info("Running ARIMAX prediction generation...")
            result = subprocess.run(
                [
                    sys.executable, forecast_script,
                    '--data-file', dataset_file,
                    '--periods', '4',
                    '--models-dir', os.path.join(self.arimax_dir, 'arimaxmodels'),
                    '--results-dir', self.results_dir,
                    '--forecasting-mode', 'individual'
                ],
                cwd=self.arimax_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Prediction generation failed: {result.stderr}',
                    'stdout': result.stdout
                }

            # Check for generated files
            import glob
            forecast_files = glob.glob(os.path.join(self.results_dir, 'future_forecasts_*.csv'))

            if forecast_files:
                # Get the most recent forecast file
                latest_file = max(forecast_files, key=os.path.getctime)

                # Count stocks in the results
                try:
                    import pandas as pd
                    df = pd.read_csv(latest_file)
                    stocks = df['ticker'].nunique() if 'ticker' in df.columns else 0
                    forecasts = len(df)

                    logger.info(f"Predictions generated successfully: {forecasts} forecasts for {stocks} stocks")
                    return {
                        'success': True,
                        'stocks': stocks,
                        'forecasts': forecasts,
                        'file': latest_file,
                        'timestamp': datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Could not read prediction results: {e}")
                    return {
                        'success': True,
                        'message': 'Predictions generated but could not read details',
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'success': False,
                    'error': 'No forecast files were generated'
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Prediction generation timed out'
            }
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the current dataset

        Returns:
            Dict containing dataset statistics
        """
        try:
            dataset_file = os.path.join(self.dataset_dir, 'stock_dataset_with_lags.csv')
            if not os.path.exists(dataset_file):
                dataset_file = os.path.join(self.dataset_dir, 'stock_dataset.csv')

            if not os.path.exists(dataset_file):
                return {
                    'exists': False,
                    'error': 'No dataset file found'
                }

            import pandas as pd
            df = pd.read_csv(dataset_file)

            info = {
                'exists': True,
                'file': dataset_file,
                'records': len(df),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(dataset_file)).isoformat()
            }

            if 'ticker' in df.columns:
                info['stocks'] = df['ticker'].nunique()
                info['tickers'] = sorted(df['ticker'].unique().tolist())

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                info['date_range'] = {
                    'start': df['Date'].min().strftime('%Y-%m-%d'),
                    'end': df['Date'].max().strftime('%Y-%m-%d')
                }

            return info

        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }

    def get_predictions_info(self) -> Dict[str, Any]:
        """
        Get information about available predictions

        Returns:
            Dict containing prediction file statistics
        """
        try:
            import glob

            # Find prediction files
            forecast_files = glob.glob(os.path.join(self.results_dir, 'future_forecasts_*.csv'))

            if not forecast_files:
                return {
                    'exists': False,
                    'count': 0,
                    'error': 'No prediction files found'
                }

            # Get the most recent file
            latest_file = max(forecast_files, key=os.path.getctime)

            import pandas as pd
            df = pd.read_csv(latest_file)

            info = {
                'exists': True,
                'count': len(forecast_files),
                'latest_file': latest_file,
                'forecasts': len(df),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(latest_file)).isoformat()
            }

            if 'ticker' in df.columns:
                info['stocks'] = df['ticker'].nunique()
                info['tickers'] = sorted(df['ticker'].unique().tolist())

            if 'future_date' in df.columns:
                df['future_date'] = pd.to_datetime(df['future_date'])
                info['date_range'] = {
                    'start': df['future_date'].min().strftime('%Y-%m-%d'),
                    'end': df['future_date'].max().strftime('%Y-%m-%d')
                }

            return info

        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }
    def generate_ticker_forecast(self, ticker: str, periods: int = 4) -> Dict[str, Any]:
        """
        Generate ARIMAX forecast for a specific ticker

        Args:
            ticker: Stock ticker symbol to forecast
            periods: Number of future periods to predict

        Returns:
            Dict containing success status and details
        """
        try:
            logger.info(f"Starting forecast generation for {ticker}...")

            # Path to the forecast script
            forecast_script = os.path.join(self.arimax_dir, 'forecast_arimax.py')

            if not os.path.exists(forecast_script):
                return {
                    'success': False,
                    'error': f'Forecast script not found: {forecast_script}'
                }

            # Check if dataset exists
            dataset_file = os.path.join(self.dataset_dir, 'stock_dataset_with_lags.csv')
            if not os.path.exists(dataset_file):
                dataset_file = os.path.join(self.dataset_dir, 'stock_dataset.csv')
                if not os.path.exists(dataset_file):
                    return {
                        'success': False,
                        'error': 'No dataset file found. Please update dataset first.'
                    }

            # Run the forecast script for specific ticker
            logger.info(f"Running ARIMAX forecast for {ticker}...")
            result = subprocess.run(
                [
                    sys.executable, forecast_script,
                    '--ticker', ticker,
                    '--data-file', dataset_file,
                    '--periods', str(periods),
                    '--models-dir', os.path.join(self.arimax_dir, 'arimaxmodels'),
                    '--results-dir', self.results_dir,
                    '--forecasting-mode', 'individual'
                ],
                cwd=self.arimax_dir,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout for single ticker
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Forecast generation failed for {ticker}: {result.stderr}',
                    'stdout': result.stdout
                }

            # Check for generated files
            import glob
            forecast_files = glob.glob(os.path.join(self.results_dir, f'future_forecasts_{ticker}_*.csv'))

            if forecast_files:
                # Get the most recent forecast file for this ticker
                latest_file = max(forecast_files, key=os.path.getctime)

                logger.info(f"Forecast generated successfully for {ticker}: {latest_file}")
                return {
                    'success': True,
                    'ticker': ticker,
                    'periods': periods,
                    'file': latest_file,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'No forecast file was generated for {ticker}'
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Forecast generation timed out for {ticker}'
            }
        except Exception as e:
            logger.error(f"Forecast generation failed for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
