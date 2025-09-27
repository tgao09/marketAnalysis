#!/usr/bin/env python3
"""
Visualization Engine for ARIMAX Frontend
Creates interactive charts and tables using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def create_prediction_chart(predictions_df: pd.DataFrame, ticker: str,
                          show_confidence: bool = True,
                          show_historical: bool = True) -> go.Figure:
    """
    Create interactive prediction chart with Plotly

    Args:
        predictions_df: DataFrame with predictions
        ticker: Stock ticker symbol
        show_confidence: Whether to show confidence intervals
        show_historical: Whether to include historical data

    Returns:
        Plotly Figure object
    """
    try:
        fig = go.Figure()

        if predictions_df.empty:
            fig.add_annotation(
                text=f"No prediction data available for {ticker}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Convert predictions to prices
        # Note: We'll handle price conversion directly here to avoid circular imports
        predictions_with_prices = predictions_df.copy()
        if not predictions_with_prices.empty:
            # Simple price conversion logic
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if not hist.empty:
                    hist = hist.reset_index()  # Ensure proper handling
                    starting_price = float(hist['Close'].iloc[-1])
                else:
                    starting_price = 100.0
            except Exception:
                starting_price = 100.0

            # Convert returns to prices
            predictions_with_prices['stock_price'] = 0.0
            for i in range(len(predictions_with_prices)):
                predicted_return = predictions_with_prices.iloc[i]['predicted_return']
                if abs(predicted_return) < 1:
                    return_multiplier = predicted_return
                else:
                    return_multiplier = predicted_return / 100

                if i == 0:
                    week_close = starting_price * (1 + return_multiplier)
                    predictions_with_prices.iloc[i, predictions_with_prices.columns.get_loc('stock_price')] = week_close
                else:
                    prev_week_close = predictions_with_prices.iloc[i-1]['stock_price']
                    current_week_close = prev_week_close * (1 + return_multiplier)
                    predictions_with_prices.iloc[i, predictions_with_prices.columns.get_loc('stock_price')] = current_week_close

        # Get historical data if requested
        historical_data = None
        if show_historical:
            try:
                import yfinance as yf
                from datetime import datetime, timedelta

                stock = yf.Ticker(ticker)
                # Get data up to and including today (if available)
                end_date = datetime.now() + timedelta(days=1)  # Include today
                start_date = end_date - timedelta(days=45)  # More buffer for weekends/holidays

                # Convert to string format for yfinance to avoid pandas timestamp issues
                end_date_str = end_date.strftime('%Y-%m-%d')
                start_date_str = start_date.strftime('%Y-%m-%d')

                hist = stock.history(start=start_date_str, end=end_date_str)
                if not hist.empty:
                    hist = hist.reset_index()
                    hist['ticker'] = ticker

                    # Handle timezone-aware dates properly
                    if 'Date' in hist.columns:
                        hist['date'] = pd.to_datetime(hist['Date'])
                        if hist['date'].dt.tz is not None:
                            hist['date'] = hist['date'].dt.tz_localize(None)
                    else:
                        hist['date'] = pd.to_datetime(hist.index)
                        if hist['date'].dt.tz is not None:
                            hist['date'] = hist['date'].dt.tz_localize(None)

                    hist['stock_price'] = hist['Close']
                    # Get the most recent 30 trading days to ensure we have current data
                    historical_data = hist[['ticker', 'date', 'stock_price']].tail(30)
            except Exception as e:
                logger.debug(f"Could not fetch historical data for {ticker}: {e}")
                historical_data = None

        # Plot historical data
        if historical_data is not None and not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['stock_price'],
                mode='lines+markers',
                name=f'{ticker} Historical',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            'Price: $%{y:.2f}<br>' +
                            '<extra></extra>'
            ))

        # Plot predictions
        if not predictions_with_prices.empty and 'stock_price' in predictions_with_prices.columns:
            fig.add_trace(go.Scatter(
                x=predictions_with_prices['date'],
                y=predictions_with_prices['stock_price'],
                mode='lines+markers',
                name=f'{ticker} Predictions',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6, symbol='square'),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            'Predicted Price: $%{y:.2f}<br>' +
                            'Return: %{customdata:.3f}<br>' +
                            '<extra></extra>',
                customdata=predictions_with_prices['predicted_return']
            ))

            # Add confidence intervals if available and requested
            if (show_confidence and 'ci_lower_price' in predictions_with_prices.columns and
                'ci_upper_price' in predictions_with_prices.columns):

                # Upper confidence bound
                fig.add_trace(go.Scatter(
                    x=predictions_with_prices['date'],
                    y=predictions_with_prices['ci_upper_price'],
                    mode='lines',
                    line=dict(width=0),
                    name='95% CI Upper',
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # Lower confidence bound (filled area)
                fig.add_trace(go.Scatter(
                    x=predictions_with_prices['date'],
                    y=predictions_with_prices['ci_lower_price'],
                    mode='lines',
                    line=dict(width=0),
                    name='95% Confidence Interval',
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    hovertemplate='<b>Confidence Interval</b><br>' +
                                'Date: %{x}<br>' +
                                'Lower: $%{y:.2f}<br>' +
                                'Upper: $%{customdata:.2f}<br>' +
                                '<extra></extra>',
                    customdata=predictions_with_prices['ci_upper_price']
                ))

            # Add separation line between historical and predictions
            if historical_data is not None and not historical_data.empty:
                separation_date = historical_data['date'].max()
                # Ensure separation_date is properly formatted for plotly
                try:
                    if hasattr(separation_date, 'to_pydatetime'):
                        separation_date = separation_date.to_pydatetime()
                    elif hasattr(separation_date, 'tz_localize'):
                        separation_date = pd.to_datetime(separation_date).tz_localize(None)
                    else:
                        # Convert to datetime if it's not already
                        separation_date = pd.to_datetime(separation_date)
                        if hasattr(separation_date, 'tz_localize') and separation_date.tz is not None:
                            separation_date = separation_date.tz_localize(None)
                except Exception as e:
                    logger.debug(f"Error handling separation_date: {e}")
                    # Fallback: use current datetime
                    separation_date = datetime.now()

                # Get y-axis range for vertical line
                all_prices = []
                if 'stock_price' in predictions_with_prices.columns:
                    all_prices.extend(predictions_with_prices['stock_price'].tolist())
                if historical_data is not None:
                    all_prices.extend(historical_data['stock_price'].tolist())

                if all_prices:
                    y_min, y_max = min(all_prices), max(all_prices)
                    y_range = y_max - y_min
                    y_min -= y_range * 0.05
                    y_max += y_range * 0.05

                    # Use add_shape instead of add_vline for better datetime compatibility
                    fig.add_shape(
                        type="line",
                        x0=separation_date, x1=separation_date,
                        y0=y_min, y1=y_max,
                        line=dict(color="green", width=2, dash="dot"),
                        xref="x", yref="y"
                    )

                    # Add annotation separately
                    fig.add_annotation(
                        x=separation_date,
                        y=y_max,
                        text="Predictions Start",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="green",
                        bgcolor="white",
                        bordercolor="green",
                        borderwidth=1
                    )

        # Customize layout
        fig.update_layout(
            title=dict(
                text=f'<b>{ticker} Stock Price Forecast</b><br>' +
                     '<sub>Historical Data + ARIMAX Predictions</sub>',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )

        # Format axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickformat='$.2f'
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating prediction chart: {e}")

        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

def create_summary_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table from predictions

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        Formatted DataFrame for display
    """
    try:
        if predictions_df.empty:
            return pd.DataFrame({'Message': ['No predictions available']})

        # Convert to prices for display using simple logic
        predictions_with_prices = predictions_df.copy()

        if 'ticker' in predictions_df.columns and len(predictions_df) > 0:
            ticker = predictions_df['ticker'].iloc[0]

            # Simple price conversion
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if not hist.empty:
                    hist = hist.reset_index()  # Ensure proper handling
                    starting_price = float(hist['Close'].iloc[-1])
                else:
                    starting_price = 100.0
            except Exception:
                starting_price = 100.0

            # Convert returns to prices
            predictions_with_prices['stock_price'] = 0.0
            for i in range(len(predictions_with_prices)):
                predicted_return = predictions_with_prices.iloc[i]['predicted_return']
                if abs(predicted_return) < 1:
                    return_multiplier = predicted_return
                else:
                    return_multiplier = predicted_return / 100

                if i == 0:
                    week_close = starting_price * (1 + return_multiplier)
                    predictions_with_prices.iloc[i, predictions_with_prices.columns.get_loc('stock_price')] = week_close
                else:
                    prev_week_close = predictions_with_prices.iloc[i-1]['stock_price']
                    current_week_close = prev_week_close * (1 + return_multiplier)
                    predictions_with_prices.iloc[i, predictions_with_prices.columns.get_loc('stock_price')] = current_week_close

        # Create summary table
        summary_data = []

        for _, row in predictions_with_prices.iterrows():
            row_data = {
                'Date': row['date'].strftime('%Y-%m-%d') if 'date' in row else 'Unknown',
                'Predicted Return': f"{row['predicted_return']:.4f}" if 'predicted_return' in row else 'N/A',
                'Return %': f"{row['predicted_return']*100:.2f}%" if 'predicted_return' in row else 'N/A'
            }

            # Add price if available
            if 'stock_price' in row:
                row_data['Predicted Price'] = f"${row['stock_price']:.2f}"

            # Add confidence intervals if available
            if 'ci_lower' in row and 'ci_upper' in row:
                row_data['CI Lower'] = f"{row['ci_lower']:.4f}"
                row_data['CI Upper'] = f"{row['ci_upper']:.4f}"

                if 'ci_lower_price' in row and 'ci_upper_price' in row:
                    row_data['Price CI'] = f"${row['ci_lower_price']:.2f} - ${row['ci_upper_price']:.2f}"

            # Add model info (from first row)
            if len(summary_data) == 0:
                row_data['Model Order'] = str(row.get('model_order', 'Unknown'))
                row_data['Model AIC'] = f"{row['model_aic']:.2f}" if 'model_aic' in row else 'N/A'

            summary_data.append(row_data)

        return pd.DataFrame(summary_data)

    except Exception as e:
        logger.error(f"Error creating summary table: {e}")
        return pd.DataFrame({'Error': [str(e)]})

def create_comparison_chart(predictions_dict: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create comparison chart for multiple tickers

    Args:
        predictions_dict: Dict mapping ticker -> predictions DataFrame

    Returns:
        Plotly Figure object
    """
    try:
        fig = go.Figure()

        if not predictions_dict:
            fig.add_annotation(
                text="No predictions to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Color palette
        colors = px.colors.qualitative.Set1

        for i, (ticker, predictions_df) in enumerate(predictions_dict.items()):
            if predictions_df.empty:
                continue

            try:
                # Convert to prices using simple logic
                predictions_with_prices = predictions_df.copy()

                # Simple price conversion
                try:
                    import yfinance as yf
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d")
                    if not hist.empty:
                        hist = hist.reset_index()  # Ensure proper handling
                        starting_price = float(hist['Close'].iloc[-1])
                    else:
                        starting_price = 100.0
                except Exception:
                    starting_price = 100.0

                # Convert returns to prices
                predictions_with_prices['stock_price'] = 0.0
                for j in range(len(predictions_with_prices)):
                    predicted_return = predictions_with_prices.iloc[j]['predicted_return']
                    if abs(predicted_return) < 1:
                        return_multiplier = predicted_return
                    else:
                        return_multiplier = predicted_return / 100

                    if j == 0:
                        week_close = starting_price * (1 + return_multiplier)
                        predictions_with_prices.iloc[j, predictions_with_prices.columns.get_loc('stock_price')] = week_close
                    else:
                        prev_week_close = predictions_with_prices.iloc[j-1]['stock_price']
                        current_week_close = prev_week_close * (1 + return_multiplier)
                        predictions_with_prices.iloc[j, predictions_with_prices.columns.get_loc('stock_price')] = current_week_close

                if 'stock_price' in predictions_with_prices.columns:
                    color = colors[i % len(colors)]

                    fig.add_trace(go.Scatter(
                        x=predictions_with_prices['date'],
                        y=predictions_with_prices['stock_price'],
                        mode='lines+markers',
                        name=ticker,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Price: $%{y:.2f}<br>' +
                                    '<extra></extra>'
                    ))

            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                continue

        # Customize layout
        fig.update_layout(
            title=dict(
                text='<b>Stock Price Predictions Comparison</b>',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)', tickformat='$.2f')

        return fig

    except Exception as e:
        logger.error(f"Error creating comparison chart: {e}")

        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating comparison chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

def create_returns_distribution_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """
    Create histogram of predicted returns

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        Plotly Figure object
    """
    try:
        fig = go.Figure()

        if predictions_df.empty or 'predicted_return' not in predictions_df.columns:
            fig.add_annotation(
                text="No return data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        returns = predictions_df['predicted_return'] * 100  # Convert to percentage

        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=20,
            name='Predicted Returns',
            marker_color='rgba(55, 128, 191, 0.7)',
            marker_line=dict(color='rgba(55, 128, 191, 1.0)', width=1)
        ))

        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0% Return")

        fig.update_layout(
            title='Distribution of Predicted Returns',
            xaxis_title='Predicted Return (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating returns distribution chart: {e}")

        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating distribution chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig