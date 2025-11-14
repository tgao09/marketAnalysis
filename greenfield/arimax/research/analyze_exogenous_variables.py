# !/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='../dataset/stock_dataset_with_lags.csv'):
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def test_stationarity(series, variable_name):
    
    try:
        result = adfuller(series.dropna())
        return {
            'variable': variable_name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except Exception as e:
        return {
            'variable': variable_name,
            'error': str(e),
            'is_stationary': None
        }

def analyze_autocorrelation(series, variable_name, lags=20):
    
    try:
        clean_series = series.dropna()
        if len(clean_series) < lags + 5:
            return None

        autocorr = acf(clean_series, nlags=lags, fft=True)
        partial_autocorr = pacf(clean_series, nlags=lags)

        # find significant lags (beyond 95% confidence)
        significant_lags = []
        confidence_bound = 1.96 / np.sqrt(len(clean_series))

        for i in range(1, len(autocorr)):
            if abs(autocorr[i]) > confidence_bound:
                significant_lags.append(i)

        return {
            'variable': variable_name,
            'autocorr': autocorr,
            'partial_autocorr': partial_autocorr,
            'significant_lags': significant_lags[:5],  # top 5
            'ljung_box_test': acorr_ljungbox(clean_series, lags=10, return_df=True)
        }
    except Exception as e:
        return {'variable': variable_name, 'error': str(e)}

def analyze_distribution(series, variable_name):
    
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return {'variable': variable_name, 'error': 'No valid data'}

    return {
        'variable': variable_name,
        'mean': clean_series.mean(),
        'std': clean_series.std(),
        'skewness': stats.skew(clean_series),
        'kurtosis': stats.kurtosis(clean_series),
        'jarque_bera': stats.jarque_bera(clean_series),
        'min': clean_series.min(),
        'max': clean_series.max(),
        'count': len(clean_series)
    }

def analyze_volatility_clustering(series, variable_name):
    
    try:
        clean_series = series.dropna()

        # calculate squared residuals (proxy for volatility)
        residuals_squared = (clean_series - clean_series.mean()) ** 2

        # test autocorrelation in squared residuals
        ljung_box_volatility = acorr_ljungbox(residuals_squared, lags=10, return_df=True)

        return {
            'variable': variable_name,
            'volatility_clustering_test': ljung_box_volatility,
            'has_arch_effects': ljung_box_volatility['lb_pvalue'].iloc[0] < 0.05
        }
    except Exception as e:
        return {'variable': variable_name, 'error': str(e)}

def analyze_cross_correlations(df, base_variables):
    
    correlations = {}

    for ticker in df['ticker'].unique()[:5]:  # sample of 5 stocks
        ticker_data = df[df['ticker'] == ticker]

        if len(ticker_data) < 50:  # need sufficient data
            continue

        corr_matrix = ticker_data[base_variables].corr()
        correlations[ticker] = corr_matrix

    if correlations:
        # average correlation across stocks
        avg_corr = pd.concat(correlations.values()).groupby(level=0).mean()
        return avg_corr

    return None

def recommend_forecasting_model(analysis_results):
    
    recommendations = {}

    for var, results in analysis_results.items():
        if 'error' in results['stationarity']:
            recommendations[var] = "Error in analysis - use simple persistence"
            continue

        model_suggestions = []

        # check stationarity
        if results['stationarity']['is_stationary']:
            model_suggestions.append("ARIMA")
        else:
            model_suggestions.append("ARIMA with differencing")

        # check for volatility clustering
        if results.get('volatility', {}).get('has_arch_effects', False):
            model_suggestions.append("GARCH")

        # check autocorrelation patterns
        autocorr_data = results.get('autocorrelation', {})
        if autocorr_data and len(autocorr_data.get('significant_lags', [])) > 2:
            model_suggestions.append("AR/ARIMA with multiple lags")

        # distribution characteristics
        dist_data = results.get('distribution', {})
        if dist_data and abs(dist_data.get('skewness', 0)) > 1:
            model_suggestions.append("Robust regression (non-normal)")

        recommendations[var] = ", ".join(model_suggestions) if model_suggestions else "Simple persistence"

    return recommendations

def main():
    print("Analyzing Exogenous Variables for Forecasting")
    print("=" * 50)

    # load data
    df = load_data()
    print(f"Loaded dataset: {len(df)} records, {df['ticker'].nunique()} stocks")

    # define base exogenous variables (excluding lags)
    base_variables = ['high_return', 'low_return', 'volume_change', 'volatility']

    # analyze each variable across multiple stocks
    analysis_results = {}

    for variable in base_variables:
        print(f"\nAnalyzing {variable}...")

        # combine data across all stocks for general patterns
        all_data = df[variable].dropna()

        # perform various analyses
        stationarity = test_stationarity(all_data, variable)
        autocorr = analyze_autocorrelation(all_data, variable)
        distribution = analyze_distribution(all_data, variable)
        volatility = analyze_volatility_clustering(all_data, variable)

        analysis_results[variable] = {
            'stationarity': stationarity,
            'autocorrelation': autocorr,
            'distribution': distribution,
            'volatility': volatility
        }

        # print key findings
        print(f"  Stationary: {stationarity.get('is_stationary', 'Unknown')}")
        print(f"  Mean: {distribution.get('mean', 0):.4f}, Std: {distribution.get('std', 0):.4f}")
        if autocorr and 'significant_lags' in autocorr:
            print(f"  Significant lags: {autocorr['significant_lags']}")
        if volatility and 'has_arch_effects' in volatility:
            print(f"  ARCH effects: {volatility['has_arch_effects']}")

    # cross-correlation analysis
    print(f"\nCross-correlation Analysis:")
    cross_corr = analyze_cross_correlations(df, base_variables)
    if cross_corr is not None:
        print("Average correlations between variables:")
        print(cross_corr.round(3))

    # generate recommendations
    print(f"\nForecasting Model Recommendations:")
    print("-" * 40)
    recommendations = recommend_forecasting_model(analysis_results)

    for variable, recommendation in recommendations.items():
        print(f"{variable:15}: {recommendation}")

    # summary statistics
    print(f"\nSummary Statistics:")
    print("-" * 20)
    for variable in base_variables:
        dist_data = analysis_results[variable].get('distribution', {})
        if 'mean' in dist_data:
            print(f"{variable:15}: mean={dist_data['mean']:6.4f}, std={dist_data['std']:6.4f}, "
                  f"skew={dist_data['skewness']:6.2f}, kurt={dist_data['kurtosis']:6.2f}")

    return analysis_results, recommendations

if __name__ == "__main__":
    analysis_results, recommendations = main()