import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal, fft, stats
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import json
from fpdf import FPDF
import io
import base64
import os
from typing import Dict, List, Tuple, Optional
from src.core.eda_manger import TimeSeriesAnalyzer, TimeSeriesRegressor,  StatisticalForecaster, SpectralAnalyzer
from src.core.logger import logger

@st.cache_data
def load_registry_data():
    """Load data from the registry API"""
    response = requests.get("http://127.0.0.1:8000/data_sources")
    data = response.json()
    return pd.DataFrame(data)

def parse_metadata(metadata):
    """Parse metadata from JSON string or dict"""
    if isinstance(metadata, dict):
        return metadata
    try:
        return json.loads(metadata)
    except json.JSONDecodeError:
        return {}

def download_file(file_path):
    """Read file content from path"""
    with open(file_path, "rb") as f:
        return f.read()

def create_summary_visualization(filtered_table):
    """Create summary visualizations for the data sources"""
    # File types distribution
    fig1 = px.pie(filtered_table, names='file_type', title='Distribution of File Types')
    
    # File formats distribution
    fig2 = px.pie(filtered_table, names='format', title='Distribution of File Formats')
    
    # File sizes over time
    sizes = filtered_table["metadata"].apply(lambda x: parse_metadata(x).get("size_bytes", 0))
    timestamps = pd.to_datetime(filtered_table["timestamp"])
    fig3 = px.line(x=timestamps, y=sizes, title='File Sizes Over Time')
    fig3.update_layout(xaxis_title="Timestamp", yaxis_title="Size (bytes)")
    
    return fig1, fig2, fig3

def create_dataset_summary(csv_data):
    """Create summary visualizations for CSV datasets"""
    # Lines and columns distribution
    fig1 = go.Figure()
    
    # Lines distribution
    lines = csv_data["metadata"].apply(lambda x: parse_metadata(x).get("num_lines", 0))
    fig1.add_trace(go.Box(y=lines, name="Lines per File"))
    
    # Columns distribution
    columns = csv_data["metadata"].apply(lambda x: parse_metadata(x).get("num_columns", 0))
    fig1.add_trace(go.Box(y=columns, name="Columns per File"))
    
    fig1.update_layout(title="Dataset Size Distribution")
    
    return fig1

def create_decomposition_plot(data, date_column, value_column, period=12):
    """
    Creates an interactive plot of time series decomposition
    """
    # Data preparation
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')  # convert to numbers
    df = df.dropna(subset=[value_column])  # remove rows with missing values
    
    # Interpolate remaining missing values if any exist
    df[value_column] = df[value_column].interpolate(method='linear')
    
    # Decomposition
    decomposition = seasonal_decompose(df[value_column], period=period)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original Series + Trend', 'Trend', 'Seasonality', 'Residuals'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add original series
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df[value_column],
            name='Original Data',
            line=dict(color='royalblue', width=1)
        ),
        row=1, col=1
    )
    
    # Add trend to the first plot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=decomposition.trend,
            name='Trend',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Separate trend plot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=decomposition.trend,
            name='Trend (separate)',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Seasonality
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=decomposition.seasonal,
            name='Seasonality',
            line=dict(color='green', width=1)
        ),
        row=3, col=1
    )
    
    # Residuals
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=decomposition.resid,
            name='Residuals',
            line=dict(color='gray', width=1),
            mode='lines'
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text="Time Series Decomposition",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig, decomposition

def calculate_statistics(data, decomposition):
    """
    Calculates time series statistics
    """
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    resid = decomposition.resid.dropna()
    
    # Trend
    trend_change = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
    trend_volatility = trend.std() / trend.mean() * 100
    
    # Seasonality
    seasonal_strength = np.var(seasonal) / (np.var(seasonal) + np.var(resid))
    seasonal_amplitude = seasonal.max() - seasonal.min()
    
    # Residuals
    noise_ratio = resid.std() / data.std()
    
    return {
        'Trend Change (%)': trend_change,
        'Trend Volatility (%)': trend_volatility,
        'Seasonality Strength (0-1)': seasonal_strength,
        'Seasonal Amplitude': seasonal_amplitude,
        'Noise Ratio': noise_ratio
    }

def create_seasonal_pattern_plot(decomposition):
    """
    Creates a plot of the seasonal pattern
    """
    seasonal = pd.DataFrame(decomposition.seasonal)
    seasonal['month'] = seasonal.index.month
    monthly_pattern = seasonal.groupby('month').mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_pattern.index,
        y=monthly_pattern.values,
        mode='lines+markers',
        name='Seasonal Pattern'
    ))
    
    fig.update_layout(
        title='Average Seasonal Pattern by Month',
        xaxis_title='Month',
        yaxis_title='Seasonality Magnitude',
        template="plotly_white"
    )
    
    return fig

def create_forecast_plots(df, target_col, train_size, results, future_predictions=None):
    """
    Creating forecast plots
    """
    train_size = int(len(df) * train_size)
    train_dates = df.index[:train_size]
    test_dates = df.index[train_size:]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Forecasts', 'Forecast Errors'),
        vertical_spacing=0.15
    )
    
    # Forecast plot
    fig.add_trace(
        go.Scatter(
            x=train_dates,
            y=df[target_col][:train_size],
            name='Training Set',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=df[target_col][train_size:],
            name='Test Set',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
    for (name, result), color in zip(results.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=result['predictions'],
                name=f'Forecast {name}',
                line=dict(color=color, dash='dash')
            ),
            row=1, col=1
        )
        
        # Error plot
        errors = df[target_col][train_size:] - result['predictions']
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=errors,
                name=f'Errors {name}',
                line=dict(color=color)
            ),
            row=2, col=1
        )
    
    # Add future forecasts if they exist
    if future_predictions is not None:
        future_dates = future_predictions.index
        for (name, predictions), color in zip(future_predictions.items(), colors):
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=predictions,
                    name=f'Forecast {name} (future)',
                    line=dict(color=color, dash='dot')
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        height=800,
        title_text="Forecast Analysis",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig


def generate_pdf_report(analysis_results, forecasting_results, figures):
    """Generate PDF report with analysis results and visualizations"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Time Series Analysis Report', ln=True, align='C')
    pdf.ln(10)
    
    # Analysis Results
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Results', ln=True)
    pdf.set_font('Arial', '', 12)
    
    # Add decomposition results
    pdf.cell(0, 10, 'Decomposition Analysis:', ln=True)
    # Add analysis text here
    
    # Add spectral analysis results
    pdf.cell(0, 10, 'Spectral Analysis:', ln=True)
    # Add spectral analysis text here
    
    # Forecasting Results
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Forecasting Results', ln=True)
    
    # Add model performance metrics
    for model_name, results in forecasting_results.items():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'{model_name} Results:', ln=True)
        pdf.set_font('Arial', '', 12)
        # Add metrics text here
    
    # Add plots
    for title, fig in figures.items():
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, title, ln=True)
        # Convert plotly figure to image and add to PDF
        img_bytes = fig.to_image(format="png")
        pdf.image(io.BytesIO(img_bytes), x=10, y=pdf.get_y(), w=190)
    
    return pdf

def render_analysis(file_path, file_details, metadata):
    """
    Render comprehensive time series analysis for CSV files.
    """
    try:
        # Load and display initial data
        df = pd.read_csv(file_path)
        
        with st.expander("File Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", file_details["file_type"])
            with col2:
                st.metric("Format", file_details["format"])
            with col3:
                st.metric(
                    "Size", 
                    f"{metadata.get('size_bytes', 0)/1024:.2f} KB"
                )
            
            # Additional metadata display
            if metadata.get("num_lines"):
                st.metric("Total Lines", metadata["num_lines"])
            if metadata.get("num_columns"):
                st.metric("Total Columns", metadata["num_columns"])

        # Data Preview Section
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head())
            
            # Data info
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Data Types")
                st.write(df.dtypes)
            with col2:
                st.write("### Missing Values")
                st.write(df.isnull().sum())

        # Initialize analyzers
        ts_analyzer = TimeSeriesAnalyzer()
        
        # Get time series data
        timeseries_dict = ts_analyzer.analyze_file(file_path)
        ts_data = ts_analyzer.create_timeseries_table()
        
        # Time Series Selection
        st.header("Time Series Analysis")
        available_series = [col for col in ts_data.columns if col != 'dates']
        selected_series = st.multiselect(
            "Select Time Series for Analysis",
            available_series,
            default=available_series[:1]
        )
        
        if not selected_series:
            st.warning("Please select at least one time series to analyze")
            return
            
        # Main Analysis Tabs
        tab1, tab2, tab3 = st.tabs([
            "Trend and Seasonality Analysis",
            "Spectral Analysis",
            "Basic Forecasting"#, 
            #"Statistical Forecasting"
        ])
        
        with tab1:
            st.subheader("Trend and Seasonality Analysis")
            
            # Analysis parameters
            col1, col2 = st.columns(2)
            with col1:
                decomposition_period = st.slider(
                    "Decomposition Period", 
                    min_value=2,
                    max_value=52,
                    value=12
                )
            with col2:
                ma_window = st.slider(
                    "Moving Average Window",
                    min_value=2,
                    max_value=30,
                    value=7
                )
            
            # Analyze each selected series
            for series_name in selected_series:
                st.write(f"### Analysis of {series_name}")
                
                # Create decomposition plots
                fig_decomp, decomposition = create_decomposition_plot(
                    ts_data,
                    'dates',
                    series_name,
                    decomposition_period
                )
                st.plotly_chart(fig_decomp, use_container_width=True)
                
                # Statistics
                stats = calculate_statistics(ts_data[series_name], decomposition)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Trend Change",
                        f"{stats['Trend Change (%)']:.2f}%"
                    )
                    st.metric(
                        "Trend Volatility",
                        f"{stats['Trend Volatility (%)']:.2f}%"
                    )
                with col2:
                    st.metric(
                        "Seasonality Strength",
                        f"{stats['Seasonality Strength (0-1)']:.3f}"
                    )
                    st.metric(
                        "Seasonal Amplitude",
                        f"{stats['Seasonal Amplitude']:.2f}"
                    )
                with col3:
                    st.metric(
                        "Noise Ratio",
                        f"{stats['Noise Ratio']:.3f}"
                    )
                
                # Seasonal pattern
                st.write("#### Seasonal Pattern")
                seasonal_fig = create_seasonal_pattern_plot(decomposition)
                st.plotly_chart(seasonal_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Spectral Analysis")
            
            for series_name in selected_series:
                st.write(f"### Spectral Analysis of {series_name}")
                
                # Perform spectral analysis
                spectral_results = ts_analyzer.perform_spectral_analysis(
                    ts_data[series_name].values
                )
                
                # Plot spectrum
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=spectral_results['frequencies'],
                    y=spectral_results['spectrum'],
                    mode='lines',
                    name='Spectrum'
                ))
                fig.update_layout(
                    title="Frequency Spectrum",
                    xaxis_title="Frequency",
                    yaxis_title="Amplitude"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display main periodicities
                st.write("#### Main Periodicities")
                periodicity_df = pd.DataFrame({
                    'Period': spectral_results['top_periods'],
                    'Amplitude': spectral_results['top_amplitudes']
                })
                st.dataframe(periodicity_df)
        
        with tab3:
            st.subheader("Time Series Forecasting")
            
            # Forecasting settings
            with st.expander("Forecasting Settings", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    train_size = st.slider(
                        "Training Data Size",
                        0.5, 0.9, 0.8
                    )
                    forecast_horizon = st.number_input(
                        "Forecast Horizon",
                        1, 365, 30
                    )
                with col2:
                    selected_models = st.multiselect(
                        "Select Models",
                        [
                            'Linear Regression',
                            'Ridge',
                            'Lasso',
                            'ElasticNet',
                            'SVR',
                            'Random Forest'
                        ],
                        default=['Linear Regression', 'Random Forest']
                    )
            
            # Feature engineering settings
            with st.expander("Feature Engineering Settings"):
                lags = st.multiselect(
                    'Lags (days)',
                    [1, 7, 14, 30, 60, 90],
                    default=[1, 7, 14, 30]
                )
                
                windows = st.multiselect(
                    'Rolling Windows (days)',
                    [7, 14, 30, 60, 90],
                    default=[7, 14, 30]
                )
            
            if st.button("Run Forecasting"):
                for series_name in selected_series:
                    st.write(f"### Forecasting {series_name}")
                    
                    # Initialize forecaster
                    forecaster = TimeSeriesRegressor()
                    # Create features and prepare data
                    df_features = forecaster.create_features(
                            ts_data.reset_index(),  # Reset index to get date column back
                            [series_name],         # target_cols as list
                            'dates',              # date_col
                            lags=lags,
                            rolling_windows=windows
                        )
                    
                    # Filter selected models
                    forecaster.models = {
                        name: model 
                        for name, model in forecaster.models.items()
                        if name in selected_models
                    }
                    
                    # Prepare and split data
                    X_train, X_test, y_train, y_test, X, y = forecaster.prepare_data(
                            df_features,         # df
                            series_name,        # target_col
                            train_size         # train_size
                        )
                    
                    # Train and evaluate
                    results = forecaster.train_and_evaluate(
                        X_train, X_test, y_train, y_test
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Training Metrics")
                        train_metrics = pd.DataFrame([
                            res['train_metrics'] 
                            for res in results.values()
                        ], index=results.keys())
                        st.dataframe(train_metrics)
                    
                    with col2:
                        st.write("#### Test Metrics")
                        test_metrics = pd.DataFrame([
                            res['test_metrics'] 
                            for res in results.values()
                        ], index=results.keys())
                        st.dataframe(test_metrics)
                    
                    # Plot forecasts
                    fig = create_forecast_plots(
                        df_features,
                        series_name,
                        train_size,
                        results
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Best model info
                    st.write(f"#### Best Model: {forecaster.best_model_name}")
            
            # Export options
            if st.button("Export Analysis Report"):
                figures = {
                    "Decomposition": fig_decomp,
                    "Spectral": fig,
                    "Forecasts": fig
                }
                pdf = generate_pdf_report(
                    ts_analyzer.stats_dict,
                    ts_data,
                    file_details,
                    figures
                )
                
                st.download_button(
                    "Download Report",
                    data=pdf.output(dest='S').encode('latin-1'),
                    file_name=f"time_series_analysis_{os.path.basename(file_path)}.pdf",
                    mime="application/pdf"
                )
    
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
        st.exception(e)

def render_spectral_analysis(file_path, detrend=True, analyze_significance=True):
    """Render spectral analysis section"""
    ts_analyzer = TimeSeriesAnalyzer()
    timeseries_dict = ts_analyzer.analyze_file(file_path)
    data = ts_analyzer.create_timeseries_table()
    analyzer = SpectralAnalyzer()
    
    # Select time series for analysis
    st.subheader("Time Series Selection")
    available_series = [col for col in data.columns if col != 'dates']  # Exclude non-series columns
    selected_series = st.multiselect(
        "Select Time Series for Analysis",
        available_series,
        default=available_series[:1],  # Select the first series by default
        key = 'spectral_series'
    )

    if not selected_series:
        st.warning("Please select at least one time series to analyze")
        return
    
    # Analysis settings
    with st.expander("Analysis Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            detrend_method = st.selectbox(
                "Detrending Method",
                ["linear", "polynomial", "moving_average"]
            )
            if detrend_method == "polynomial":
                degree = st.slider("Polynomial Degree", 1, 5, 2)
            elif detrend_method == "moving_average":
                window = st.slider("Window Size", 5, len(data)//2, len(data)//10)
                
        with col2:
            wavelet_type = st.selectbox(
                "Wavelet Type",
                ["morlet", "ricker"]
            )
            min_scale = st.number_input("Minimum Scale", 1, 10, 1)
            max_scale = st.number_input(
                "Maximum Scale",
                min_scale + 1,
                len(data)//2,
                len(data)//4
            )
    # Perform analysis for each selected series
    for series_name in selected_series:
        st.subheader(f"Spectral Analysis for {series_name}")
        # Extract series data
        series_data = data[series_name]
        if detrend:
            if detrend_method == "polynomial":
                detrended_data = analyzer.detrend_series(data, method=detrend_method, degree=degree)
            elif detrend_method == "moving_average":
                detrended_data = analyzer.detrend_series(data, method=detrend_method, window=window)
            else:
                detrended_data = analyzer.detrend_series(data, method=detrend_method)
        else:
            detrended_data = data
        # Fourier Analysis
        st.subheader("Fourier Analysis")
        fft_results = analyzer.perform_fft(detrended_data, detrend=False)
    
        # Show FFT plot
        st.plotly_chart(
            analyzer.create_fft_plot(fft_results),
            use_container_width=True
        )
    
        # Display dominant periods
        st.write("#### Main Periodicities")
        periods_df = pd.DataFrame({
            'Period': fft_results['dominant_periods'],
            'Frequency': fft_results['dominant_frequencies'],
            'Amplitude': fft_results['dominant_amplitudes']
        })
        st.dataframe(periods_df)
    
        # Statistical Significance
        if analyze_significance:
            st.subheader("Statistical Significance Analysis")
            significance_results = analyzer.analyze_significance(detrended_data)
        
            # Plot with confidence intervals
            fig = go.Figure()
        
            fig.add_trace(
                go.Scatter(
                    x=fft_results['frequencies'],
                    y=significance_results['confidence_levels']['95%'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255,0,0,0.2)',
                    name='95% Confidence'
                )
            )
        
            fig.add_trace(
                go.Scatter(
                    x=fft_results['frequencies'],
                    y=significance_results['confidence_levels']['99%'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255,0,0,0.1)',
                    name='99% Confidence'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=fft_results['frequencies'],
                    y=fft_results['spectrum'],
                    mode='lines',
                    name='Spectrum'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        # Wavelet Analysis
        st.write("### Wavelet Analysis")
        wavelet_results = analyzer.perform_wavelet(
            detrended_data,
            wavelet=wavelet_type,
            min_scale=min_scale,
            max_scale=max_scale
        )

        st.plotly_chart(
            analyzer.create_wavelet_plot(wavelet_results),
            use_container_width=True
        )

        # Interactive scale analysis
        if st.checkbox("Analyze Specific Scale"):
            scale = st.slider(
                "Select Scale",
                int(min_scale),
                int(max_scale - 1),
                int(min_scale)
            )

            # Plot time series at selected scale
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    y=np.abs(wavelet_results['cwt_matrix'][scale - min_scale]),
                    mode='lines',
                    name=f'Scale {scale}'
                )
            )

            fig.update_layout(
                title=f"Time Series at Scale {scale}",
                xaxis_title="Time",
                yaxis_title="Amplitude"
            )

            st.plotly_chart(fig, use_container_width=True)

def render_statistical_forecasting(file_path, date_column=None):
    """Render statistical forecasting section"""
    ts_analyzer = TimeSeriesAnalyzer()
    timeseries_dict = ts_analyzer.analyze_file(file_path)
    data = ts_analyzer.create_timeseries_table()
    st.subheader("Statistical Forecasting")
    
    # Model selection and parameters
    with st.expander("Model Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model",
                ['Auto ARIMA', 'SARIMA', 'ETS', 'Prophet']
            )
            
        with col2:
            forecast_steps = st.number_input(
                "Forecast Horizon",
                min_value=1,
                max_value=365,
                value=30
            )
    
        # Model-specific parameters
        if model_type == 'SARIMA':
            col1, col2 = st.columns(2)
            with col1:
                p = st.number_input("p (AR order)", 0, 5, 1)
                d = st.number_input("d (Difference order)", 0, 2, 1)
                q = st.number_input("q (MA order)", 0, 5, 1)
            with col2:
                P = st.number_input("P (Seasonal AR order)", 0, 2, 1)
                D = st.number_input("D (Seasonal difference order)", 0, 1, 1)
                Q = st.number_input("Q (Seasonal MA order)", 0, 2, 1)
            seasonal_period = st.number_input("Seasonal Period", 1, 52, 12)
            
            model_params = {
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, seasonal_period)
            }
            
        elif model_type == 'ETS':
            seasonal_period = st.number_input("Seasonal Period", 1, 52, 12)
            seasonal_type = st.selectbox(
                "Seasonal Type",
                ['add', 'mul', None]
            )
            
            model_params = {
                'seasonal_periods': seasonal_period,
                'seasonal': seasonal_type
            }
            
        elif model_type == 'Prophet':
            col1, col2 = st.columns(2)
            with col1:
                yearly_seasonality = st.checkbox("Yearly Seasonality", True)
                weekly_seasonality = st.checkbox("Weekly Seasonality", True)
            with col2:
                daily_seasonality = st.checkbox("Daily Seasonality", False)
                
            model_params = {
                'yearly_seasonality': yearly_seasonality,
                'weekly_seasonality': weekly_seasonality,
                'daily_seasonality': daily_seasonality
            }
            
        elif model_type == 'ARIMA':
            col1, col2 = st.columns(2)
            with col1:
                p = st.number_input("p (AR order)", 0, 5, 1)
                d = st.number_input("d (Difference order)", 0, 2, 1)
            with col2:
                q = st.number_input("q (MA order)", 0, 5, 1)

            model_params = {
                'order': (p, d, q)
            }
    
    # Fit model and generate forecast
    if st.button("Generate Forecast"):
        with st.spinner("Fitting model and generating forecast..."):
            try:
                forecaster = StatisticalForecaster()
                
                # Check and prepare data
                if date_column is not None:
                    ts_data = pd.Series(data[data.columns[0]].values, index=pd.to_datetime(data[date_column]))
                else:
                    ts_data = pd.Series(data)
                
                # Stationarity analysis
                stationarity_results = forecaster.analyze_stationarity(ts_data)
                
                st.write("### Stationarity Tests")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**ADF Test**")
                    st.write(f"Statistic: {stationarity_results['adf_test']['statistic']:.4f}")
                    st.write(f"p-value: {stationarity_results['adf_test']['pvalue']:.4f}")
                with col2:
                    st.write("**KPSS Test**")
                    st.write(f"Statistic: {stationarity_results['kpss_test']['statistic']:.4f}")
                    st.write(f"p-value: {stationarity_results['kpss_test']['pvalue']:.4f}")
                
                # Fit model and generate forecast
                forecaster.fit(ts_data, model_type, **model_params)
                forecast_results = forecaster.predict(steps=forecast_steps)
                
                # Plot results
                st.write("### Forecast Results")
                fig = forecaster.plot_forecast(ts_data, date_index=ts_data.index)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model diagnostics
                st.write("### Model Diagnostics")
                
                # Plot diagnostic charts
                diag_fig = forecaster.plot_diagnostics()
                st.plotly_chart(diag_fig, use_container_width=True)
                
                # Statistical tests
                diagnostics = forecaster.get_diagnostics()
                
                st.write("### Diagnostic Tests")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Ljung-Box Test (Autocorrelation)**")
                    st.write(f"Normality_p_value', {diagnostics['normality_p_value']}")
                    st.write(f"Skewness: {diagnostics['skewness']}")
                    st.write(f"Kurtosis: {diagnostics['kurtosis']}")
                with col2:
                    st.write("**Ljung-Box Test (Autocorrelation)**")
                    st.write(f"Statistic: {diagnostics['ljung_box_test']['statistic']:.4f}")
                    st.write(f"p-value: {diagnostics['ljung_box_test']['pvalue']:.4f}")
                    
                    st.write("**Jarque-Bera Test (Normality)**")
                    st.write(f"Statistic: {diagnostics['jarque_bera_test']['statistic']:.4f}")
                    st.write(f"p-value: {diagnostics['jarque_bera_test']['pvalue']:.4f}")
                
                with col3:
                    st.write("**Heteroskedasticity Test**")
                    st.write(f"Statistic: {diagnostics['heteroskedasticity_test']['statistic']:.4f}")
                    st.write(f"p-value: {diagnostics['heteroskedasticity_test']['pvalue']:.4f}")
                
                # Export results
                if st.button("Export Results"):
                    # Create DataFrame with results
                    forecast_df = pd.DataFrame({
                        'Forecast': forecast_results['forecast'],
                        'Lower CI': forecast_results.get('lower', None),
                        'Upper CI': forecast_results.get('upper', None)
                    })
                    if date_column is not None:
                        # Add future dates
                        last_date = ts_data.index[-1]
                        future_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=forecast_steps,
                            freq='D'
                        )
                        forecast_df.index = future_dates
                    
                    # Convert to CSV
                    csv = forecast_df.to_csv()
                    
                    # Create download button
                    st.download_button(
                        label="Download Forecast Results",
                        data=csv,
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
                st.exception(e)

def render():
    """Main render function for the enhanced analysis interface"""
    st.title("Enhanced Time Series Analysis")
    
    # Load registry data
    table = load_registry_data()
    
    # Sidebar filters
    st.sidebar.header("Data Source Filters")
    file_types = table["file_type"].unique()
    selected_types = st.sidebar.multiselect(
        "Select File Types", 
        file_types, 
        default=file_types
    )
    
    # Format filter
    formats = table["format"].unique()
    selected_formats = st.sidebar.multiselect(
        "Select Formats", 
        formats, 
        default=formats
    )
    # Additional filters
    with st.sidebar.expander("Advanced Filters"):
        # Size filter
        size_range = st.slider(
            "File Size (MB)",
            0.0,
            max(table["metadata"].apply(
                lambda x: parse_metadata(x).get("size_bytes", 0)
            )) / (1024 * 1024),
            (0.0, 100.0)
        )
        
        # Date filter
        date_range = st.date_input(
            "Date Range",
            value=(
                pd.to_datetime(table["timestamp"]).min(),
                pd.to_datetime(table["timestamp"]).max()
            )
        )
    # Apply filters
    filtered_table = table[
        (table["file_type"].isin(selected_types)) &
        (table["format"].isin(selected_formats)) &
        (table["metadata"].apply(lambda x: 
            parse_metadata(x).get("size_bytes", 0) / (1024 * 1024)
        ).between(*size_range)) &
        (pd.to_datetime(table["timestamp"]).dt.date.between(*date_range))
    ]
    
    # Data Source Overview Section
    with st.expander("Data Source Overview", expanded=True):
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(table))
        with col2:
            st.metric("Filtered Files", len(filtered_table))
        with col3:
            total_size = filtered_table["metadata"].apply(
                lambda x: parse_metadata(x).get("size_bytes", 0)
            ).sum()
            st.metric("Total Size", f"{total_size/1024/1024:.2f} MB")
        with col4:
            unique_formats = len(filtered_table["format"].unique())
            st.metric("Unique Formats", unique_formats)
        
        # Summary visualizations
        tab1, tab2 = st.tabs(["Distribution Charts", "Dataset Statistics"])
        
        with tab1:
            fig1, fig2, fig3 = create_summary_visualization(filtered_table)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            csv_data = filtered_table[filtered_table["format"] == "csv"]
            if not csv_data.empty:
                st.subheader("CSV Dataset Statistics")
                fig = create_dataset_summary(csv_data)
                st.plotly_chart(fig, use_container_width=True)
    
    # File Selection Section
    st.header("File Analysis")
    # Enhanced file selector
    col1, col2 = st.columns([3, 1])
    with col1:
        file_paths = filtered_table["file_path"].tolist()
        selected_file = st.selectbox(
            "Select File for Analysis",
            file_paths,
            key="file_selector"
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Name", "Date", "Size", "Format"]
        )
        if sort_by == "Name":
            file_paths.sort()
        elif sort_by == "Date":
            file_paths.sort(key=lambda x: filtered_table[
                filtered_table["file_path"] == x
            ]["timestamp"].iloc[0])
        elif sort_by == "Size":
            file_paths.sort(key=lambda x: parse_metadata(
                filtered_table[filtered_table["file_path"] == x]["metadata"].iloc[0]
            ).get("size_bytes", 0))
        elif sort_by == "Format":
            file_paths.sort(key=lambda x: filtered_table[
                filtered_table["file_path"] == x
            ]["format"].iloc[0])
    
    if selected_file:
        file_details = filtered_table[
            filtered_table["file_path"] == selected_file
        ].iloc[0]
        metadata = parse_metadata(file_details["metadata"])
        
        # Different handling for CSV and non-CSV files
        if file_details["format"] == "csv":
                try:
                    # Initialize analyzers
                    ts_analyzer = TimeSeriesAnalyzer()
        
                    # Get time series data
                    timeseries_dict = ts_analyzer.analyze_file(selected_file)
                    df = ts_analyzer.create_timeseries_table()
                    
        
                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
        
                    # Analysis tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Basic Analysis",
                        "Spectral Analysis",
                        "Statistical Forecasting",
                        "ML-based Forecasting"
                    ])
        
                    with tab1:
                        render_analysis(selected_file, file_details, metadata)
            
                    with tab2:
                        # Detect date and value columns if possible
                        date_cols = [col for col in df.columns if "date" in col.lower()]
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
            
                        # Column selection for spectral analysis
                        col1, col2 = st.columns(2)
                        with col1:
                            date_col = st.selectbox(
                                "Select Date Column",
                                date_cols if date_cols else df.columns,
                                key="spectral_date_col"
                            )
                        with col2:
                            value_col = st.selectbox(
                                "Select Value Column",
                                [col for col in numeric_cols if col != date_col],
                                key="spectral_value_col"
                            )
            
                        render_spectral_analysis(selected_file, date_col, value_col)
            
                    with tab3:
                        # Column selection for statistical forecasting
                        col1, col2 = st.columns(2)
                        with col1:
                            date_col = st.selectbox(
                                "Select Date Column",
                                date_cols if date_cols else df.columns,
                                key="stat_date_col"
                            )
                        with col2:
                            value_col = st.selectbox(
                                "Select Value Column",
                                [col for col in numeric_cols if col != date_col],
                                key="stat_value_col"
                            )
            
                        # Prepare data for forecasting
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.sort_values(date_col)
                        ts_data = df[value_col]
            
                        render_statistical_forecasting(selected_file, date_col)
            
                    with tab4:
                        # Column selection for ML forecasting
                        col1, col2 = st.columns(2)
                        with col1:
                            date_col = st.selectbox(
                                "Select Date Column",
                                date_cols if date_cols else df.columns,
                                key="ml_date_col"
                            )
                        with col2:
                            value_col = st.selectbox(
                                "Select Value Column",
                                [col for col in numeric_cols if col != date_col],
                                key="ml_value_col"
                            )
            
                        #render_ml_analysis(df, date_col, value_col)
            
                except Exception as e:
                    st.error(f"Error analyzing CSV file: {str(e)}")
                    st.exception(e)
           
        else:
            # Handle non-CSV files
            st.write("### File Preview")
            try:
                # Try to read and display file preview based on format
                if file_details["format"] in ["txt", "json", "md"]:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        st.code(
                            content[:1000] + ("..." if len(content) > 1000 else ""),
                            language=file_details["format"]
                        )
                else:
                    st.info("Preview not available for this file format")
                
                # Add download button
                file_content = download_file(selected_file)
                st.download_button(
                    "Download File",
                    data=file_content,
                    file_name=os.path.basename(selected_file),
                    mime="application/octet-stream"
                )
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    render()