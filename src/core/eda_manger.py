import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, fft, stats
import statsmodels.api as sm
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List
from src.core.logger import logger

def morlet2(M, w=5.0, s=1.0):
    """
    Custom implementation of morlet2 wavelet function.
    
    Parameters:
        M: int - Length of the wavelet.
        w: float - Frequency of the wavelet.
        s: float - Scaling factor.
    
    Returns:
        np.ndarray - Morlet wavelet.
    """
    t = np.linspace(-M / 2, M / 2, M, endpoint=False)
    wavelet = np.exp(2j * np.pi * w * t / s) * np.exp(-t**2 / (2 * s**2))
    return wavelet

def filter_numeric_columns(df, exclude_columns=None):
    """
    Filter numeric columns from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        exclude_columns (list or None): List of column names to exclude from numeric filtering.
    
    Returns:
        pd.DataFrame: DataFrame containing only numeric columns (excluding specified ones).
    """
    exclude_columns = exclude_columns or []
    numeric_cols = []
    
    for col in df.columns:
        if col in exclude_columns:
            continue
        try:
            pd.to_numeric(df[col], errors='raise')  # Check if column is numeric
            numeric_cols.append(col)
        except ValueError:
            continue

    return df[numeric_cols]

class TimeSeriesAnalyzer:
    def __init__(self, data=None, date_column=None, value_column=None):
        self.data = data
        self.date_column = date_column
        self.value_column = value_column
        self.timeseries_dict = {}
        self.stats_dict = {}
        if all(v is not None for v in [data, date_column, value_column]):
            self.prepare_data()

    @staticmethod
    def is_date_column(col_name: str) -> bool:
        """Check if column name is in YYYY-MM-DD format"""
        try:
            datetime.strptime(col_name, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    @staticmethod
    def detect_date_columns(df: pd.DataFrame) -> List[str]:
        """Find all columns with dates in headers"""
        return [col for col in df.columns if TimeSeriesAnalyzer.is_date_column(col)]

    @staticmethod
    def create_key(row: pd.Series, text_columns: List[str]) -> str:
        """Create time series key from text columns"""
        return "-".join(str(row[col]) for col in text_columns if pd.notna(row[col]))

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze file and extract time series"""
        if not file_path.endswith('.csv'):
            raise ValueError("Unsupported file type")
        return self._analyze_csv(file_path)

    def _analyze_csv(self, file_path: str) -> Dict:
        """Analyze CSV file and extract time series"""
        df = pd.read_csv(file_path)
        date_columns = self.detect_date_columns(df)
        
        if not date_columns:
            raise ValueError("No date columns found")
        
        # Find text columns before first date column
        first_date_col_idx = min(df.columns.get_loc(col) for col in date_columns)
        text_columns = list(df.columns[:first_date_col_idx])
        
        # Create time series for each row
        for idx, row in df.iterrows():
            key = self.create_key(row, text_columns)
            values = row[date_columns].astype(float)
            dates = pd.to_datetime(date_columns)
            
            self.timeseries_dict[key] = {
                'dates': dates,
                'values': values,
                'original_row': row
            }
            
            self.stats_dict[key] = self._calculate_stats(values, dates)
        
        return self.timeseries_dict

    def prepare_data(self):
        """Prepare time series data for analysis"""
        if isinstance(self.data, pd.DataFrame):
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data = self.data.sort_values(self.date_column)
            self.data = self.data.set_index(self.date_column)
            self.data[self.value_column] = pd.to_numeric(self.data[self.value_column], errors='coerce')
            self.data = self.data.dropna(subset=[self.value_column])

    def perform_decomposition(self, period=12):
        """Perform time series decomposition"""
        if self.data is None or self.value_column is None:
            raise ValueError("Data and value column must be set before decomposition")
        return sm.tsa.seasonal_decompose(self.data[self.value_column], period=period)

    def perform_spectral_analysis(self, series=None):
        """Perform spectral analysis using FFT"""
        if series is None and self.data is not None and self.value_column is not None:
            values = self.data[self.value_column].values
        elif series is not None:
            values = series
        else:
            raise ValueError("No data available for spectral analysis")

        frequencies = fft.fftfreq(len(values))
        spectrum = np.abs(fft.fft(values))
        
        # Find main frequencies (excluding zero frequency)
        main_freq = frequencies[1:len(frequencies)//2]
        main_amp = spectrum[1:len(spectrum)//2]
        
        # Sort by amplitude
        sorted_idx = np.argsort(main_amp)[::-1]
        top_periods = 1/main_freq[sorted_idx][:5]
        top_amplitudes = main_amp[sorted_idx][:5]
        
        return {
            'frequencies': frequencies[1:len(frequencies)//2],
            'spectrum': spectrum[1:len(spectrum)//2],
            'top_periods': np.abs(top_periods),
            'top_amplitudes': top_amplitudes
        }

    def _calculate_stats(self, values: pd.Series, dates: pd.Series) -> Dict:
        """Calculate statistical characteristics of time series"""
        stats = {
            'duration': (dates.max() - dates.min()).days,
            'frequency': self._calculate_frequency(dates),
            'frequency_variance': np.std(np.diff(dates.astype(np.int64) // 10**9)),
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'quartiles': [values.quantile(q) for q in [0.25, 0.5, 0.75]]
        }
        
        # Add spectral analysis if possible
        try:
            spectral_stats = self.perform_spectral_analysis(values)
            stats.update({'spectral_analysis': spectral_stats})
        except Exception as e:
            stats['spectral_analysis_error'] = str(e)
            
        return stats

    def _calculate_frequency(self, dates: pd.Series) -> str:
        """Determine time series frequency"""
        diff = pd.Series(np.diff(dates.astype(np.int64) // 10**9))
        median_diff = diff.median()
        days = median_diff / (24 * 60 * 60)
        
        frequencies = {
            1: 'daily',
            7: 'weekly',
            30: 'monthly',
            90: 'quarterly',
            365: 'yearly'
        }
        
        for period, name in frequencies.items():
            if abs(days - period) < period * 0.1:
                return name
                
        return f'custom ({days:.1f} days)'

    def create_timeseries_table(self) -> pd.DataFrame:
        """Create table with DatetimeIndex where each time series is a column"""
        if not self.timeseries_dict:
            raise ValueError("No data available. Run analyze_file() first")
            
        all_dates = set()
        for ts in self.timeseries_dict.values():
            all_dates.update(ts['dates'])
        
        date_index = pd.DatetimeIndex(sorted(all_dates))
        df = pd.DataFrame(index=date_index)
        df['dates'] = date_index
        
        for key, ts in self.timeseries_dict.items():
            series = pd.Series(ts['values'].values, index=ts['dates'])
            df[key] = series
            
        return df.sort_index()

    def print_summary(self) -> None:
        """Print summary information for all time series"""
        if self.timeseries_dict:
            logger.info(f"Found time series: {len(self.timeseries_dict)}")
            for key, stats in self.stats_dict.items():
                logger.info(f"\nSeries: {key}")
                for stat_name, stat_value in stats.items():
                    if stat_name != 'spectral_analysis':
                        if isinstance(stat_value, (int, float)):
                            logger.info(f"{stat_name}: {stat_value:.2f}")
                        else:
                            logger.info(f"{stat_name}: {stat_value}")
        elif self.data is not None:
            logger.info("Single time series analysis:")
            logger.info(f"Total observations: {len(self.data)}")
            logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            desc = self.data[self.value_column].describe()
            logger.info("\nSummary statistics:")
            logger.info(desc)

class TimeSeriesRegressor:
    """
    Class for time series forecasting using various regression models
    """
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'SVR': SVR(kernel='rbf'),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.numeric_columns = None
        
    def get_numeric_columns(self, df):
        """Identifying numeric columns"""
        numeric_cols = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except:
                continue
        return numeric_cols
    
    def create_features(self, df, target_cols, date_col, lags=None, rolling_windows=None):
        """
        Creating features for forecasting
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Target variable
        date_col : str
            Column with dates
        lags : list
            List of lags for feature creation
        rolling_windows : list
            List of windows for rolling statistics
        """
        df = df.copy()
        # Data check and conversion
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Data conversion error: {str(e)}")
        df.set_index(date_col, inplace=True)
        
        # Identify numeric columns
        self.numeric_columns = self.get_numeric_columns(df)
        
        # Check if target columns are numeric
        for target_col in target_cols:
            if target_col not in self.numeric_columns:
                raise ValueError(f"Column {target_col} is not numeric")
        
        # Basic time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Create features for each target variable
        for target_col in target_cols:
            # Convert and handle missing values in target variable
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df[target_col] = df[target_col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # Lag features
            if lags is None:
                lags = [1, 7, 14, 30]
            
            for lag in lags:
                col_name = f'{target_col}_lag_{lag}'
                df[col_name] = df[target_col].shift(lag)
                
            # Rolling statistics
            if rolling_windows is None:
                rolling_windows = [7, 14, 30]
                
            for window in rolling_windows:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
                df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
                df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        # Fill missing values in features
        for col in df.columns:
            if col not in target_cols and col in self.numeric_columns:
                if 'rolling' in col or 'lag' in col:
                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        # Save feature names (excluding target variables)
        self.feature_names = [col for col in df.columns if col not in target_cols and col in self.numeric_columns]
        
        return df
    
    def prepare_data(self, df, target_col, train_size=0.8):
        """
        Preparing data for training
        """
        # Check if data exists
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Make sure feature_names are defined
        if self.feature_names is None:
            raise ValueError("Features must first be created using create_features")
        
        # Check if all columns exist
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Handle missing values
        df = df.copy()
        df[self.feature_names] = df[self.feature_names].apply(pd.to_numeric, errors='coerce')
        
        # Check for missing columns again
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Fill missing values
        for col in self.feature_names:
            if df[col].isnull().any():
                if 'rolling' in col or 'lag' in col:
                    df[col] = df[col].fillna(method='bfill')
                else:
                    df[col] = df[col].fillna(df[col].mean())
    
        # Remove rows where target variable is missing
        df = df.dropna(subset=[target_col])
        
        # Split into features and target
        X = df[self.feature_names]
        y = df[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Check for missing values after scaling
        if np.isnan(X_scaled).any():
            raise ValueError("Missing values appeared after scaling")
        
        # Split into training and test sets
        train_size = int(len(df) * train_size)
        X_train = X_scaled[:train_size]
        X_test = X_scaled[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test, X, y
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Training and evaluating models
        """
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            results[name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictions': y_pred_test,
                'model': model
            }
        
        # Select best model based on test MAE
        best_mae = float('inf')
        for name, result in results.items():
            if result['test_metrics']['MAE'] < best_mae:
                best_mae = result['test_metrics']['MAE']
                self.best_model = result['model']
                self.best_model_name = name
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculating quality metrics
        """
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
    
    def make_future_features(self, df, target_col, date_col, future_dates):
        """
        Creating features for future dates
        """
        last_date = df[date_col].max()
        future_df = pd.DataFrame({'date': future_dates})
        future_df[date_col] = pd.to_datetime(future_df['date'])
        future_df.set_index(date_col, inplace=True)
        
        # Create time features
        future_df['year'] = future_df.index.year
        future_df['month'] = future_df.index.month
        future_df['day'] = future_df.index.day
        future_df['dayofweek'] = future_df.index.dayofweek
        future_df['quarter'] = future_df.index.quarter
        
        # For lags and rolling statistics, use last known values
        for feature in self.feature_names:
            if feature not in future_df.columns:
                future_df[feature] = df[feature].iloc[-1]
        
        return future_df

class StatisticalForecaster:
    """Statistical forecasting models including ARIMA, SARIMA, ETS, and Prophet"""
    
    def __init__(self):
        self.models = {
            'Auto ARIMA': self._fit_auto_arima,
            'SARIMA': self._fit_sarima,
            'ETS': self._fit_ets,
            'Prophet': self._fit_prophet
        }
        self.fitted_model = None
        self.model_name = None
        self.forecast = None
        self.residuals = None
        
    def _prepare_data(self, data, date_column=None):
        """
        Prepare data for modeling.
        Converts DataFrame to Series if necessary.
        """
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, pd.DataFrame):
            if date_column is None:
                return data.iloc[:, 0]
            return data[date_column]
        else:
            raise ValueError("Data must be pandas Series or DataFrame")
            
    def analyze_stationarity(self, data):
        """Analyze time series stationarity"""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        # ADF test
        adf_result = adfuller(data)
        
        # KPSS test
        kpss_result = kpss(data)
        
        return {
            'adf_test': {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values': adf_result[4]
            },
            'kpss_test': {
                'statistic': kpss_result[0],
                'pvalue': kpss_result[1],
                'critical_values': kpss_result[3]
            }
        }
    
    def _fit_arima(self, data, order=(1, 1, 1)):
        """
        Fit an ARIMA model to the data.
        """
        model = ARIMA(data, order=order).fit()
        return model
        
    def _fit_sarima(self, data, **kwargs):
        """Fit SARIMA model"""
        order = kwargs.get('order', (1,1,1))
        seasonal_order = kwargs.get('seasonal_order', (1,1,1,12))
        
        model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order
        )
        return model.fit()
        
    def _fit_ets(self, data, **kwargs):
        """Fit ETS (Exponential Smoothing) model"""
        seasonal_periods = kwargs.get('seasonal_periods', None)
        
        model = ExponentialSmoothing(
            data,
            seasonal_periods=seasonal_periods,
            seasonal=kwargs.get('seasonal', 'add')
        )
        return model.fit()
        
    def _fit_prophet(self, data, **kwargs):
        """Fit Prophet model"""
        # Prepare data for Prophet
        if isinstance(data, pd.Series):
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
        else:
            df = data.copy()
            df.columns = ['ds', 'y']
            
        model = Prophet(
            yearly_seasonality=kwargs.get('yearly_seasonality', True),
            weekly_seasonality=kwargs.get('weekly_seasonality', True),
            daily_seasonality=kwargs.get('daily_seasonality', False)
        )
        model.fit(df)
        return model
        
    def fit(self, data, model_name, **kwargs):
        """Fit selected model to data"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        data_prepared = self._prepare_data(data)    
        self.model_name = model_name
        self.fitted_model = self.models[model_name](data, **kwargs)
        
        # Calculate residuals
        if model_name == 'Prophet':
            forecast = self.fitted_model.predict(
                pd.DataFrame({'ds': data.index})
            )
            self.residuals = data.values - forecast['yhat'].values
        else:
            self.residuals = self.fitted_model.resid()
            
        return self
        
    def predict(self, steps=30, return_conf_int=True, alpha=0.05):
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        if self.model_name == 'Prophet':
            future = self.fitted_model.make_future_dataframe(
                periods=steps,
                freq='D'
            )
            forecast = self.fitted_model.predict(future)
            result = {
                #'ds':forecast['ds'].values[-steps:],
                'forecast': forecast['yhat'].values[-steps:],
                'lower': forecast['yhat_lower'].values[-steps:],
                'upper': forecast['yhat_upper'].values[-steps:]
            }
        else:
            forecast = self.fitted_model.forecast(steps)
            if return_conf_int:
                conf_int = self.fitted_model.get_forecast(steps).conf_int(alpha=alpha)
                result = {
                    'forecast': forecast,
                    'lower': conf_int.iloc[:, 0],
                    'upper': conf_int.iloc[:, 1]
                }
            else:
                result = {
                    'forecast': forecast
                }
                
        self.forecast = result
        return result
        
    def get_diagnostics(self):
        """Get model diagnostics"""
        from scipy import stats
        
        if self.residuals is None:
            raise ValueError("Model must be fitted first")
            
        # Ljung-Box test for autocorrelation
        lb_test = stats.acorr_ljungbox(self.residuals, lags=[10], return_df=True)
        
        # Jarque-Bera test for normality
        jb_test = stats.jarque_bera(self.residuals)
        
        # Heteroskedasticity test
        het_test = stats.het_breuschpagan(self.residuals, np.ones_like(self.residuals))
        
        return {
            'normality_p_value': stats.normaltest(self.residuals).pvalue,
            'skewness': stats.skew(self.residuals),
            'kurtosis': stats.kurtosis(self.residuals),
            'ljung_box_test': {
                'statistic': lb_test['lb_stat'].values[0],
                'pvalue': lb_test['lb_pvalue'].values[0]
            },
            'jarque_bera_test': {
                'statistic': jb_test[0],
                'pvalue': jb_test[1]
            },
            'heteroskedasticity_test': {
                'statistic': het_test[0],
                'pvalue': het_test[1]
            }
        }
        
    def plot_forecast(self, data, date_index=None):
        """Plot the forecast with confidence intervals"""
        if self.forecast is None:
            raise ValueError("Must generate forecast first")
            
        fig = go.Figure()
        
        # Plot original data
        fig.add_trace(
            go.Scatter(
                x=date_index[:len(data)] if date_index is not None else np.arange(len(data)),
                y=data,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            )
        )
        
        # Plot forecast
        forecast_index = (
            date_index[len(data):] if date_index is not None 
            else np.arange(len(data), len(data) + len(self.forecast['forecast']))
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=self.forecast['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Add confidence intervals if available
        if 'lower' in self.forecast and 'upper' in self.forecast:
            fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=self.forecast['upper'],
                    mode='lines',
                    name='Upper CI',
                    line=dict(color='rgba(255,0,0,0.2)')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=self.forecast['lower'],
                    mode='lines',
                    name='Lower CI',
                    fill='tonexty',
                    line=dict(color='rgba(255,0,0,0.2)')
                )
            )
            
        fig.update_layout(
            title='Time Series Forecast',
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
        
    def plot_diagnostics(self):
        """Plot model diagnostics"""
        if self.residuals is None:
            raise ValueError("Model must be fitted first")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals',
                'Q-Q Plot',
                'Residuals Histogram',
                'Residuals ACF'
            )
        )
        
        # Residuals plot
        fig.add_trace(
            go.Scatter(
                y=self.residuals,
                mode='lines',
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # Q-Q plot
        qq = stats.probplot(self.residuals)
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name='Q-Q Plot'
            ),
            row=1, col=2
        )
        
        # Add the diagonal line
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][0] * qq[1][0] + qq[1][1],
                mode='lines',
                name='Normal Line',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(
                x=self.residuals,
                name='Histogram',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # ACF plot
        from statsmodels.stats.diagnostic import acf
        acf_values = acf(self.residuals, nlags=40)
        fig.add_trace(
            go.Bar(
                x=np.arange(len(acf_values)),
                y=acf_values,
                name='ACF'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig

class SpectralAnalyzer:
    """Class for time series spectral analysis including Fourier and Wavelet transforms"""
    
    def __init__(self, data=None):
        self.data = data
        self.sampling_rate = 1.0  # Default daily sampling
        
    def detrend_series(self, data, method='linear', window=None, degree=1):
        """
    Remove trend from a time series.
    
    Parameters:
        data: pd.Series - Time series data (values must be numeric).
        method: str - Method of detrending ('linear', 'polynomial', 'moving_average').
        window: int - Rolling window size (used for 'moving_average').
        degree: int - Polynomial degree (used for 'polynomial').
        
    Returns:
        pd.Series - Detrended time series.
    """
        # Ensure data is numeric
        data = pd.to_numeric(data, errors='coerce').dropna()
        
        if method == 'linear':
            x = np.arange(len(data))
            data = pd.to_numeric(data, errors='coerce').dropna() 
            coeffs = np.polyfit(x, data, 1)
            trend = np.polyval(coeffs, x)
            return pd.Series(data - trend, index=data.index)
        elif method == 'polynomial':
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, degree)
            trend = np.polyval(coeffs, x)
            return pd.Series(data - trend, index=data.index)
        elif method == 'moving_average':
            if window is None:
                window = len(data) // 10
            trend = pd.Series(data).rolling(window=window, center=True).mean()
            return pd.Series(data - trend.fillna(method='bfill').fillna(method='ffill'), index=data.index)
        return data
    
    def perform_fft(self, data, detrend=True, detrend_method='linear'):
        """Perform Fast Fourier Transform analysis"""
        if detrend:
            data = self.detrend_series(data, method=detrend_method)
            
        # Apply window function to reduce spectral leakage
        window = signal.windows.hann(len(data))
        windowed_data = data * window
        
        # Compute FFT
        frequencies = fft.fftfreq(len(data), d=1/self.sampling_rate)
        spectrum = np.abs(fft.fft(windowed_data))
        
        # Get positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        spectrum = spectrum[positive_freq_idx]
        
        # Find dominant frequencies
        peaks_idx = signal.find_peaks(spectrum, height=np.mean(spectrum))[0]
        sorted_peaks_idx = peaks_idx[np.argsort(spectrum[peaks_idx])[::-1]]
        
        dominant_freq = frequencies[sorted_peaks_idx[:5]]
        dominant_amp = spectrum[sorted_peaks_idx[:5]]
        dominant_periods = 1/dominant_freq
        
        return {
            'frequencies': frequencies,
            'spectrum': spectrum,
            'dominant_frequencies': dominant_freq,
            'dominant_amplitudes': dominant_amp,
            'dominant_periods': dominant_periods,
            'peaks_idx': peaks_idx
        }
    
    def perform_wavelet(self, data, wavelet='morlet', min_scale=1, max_scale=None):
        """
        Perform wavelet transform on time series data.
    
        Parameters:
            data: np.ndarray or pd.Series - Input time series data.
            wavelet: str - Type of wavelet to use ('morlet' or custom).
            min_scale: int - Minimum scale for wavelet transform.
            max_scale: int - Maximum scale for wavelet transform.
    
        Returns:
            dict - Wavelet transform results.
        """
        if max_scale is None:
            max_scale = len(data) // 4
            
        scales = np.arange(min_scale, max_scale)
        
        if wavelet == 'morlet':
            wavelet_function = morlet2
        else:
            wavelet_function = signal.ricker
            
        # Compute CWT
        cwt_matrix = signal.cwt(data, wavelet_function, scales)
        
        # Find dominant scales
        scale_powers = np.sum(np.abs(cwt_matrix), axis=1)
        peaks_idx = signal.find_peaks(scale_powers)[0]
        sorted_peaks_idx = peaks_idx[np.argsort(scale_powers[peaks_idx])[::-1]]
        
        dominant_scales = scales[sorted_peaks_idx[:5]]
        
        return {
            'cwt_matrix': cwt_matrix,
            'scales': scales,
            'time': np.arange(len(data)),
            'dominant_scales': dominant_scales,
            'scale_powers': scale_powers
        }
    
    def analyze_significance(self, detrended_data):
        """
        Analyze statistical significance of spectral peaks.

        Parameters:
            detrended_data: np.array or pd.Series - The detrended time series.

        Returns:
            dict - Results containing confidence levels and significant peaks.
        """
        # Perform FFT
        fft_results = self.perform_fft(detrended_data)
        spectrum = fft_results['spectrum']
        frequencies = fft_results['frequencies']

        # Generate confidence levels (example)
        confidence_levels = {
            '95%': np.random.uniform(0.8, 1.2, len(frequencies)),  # Replace with actual computation
            '99%': np.random.uniform(0.9, 1.3, len(frequencies))   # Replace with actual computation
        }

        # Align lengths if necessary
        if len(spectrum) != len(confidence_levels['95%']):
            from scipy.interpolate import interp1d

            # Interpolate confidence levels to match spectrum length
            confidence_levels['95%'] = interp1d(
                np.linspace(0, 1, len(confidence_levels['95%'])),
                confidence_levels['95%'],
                kind='linear',
                fill_value='extrapolate'
            )(np.linspace(0, 1, len(spectrum)))

            confidence_levels['99%'] = interp1d(
                np.linspace(0, 1, len(confidence_levels['99%'])),
                confidence_levels['99%'],
                kind='linear',
                fill_value='extrapolate'
            )(np.linspace(0, 1, len(spectrum)))

        # Determine significant peaks
        significant_peaks = spectrum > confidence_levels['95%']

        return {
            'confidence_levels': confidence_levels,
            'significant_peaks': significant_peaks
        }
    
    def create_fft_plot(self, fft_results):
        """Create FFT analysis plot"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Frequency Spectrum', 'Period Spectrum')
        )
        
        # Frequency spectrum
        fig.add_trace(
            go.Scatter(
                x=fft_results['frequencies'],
                y=fft_results['spectrum'],
                mode='lines',
                name='Spectrum'
            ),
            row=1, col=1
        )
        
        # Mark dominant frequencies
        fig.add_trace(
            go.Scatter(
                x=fft_results['dominant_frequencies'],
                y=fft_results['dominant_amplitudes'],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Dominant Frequencies'
            ),
            row=1, col=1
        )
        
        # Period spectrum
        periods = 1/fft_results['frequencies']
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=fft_results['spectrum'],
                mode='lines',
                name='Period Spectrum'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_wavelet_plot(self, wavelet_results):
        """Create wavelet analysis plot"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Wavelet Transform', 'Scale Power')
        )
        
        # Wavelet transform heatmap
        fig.add_trace(
            go.Heatmap(
                z=np.abs(wavelet_results['cwt_matrix']),
                x=wavelet_results['time'],
                y=wavelet_results['scales'],
                colorscale='Viridis',
                name='Wavelet Power'
            ),
            row=1, col=1
        )
        
        # Scale power
        fig.add_trace(
            go.Scatter(
                x=wavelet_results['scales'],
                y=wavelet_results['scale_powers'],
                mode='lines',
                name='Scale Power'
            ),
            row=2, col=1
        )
        
        # Mark dominant scales
        fig.add_trace(
            go.Scatter(
                x=wavelet_results['dominant_scales'],
                y=wavelet_results['scale_powers'][wavelet_results['dominant_scales'].astype(int)],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Dominant Scales'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig

    def select_order(self, data):
        # ACF/PACF analysis
        # Order selection
        pass

    def fit_predict(self, data, model_type='ARIMA', **params):
        pass

    def evaluate(self):
        # Specific evaluation methods
        pass
