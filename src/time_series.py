"""
Time Series Forecasting Module
===============================
Uses Prophet to build expected price ranges over time, then flags
events that fall outside the forecast confidence interval.

This is fundamentally different from point-in-time anomaly detection:
- Isolation Forest: "Is this single data point weird?"
- Time Series: "Is this data point weird GIVEN THE TREND?"

A gradually increasing price might not trigger statistical alarms
but the trend itself could be anomalous.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Prophet is optional - graceful degradation if not installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Time series forecasting will be disabled.")


class TimeSeriesForecaster:
    """
    Prophet-based time series forecaster for pricing anomaly detection.
    
    Approach:
    1. Aggregate prices to hourly level per category
    2. Fit Prophet model with daily + weekly seasonality
    3. Generate forecast with confidence intervals
    4. Flag events outside the confidence interval
    
    Why Prophet?
    - Handles missing data gracefully
    - Captures multiple seasonality patterns
    - Provides uncertainty intervals (not just point forecasts)
    - Works well with 30-90 days of history
    - Interpretable components (trend, seasonality, holidays)
    """
    
    def __init__(self, config: Optional[dict] = None):
        prophet_config = (config or {}).get('prophet', {})
        self.changepoint_prior_scale = prophet_config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = prophet_config.get('seasonality_prior_scale', 10.0)
        self.interval_width = prophet_config.get('interval_width', 0.95)
        self.models: Dict[str, Prophet] = {}
        self.forecasts: Dict[str, pd.DataFrame] = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'TimeSeriesForecaster':
        """
        Fit Prophet models per category on hourly aggregated data.
        """
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping time series fitting")
            self._is_fitted = True
            return self
        
        logger.info(f"Fitting time series models on {len(df)} events")
        
        df_work = df.copy()
        df_work['timestamp'] = pd.to_datetime(df_work['timestamp'])
        
        for category in df_work['category'].unique():
            cat_data = df_work[df_work['category'] == category].copy()
            
            # Aggregate to hourly
            cat_data.set_index('timestamp', inplace=True)
            hourly = cat_data['final_price'].resample('h').agg(['mean', 'std', 'count'])
            hourly = hourly[hourly['count'] >= 1].copy()
            
            if len(hourly) < 48:  # Need at least 2 days of data
                logger.warning(f"Insufficient data for {category}, skipping")
                continue
            
            # Prophet requires 'ds' and 'y' columns
            prophet_df = pd.DataFrame({
                'ds': hourly.index,
                'y': hourly['mean']
            }).reset_index(drop=True)
            
            # Remove extreme outliers before fitting (>3 sigma)
            y_mean = prophet_df['y'].mean()
            y_std = prophet_df['y'].std()
            prophet_df = prophet_df[
                (prophet_df['y'] > y_mean - 3 * y_std) & 
                (prophet_df['y'] < y_mean + 3 * y_std)
            ]
            
            # Fit Prophet model
            model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                interval_width=self.interval_width,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Not enough data for yearly
            )
            
            # Suppress Prophet's verbose logging
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            
            model.fit(prophet_df)
            
            # Generate in-sample forecast for anomaly detection
            forecast = model.predict(prophet_df)
            
            self.models[category] = model
            self.forecasts[category] = forecast
            
            logger.info(
                f"  {category}: fitted on {len(prophet_df)} hourly points, "
                f"trend range [{forecast['yhat'].min():.2f}, {forecast['yhat'].max():.2f}]"
            )
        
        self._is_fitted = True
        logger.info(f"Fitted {len(self.models)} time series models")
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if pricing events fall within Prophet's confidence interval.
        
        Fully vectorized — uses merge-based lookups instead of row-by-row
        iteration. On 100K events this runs in seconds instead of minutes.
        
        Returns DataFrame with:
        - ts_anomaly: bool (price outside confidence interval)
        - ts_score: float [0, 1] (how far outside the interval)
        - ts_expected: float (Prophet's point forecast)
        - ts_lower: float (lower confidence bound)
        - ts_upper: float (upper confidence bound)
        """
        results = pd.DataFrame(index=df.index)
        results['ts_anomaly'] = False
        results['ts_score'] = 0.0
        results['ts_expected'] = np.nan
        results['ts_lower'] = np.nan
        results['ts_upper'] = np.nan
        
        if not PROPHET_AVAILABLE or not self.models:
            logger.warning("Time series models not available, returning defaults")
            return results
        
        df_work = df.copy()
        df_work['timestamp'] = pd.to_datetime(df_work['timestamp'])
        df_work['_hour'] = df_work['timestamp'].dt.floor('h')
        df_work['_orig_idx'] = df_work.index
        
        for category in df_work['category'].unique():
            if category not in self.models:
                continue
            
            model = self.models[category]
            mask = df_work['category'] == category
            cat_data = df_work.loc[mask].copy()
            
            if cat_data.empty:
                continue
            
            # Get unique hours and predict in one batch
            unique_hours = cat_data['_hour'].unique()
            future_df = pd.DataFrame({'ds': unique_hours})
            
            if len(future_df) == 0:
                continue
            
            forecast = model.predict(future_df)
            forecast_cols = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_cols = forecast_cols.rename(columns={
                'ds': '_hour', 'yhat': '_expected',
                'yhat_lower': '_lower', 'yhat_upper': '_upper'
            })
            
            # Merge forecast onto all events for this category (vectorized join)
            merged = cat_data.merge(forecast_cols, on='_hour', how='left')
            
            has_forecast = merged['_expected'].notna()
            if not has_forecast.any():
                continue
            
            idx = merged.loc[has_forecast, '_orig_idx'].values
            prices = merged.loc[has_forecast, 'final_price'].values
            expected = merged.loc[has_forecast, '_expected'].values
            lower = merged.loc[has_forecast, '_lower'].values
            upper = merged.loc[has_forecast, '_upper'].values
            
            # Assign forecast bounds (vectorized)
            results.loc[idx, 'ts_expected'] = expected
            results.loc[idx, 'ts_lower'] = lower
            results.loc[idx, 'ts_upper'] = upper
            
            # Detect anomalies: price outside confidence interval (vectorized)
            outside = (prices < lower) | (prices > upper)
            anomaly_idx = idx[outside]
            results.loc[anomaly_idx, 'ts_anomaly'] = True
            
            # Score: how far outside the interval (vectorized)
            interval_width = upper - lower
            safe_width = np.where(interval_width > 0, interval_width, 1.0)
            deviation = np.where(
                prices < lower,
                (lower - prices) / safe_width,
                np.where(
                    prices > upper,
                    (prices - upper) / safe_width,
                    0.0
                )
            )
            scores = np.clip(deviation, 0.0, 1.0)
            results.loc[idx, 'ts_score'] = scores
        
        n_ts_anomalies = results['ts_anomaly'].sum()
        logger.info(f"Time series found {n_ts_anomalies} anomalies")
        
        return results
    
    def get_forecast_for_category(
        self, category: str, periods: int = 24
    ) -> Optional[pd.DataFrame]:
        """
        Generate future forecast for a category.
        Useful for the dashboard to show expected price ranges.
        """
        if not PROPHET_AVAILABLE or category not in self.models:
            return None
        
        model = self.models[category]
        future = model.make_future_dataframe(periods=periods, freq='h')
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend',
                         'daily', 'weekly']].tail(periods)
    
    def get_components(self, category: str) -> Optional[pd.DataFrame]:
        """Get decomposed trend/seasonality components for visualization."""
        if category not in self.forecasts:
            return None
        return self.forecasts[category]
