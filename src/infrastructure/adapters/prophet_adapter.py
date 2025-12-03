"""Facebook Prophet adapter for time series and seasonality analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


class ProphetAdapter:
    """Adapter for Facebook Prophet time series forecasting.
    
    Uses Prophet for:
    - Seasonality decomposition
    - Trend analysis
    - Holiday effects
    - Attendance forecasting with uncertainty
    """
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.forecast_df = None
        self.components_df = None
    
    def create_prophet_dataframe(
        self,
        dates: List[datetime],
        values: List[float],
        regressors: Optional[Dict[str, List[float]]] = None
    ) -> pd.DataFrame:
        """Create Prophet-compatible DataFrame.
        
        Args:
            dates: List of dates
            values: Target values (attendance)
            regressors: Optional dictionary of regressor columns
        
        Returns:
            DataFrame with 'ds' and 'y' columns plus regressors
        """
        df = pd.DataFrame({
            "ds": pd.to_datetime(dates),
            "y": values
        })
        
        if regressors:
            for name, values in regressors.items():
                df[name] = values
        
        return df
    
    def fit(
        self,
        df: pd.DataFrame,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        regressors: Optional[List[str]] = None,
        holidays_df: Optional[pd.DataFrame] = None
    ) -> "ProphetAdapter":
        """Fit Prophet model to data.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Trend flexibility
            seasonality_prior_scale: Seasonality flexibility
            holidays_prior_scale: Holiday effect flexibility
            regressors: List of regressor column names
            holidays_df: DataFrame with holiday information
        
        Returns:
            self
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is required. Install with: pip install prophet"
            )
        
        # Initialize model
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
        )
        
        # Add custom seasonalities for sports
        # Football season runs Aug-Dec, so add month-of-season effect
        self.model.add_seasonality(
            name="football_season",
            period=120,  # ~4 months
            fourier_order=3,
        )
        
        # Add holidays if provided
        if holidays_df is not None:
            self.model.holidays = holidays_df
        
        # Add regressors
        if regressors:
            for regressor in regressors:
                if regressor in df.columns:
                    self.model.add_regressor(regressor)
        
        # Fit model
        self.model.fit(df)
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        periods: int = 10,
        freq: str = "W",
        include_history: bool = True,
        future_regressors: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate predictions.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D', 'W', 'M')
            include_history: Include historical fitted values
            future_regressors: DataFrame with future regressor values
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        # Add future regressor values if provided
        if future_regressors is not None:
            for col in future_regressors.columns:
                if col != "ds":
                    future[col] = future_regressors[col].values[:len(future)]
        
        # Generate forecast
        self.forecast_df = self.model.predict(future)
        
        return self.forecast_df
    
    def get_forecast_at_dates(
        self,
        dates: List[datetime],
        regressors: Optional[Dict[str, List[float]]] = None
    ) -> pd.DataFrame:
        """Get predictions for specific dates.
        
        Args:
            dates: List of dates to predict
            regressors: Regressor values for each date
        
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = pd.DataFrame({"ds": pd.to_datetime(dates)})
        
        if regressors:
            for name, values in regressors.items():
                future[name] = values
        
        forecast = self.model.predict(future)
        
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    
    def decompose_seasonality(self) -> Dict[str, pd.DataFrame]:
        """Decompose time series into trend and seasonal components.
        
        Returns:
            Dictionary with component DataFrames
        """
        if not self.is_fitted or self.forecast_df is None:
            raise ValueError("Must fit model and generate forecast first")
        
        components = {}
        
        # Trend
        components["trend"] = self.forecast_df[["ds", "trend"]].copy()
        
        # Yearly seasonality
        if "yearly" in self.forecast_df.columns:
            components["yearly"] = self.forecast_df[["ds", "yearly"]].copy()
        
        # Football season effect
        if "football_season" in self.forecast_df.columns:
            components["football_season"] = self.forecast_df[
                ["ds", "football_season"]
            ].copy()
        
        # Holidays if present
        if "holidays" in self.forecast_df.columns:
            components["holidays"] = self.forecast_df[["ds", "holidays"]].copy()
        
        self.components_df = components
        
        return components
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trend in attendance.
        
        Returns:
            Dictionary with trend analysis
        """
        if self.forecast_df is None:
            raise ValueError("Must generate forecast first")
        
        trend = self.forecast_df["trend"].values
        
        # Calculate trend metrics
        trend_start = trend[0]
        trend_end = trend[-1]
        trend_change = trend_end - trend_start
        trend_change_pct = (trend_change / trend_start) * 100 if trend_start != 0 else 0
        
        # Identify changepoints
        if hasattr(self.model, "changepoints"):
            changepoints = self.model.changepoints.tolist()
        else:
            changepoints = []
        
        return {
            "trend_start": float(trend_start),
            "trend_end": float(trend_end),
            "trend_change": float(trend_change),
            "trend_change_pct": float(trend_change_pct),
            "trend_direction": "increasing" if trend_change > 0 else "decreasing",
            "changepoints": changepoints,
            "avg_trend_value": float(np.mean(trend)),
        }
    
    def get_seasonality_effects(self) -> Dict[str, Dict[str, float]]:
        """Get seasonality effect magnitudes.
        
        Returns:
            Dictionary with seasonality effect statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        effects = {}
        
        # Create a full year of dates to evaluate seasonality
        year_dates = pd.date_range(
            start="2024-01-01",
            end="2024-12-31",
            freq="D"
        )
        year_df = pd.DataFrame({"ds": year_dates})
        
        # Get predictions for the year
        year_forecast = self.model.predict(year_df)
        
        # Yearly seasonality
        if "yearly" in year_forecast.columns:
            yearly = year_forecast["yearly"]
            effects["yearly"] = {
                "min": float(yearly.min()),
                "max": float(yearly.max()),
                "range": float(yearly.max() - yearly.min()),
                "peak_month": year_forecast.loc[yearly.idxmax(), "ds"].month,
                "trough_month": year_forecast.loc[yearly.idxmin(), "ds"].month,
            }
        
        # Football season effect
        if "football_season" in year_forecast.columns:
            fb_season = year_forecast["football_season"]
            effects["football_season"] = {
                "min": float(fb_season.min()),
                "max": float(fb_season.max()),
                "range": float(fb_season.max() - fb_season.min()),
            }
        
        return effects
    
    def create_football_holidays(
        self,
        years: List[int]
    ) -> pd.DataFrame:
        """Create holiday DataFrame for football events.
        
        Args:
            years: List of years to include
        
        Returns:
            DataFrame with holiday definitions
        """
        holidays = []
        
        for year in years:
            # Homecoming (typically late September/early October)
            holidays.append({
                "holiday": "homecoming",
                "ds": datetime(year, 10, 1),
                "lower_window": -7,
                "upper_window": 0,
            })
            
            # Senior Day (last home game, typically November)
            holidays.append({
                "holiday": "senior_day",
                "ds": datetime(year, 11, 15),
                "lower_window": 0,
                "upper_window": 7,
            })
            
            # Rivalry week (Battle of I-75)
            holidays.append({
                "holiday": "rivalry_week",
                "ds": datetime(year, 11, 1),
                "lower_window": -3,
                "upper_window": 3,
            })
            
            # Home opener boost
            holidays.append({
                "holiday": "home_opener",
                "ds": datetime(year, 9, 1),
                "lower_window": 0,
                "upper_window": 14,
            })
        
        return pd.DataFrame(holidays)
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        initial: str = "730 days",
        period: str = "180 days",
        horizon: str = "60 days"
    ) -> pd.DataFrame:
        """Perform time series cross-validation.
        
        Args:
            df: Training DataFrame
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
        
        Returns:
            DataFrame with cross-validation results
        """
        try:
            from prophet.diagnostics import cross_validation
        except ImportError:
            raise ImportError("Prophet diagnostics required")
        
        if not self.is_fitted:
            self.fit(df)
        
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        return cv_results
    
    def calculate_performance_metrics(
        self,
        cv_results: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics from cross-validation.
        
        Args:
            cv_results: Cross-validation results DataFrame
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            from prophet.diagnostics import performance_metrics
        except ImportError:
            raise ImportError("Prophet diagnostics required")
        
        metrics = performance_metrics(cv_results)
        
        return {
            "mse": float(metrics["mse"].mean()),
            "rmse": float(metrics["rmse"].mean()),
            "mae": float(metrics["mae"].mean()),
            "mape": float(metrics["mape"].mean()) * 100,
            "coverage": float(metrics["coverage"].mean()) * 100,
        }
    
    def save_model(self, path: str):
        """Save fitted model to disk.
        
        Args:
            path: Path to save model
        """
        import pickle
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "is_fitted": self.is_fitted,
                "forecast_df": self.forecast_df,
            }, f)
    
    def load_model(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        import pickle
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
        self.forecast_df = data.get("forecast_df")
    
    def plot_components(self, figsize: Tuple[int, int] = (12, 10)):
        """Plot forecast components.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.forecast_df is None:
            raise ValueError("Must fit model and generate forecast first")
        
        return self.model.plot_components(self.forecast_df, figsize=figsize)
    
    def plot_forecast(self, figsize: Tuple[int, int] = (12, 6)):
        """Plot forecast with uncertainty intervals.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.forecast_df is None:
            raise ValueError("Must fit model and generate forecast first")
        
        return self.model.plot(self.forecast_df, figsize=figsize)
