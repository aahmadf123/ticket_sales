"""Weather API adapter for fetching weather data."""

import requests
from typing import Dict, Optional, Any, List
from datetime import datetime, date
from dataclasses import dataclass


@dataclass
class WeatherData:
    """Weather data for a specific date/time."""
    
    date: date
    temperature_f: float
    feels_like_f: float
    humidity: float
    precipitation_prob: float
    precipitation_amount: float
    wind_speed_mph: float
    wind_gust_mph: float
    cloud_cover: float
    weather_condition: str
    weather_description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "temperature_f": self.temperature_f,
            "feels_like_f": self.feels_like_f,
            "humidity": self.humidity,
            "precipitation_prob": self.precipitation_prob,
            "precipitation_amount": self.precipitation_amount,
            "wind_speed_mph": self.wind_speed_mph,
            "wind_gust_mph": self.wind_gust_mph,
            "cloud_cover": self.cloud_cover,
            "weather_condition": self.weather_condition,
            "weather_description": self.weather_description,
        }


class WeatherAPIAdapter:
    """Adapter for OpenWeatherMap API.
    
    Fetches current, forecast, and historical weather data
    for Toledo, Ohio (Glass Bowl stadium location).
    """
    
    # Toledo, Ohio coordinates
    DEFAULT_LAT = 41.6528
    DEFAULT_LON = -83.6108
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(
        self,
        api_key: str,
        lat: float = DEFAULT_LAT,
        lon: float = DEFAULT_LON
    ):
        """Initialize weather adapter.
        
        Args:
            api_key: OpenWeatherMap API key
            lat: Latitude (default: Toledo, OH)
            lon: Longitude (default: Toledo, OH)
        """
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
    
    def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather conditions.
        
        Returns:
            WeatherData object or None if request fails
        """
        url = f"{self.BASE_URL}/weather"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "units": "imperial",
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_current_weather(data)
        
        except requests.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return None
    
    def get_forecast(self, days: int = 5) -> List[WeatherData]:
        """Get weather forecast.
        
        Args:
            days: Number of days to forecast (max 5 for free tier)
        
        Returns:
            List of WeatherData objects
        """
        url = f"{self.BASE_URL}/forecast"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.api_key,
            "units": "imperial",
            "cnt": min(days * 8, 40),  # 3-hour intervals, max 40
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_forecast(data)
        
        except requests.RequestException as e:
            print(f"Error fetching forecast: {e}")
            return []
    
    def get_weather_for_date(
        self,
        target_date: date,
        target_hour: int = 15  # 3 PM default for game time
    ) -> Optional[WeatherData]:
        """Get weather for a specific date from forecast.
        
        Args:
            target_date: Date to get weather for
            target_hour: Target hour (24-hour format)
        
        Returns:
            WeatherData closest to target date/time
        """
        forecasts = self.get_forecast(days=7)
        
        if not forecasts:
            return None
        
        # Find closest forecast to target
        target_dt = datetime.combine(target_date, datetime.min.time())
        target_dt = target_dt.replace(hour=target_hour)
        
        closest = None
        min_diff = float("inf")
        
        for forecast in forecasts:
            forecast_dt = datetime.combine(forecast.date, datetime.min.time())
            diff = abs((forecast_dt - target_dt).total_seconds())
            
            if diff < min_diff:
                min_diff = diff
                closest = forecast
        
        return closest
    
    def get_historical_weather(
        self,
        target_date: date
    ) -> Optional[WeatherData]:
        """Get historical weather for a past date.
        
        Note: Requires OpenWeatherMap One Call API subscription
        for dates older than 5 days.
        
        Args:
            target_date: Historical date
        
        Returns:
            WeatherData or None
        """
        # For historical data, would need One Call API
        # This is a placeholder that returns estimated data
        print(f"Historical weather for {target_date} - using estimates")
        
        return self._estimate_historical_weather(target_date)
    
    def _parse_current_weather(self, data: Dict) -> WeatherData:
        """Parse current weather API response.
        
        Args:
            data: API response dictionary
        
        Returns:
            WeatherData object
        """
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        weather = data.get("weather", [{}])[0]
        rain = data.get("rain", {})
        snow = data.get("snow", {})
        
        # Calculate precipitation amount
        precip = rain.get("1h", 0) + snow.get("1h", 0)
        
        return WeatherData(
            date=date.today(),
            temperature_f=main.get("temp", 60),
            feels_like_f=main.get("feels_like", 60),
            humidity=main.get("humidity", 50),
            precipitation_prob=0.0,  # Not available in current weather
            precipitation_amount=precip,
            wind_speed_mph=wind.get("speed", 0),
            wind_gust_mph=wind.get("gust", 0),
            cloud_cover=clouds.get("all", 0),
            weather_condition=weather.get("main", "Clear"),
            weather_description=weather.get("description", "clear sky"),
        )
    
    def _parse_forecast(self, data: Dict) -> List[WeatherData]:
        """Parse forecast API response.
        
        Args:
            data: API response dictionary
        
        Returns:
            List of WeatherData objects
        """
        forecasts = []
        
        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item.get("dt", 0))
            main = item.get("main", {})
            wind = item.get("wind", {})
            clouds = item.get("clouds", {})
            weather = item.get("weather", [{}])[0]
            rain = item.get("rain", {})
            snow = item.get("snow", {})
            pop = item.get("pop", 0)
            
            precip = rain.get("3h", 0) + snow.get("3h", 0)
            
            weather_data = WeatherData(
                date=dt.date(),
                temperature_f=main.get("temp", 60),
                feels_like_f=main.get("feels_like", 60),
                humidity=main.get("humidity", 50),
                precipitation_prob=pop,
                precipitation_amount=precip,
                wind_speed_mph=wind.get("speed", 0),
                wind_gust_mph=wind.get("gust", 0),
                cloud_cover=clouds.get("all", 0),
                weather_condition=weather.get("main", "Clear"),
                weather_description=weather.get("description", "clear sky"),
            )
            
            forecasts.append(weather_data)
        
        return forecasts
    
    def _estimate_historical_weather(self, target_date: date) -> WeatherData:
        """Estimate historical weather based on typical conditions.
        
        Uses Toledo climate averages for rough estimation.
        
        Args:
            target_date: Historical date
        
        Returns:
            Estimated WeatherData
        """
        month = target_date.month
        
        # Toledo monthly averages (approximate)
        monthly_temps = {
            1: 28, 2: 31, 3: 40, 4: 52, 5: 63,
            6: 72, 7: 76, 8: 74, 9: 67, 10: 55,
            11: 43, 12: 32
        }
        
        monthly_precip_prob = {
            1: 0.3, 2: 0.3, 3: 0.35, 4: 0.35, 5: 0.35,
            6: 0.3, 7: 0.3, 8: 0.3, 9: 0.25, 10: 0.3,
            11: 0.35, 12: 0.35
        }
        
        temp = monthly_temps.get(month, 55)
        precip_prob = monthly_precip_prob.get(month, 0.3)
        
        return WeatherData(
            date=target_date,
            temperature_f=temp,
            feels_like_f=temp - 3,  # Wind chill adjustment
            humidity=60,
            precipitation_prob=precip_prob,
            precipitation_amount=0,
            wind_speed_mph=10,
            wind_gust_mph=15,
            cloud_cover=50,
            weather_condition="Clear" if precip_prob < 0.3 else "Cloudy",
            weather_description="estimated from historical averages",
        )
    
    def calculate_comfort_index(self, weather: WeatherData) -> float:
        """Calculate weather comfort index for outdoor events.
        
        Args:
            weather: WeatherData object
        
        Returns:
            Comfort index (0-100)
        """
        comfort = 100.0
        
        # Temperature penalty
        temp = weather.temperature_f
        if temp < 40:
            comfort -= (40 - temp) * 1.5
        elif temp > 85:
            comfort -= (temp - 85) * 1.5
        
        # Wind penalty
        if weather.wind_speed_mph > 20:
            comfort -= (weather.wind_speed_mph - 20) * 0.5
        
        # Precipitation penalty
        comfort -= weather.precipitation_prob * 30
        comfort -= weather.precipitation_amount * 20
        
        # Humidity penalty (if extreme)
        if weather.humidity > 80:
            comfort -= (weather.humidity - 80) * 0.3
        
        return max(0.0, min(100.0, comfort))
    
    def get_game_day_conditions(
        self,
        game_date: date,
        kickoff_hour: int = 15
    ) -> Dict[str, Any]:
        """Get comprehensive weather conditions for game day.
        
        Args:
            game_date: Date of game
            kickoff_hour: Kickoff time (24-hour format)
        
        Returns:
            Dictionary with weather conditions and derived metrics
        """
        weather = self.get_weather_for_date(game_date, kickoff_hour)
        
        if weather is None:
            weather = self._estimate_historical_weather(game_date)
        
        comfort_index = self.calculate_comfort_index(weather)
        
        # Determine weather category
        if weather.precipitation_prob > 0.5:
            category = "poor"
        elif weather.temperature_f < 40 or weather.temperature_f > 85:
            category = "challenging"
        elif weather.wind_speed_mph > 25:
            category = "windy"
        elif comfort_index > 70:
            category = "excellent"
        else:
            category = "good"
        
        return {
            "weather": weather.to_dict(),
            "comfort_index": comfort_index,
            "category": category,
            "attendance_impact": self._estimate_attendance_impact(comfort_index),
        }
    
    def _estimate_attendance_impact(self, comfort_index: float) -> float:
        """Estimate attendance impact based on weather comfort.
        
        Args:
            comfort_index: Weather comfort index (0-100)
        
        Returns:
            Multiplier for expected attendance (0.7 - 1.1)
        """
        if comfort_index >= 80:
            return 1.05  # 5% boost for great weather
        elif comfort_index >= 60:
            return 1.0   # Normal
        elif comfort_index >= 40:
            return 0.90  # 10% reduction
        elif comfort_index >= 20:
            return 0.80  # 20% reduction
        else:
            return 0.70  # 30% reduction for terrible weather
