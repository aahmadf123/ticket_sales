"""Use case for forecasting game attendance."""

from typing import Dict, List, Optional, Any
from datetime import datetime, date
from dataclasses import dataclass
import pandas as pd

from src.domain.entities.game import Game
from src.domain.entities.prediction import Prediction, PredictionHorizon
from src.domain.services.attendance_forecasting_service import AttendanceForecastingService
from src.domain.services.feature_engineering_service import FeatureEngineeringService
from src.application.ports.ports import (
    GameRepositoryPort,
    PredictionRepositoryPort,
    WeatherServicePort,
)


@dataclass
class ForecastResult:
    """Result of attendance forecast."""
    
    game_id: str
    game_date: date
    opponent: str
    prediction: Prediction
    weather_conditions: Dict[str, Any]
    model_version: str
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "game_date": self.game_date.isoformat(),
            "opponent": self.opponent,
            "prediction": self.prediction.to_dict(),
            "weather_conditions": self.weather_conditions,
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat(),
        }


class ForecastAttendanceUseCase:
    """Use case for generating attendance forecasts.
    
    Orchestrates the process of:
    1. Loading game data
    2. Fetching weather forecasts
    3. Engineering features
    4. Generating predictions
    5. Storing results
    """
    
    def __init__(
        self,
        game_repository: GameRepositoryPort,
        prediction_repository: PredictionRepositoryPort,
        weather_service: WeatherServicePort,
        forecasting_service: AttendanceForecastingService,
        feature_engineering_service: FeatureEngineeringService,
    ):
        """Initialize use case.
        
        Args:
            game_repository: Repository for game data
            prediction_repository: Repository for predictions
            weather_service: Service for weather data
            forecasting_service: ML forecasting service
            feature_engineering_service: Feature engineering service
        """
        self.game_repo = game_repository
        self.prediction_repo = prediction_repository
        self.weather_service = weather_service
        self.forecasting_service = forecasting_service
        self.feature_service = feature_engineering_service
    
    def forecast_game(
        self,
        game_id: str,
        horizon: Optional[str] = None
    ) -> ForecastResult:
        """Generate attendance forecast for a specific game.
        
        Args:
            game_id: ID of the game to forecast
            horizon: Prediction horizon (T_7, T_72, T_24, GAME_DAY)
                    If None, determines automatically based on days until game
        
        Returns:
            ForecastResult with prediction details
        
        Raises:
            ValueError: If game not found or invalid horizon
        """
        # Load game
        game = self.game_repo.get_by_id(game_id)
        if game is None:
            raise ValueError(f"Game not found: {game_id}")
        
        # Determine horizon if not specified
        if horizon is None:
            horizon = self._determine_horizon(game.game_date)
        
        horizon_enum = PredictionHorizon[horizon]
        
        # Get weather forecast
        weather_data = self._get_weather_for_game(game)
        
        # Update game with weather data
        game = self._update_game_weather(game, weather_data)
        
        # Engineer features
        features = self.feature_service.engineer_game_features(game)
        features_df = pd.DataFrame([features])
        
        # Generate prediction
        predictions = self.forecasting_service.predict(
            features_df,
            horizon=horizon_enum
        )
        
        prediction = predictions[0]
        prediction.game_id = game_id
        
        # Save prediction
        self.prediction_repo.save(prediction)
        
        # Update game in repository
        self.game_repo.update(game)
        
        return ForecastResult(
            game_id=game_id,
            game_date=game.game_date,
            opponent=game.opponent,
            prediction=prediction,
            weather_conditions=weather_data,
            model_version=prediction.model_version,
            generated_at=datetime.now(),
        )
    
    def forecast_upcoming_games(
        self,
        days_ahead: int = 14
    ) -> List[ForecastResult]:
        """Generate forecasts for all upcoming games.
        
        Args:
            days_ahead: Number of days to look ahead
        
        Returns:
            List of ForecastResults
        """
        upcoming_games = self.game_repo.get_upcoming(days_ahead)
        
        results = []
        for game in upcoming_games:
            try:
                result = self.forecast_game(game.game_id)
                results.append(result)
            except Exception as e:
                print(f"Error forecasting game {game.game_id}: {e}")
                continue
        
        return results
    
    def generate_scenario_forecasts(
        self,
        game_id: str,
        scenarios: List[Dict[str, Any]]
    ) -> List[ForecastResult]:
        """Generate forecasts for different scenarios.
        
        Args:
            game_id: ID of the game
            scenarios: List of scenario dictionaries with feature overrides
        
        Returns:
            List of ForecastResults for each scenario
        """
        game = self.game_repo.get_by_id(game_id)
        if game is None:
            raise ValueError(f"Game not found: {game_id}")
        
        base_features = self.feature_service.engineer_game_features(game)
        
        results = []
        for i, scenario in enumerate(scenarios):
            # Override features with scenario values
            scenario_features = base_features.copy()
            scenario_features.update(scenario)
            
            features_df = pd.DataFrame([scenario_features])
            
            predictions = self.forecasting_service.predict(features_df)
            prediction = predictions[0]
            prediction.notes = f"Scenario {i + 1}: {scenario.get('name', 'Custom')}"
            
            result = ForecastResult(
                game_id=game_id,
                game_date=game.game_date,
                opponent=game.opponent,
                prediction=prediction,
                weather_conditions=scenario.get("weather", {}),
                model_version=prediction.model_version,
                generated_at=datetime.now(),
            )
            results.append(result)
        
        return results
    
    def train_model(
        self,
        historical_games: Optional[List[Game]] = None,
        min_seasons: int = 3
    ) -> Dict[str, Any]:
        """Train or retrain the forecasting model.
        
        Args:
            historical_games: List of historical games with actual attendance
                             If None, loads from repository
            min_seasons: Minimum number of seasons required
        
        Returns:
            Training metrics dictionary
        """
        if historical_games is None:
            historical_games = self.game_repo.get_past_games(limit=500)
        
        # Filter games with actual attendance
        games_with_actuals = [
            g for g in historical_games
            if g.actual_attendance is not None
        ]
        
        if len(games_with_actuals) < 20:
            raise ValueError(
                f"Insufficient training data: {len(games_with_actuals)} games. "
                f"Need at least 20 games with actual attendance."
            )
        
        # Check season coverage
        seasons = set(g.season for g in games_with_actuals)
        if len(seasons) < min_seasons:
            raise ValueError(
                f"Insufficient season coverage: {len(seasons)} seasons. "
                f"Need at least {min_seasons} seasons."
            )
        
        # Engineer features for all games
        features_list = []
        targets = []
        
        for game in games_with_actuals:
            features = self.feature_service.engineer_game_features(game)
            features_list.append(features)
            targets.append(game.actual_attendance)
        
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets)
        
        # Add season column for cross-validation
        features_df["_season"] = [g.season for g in games_with_actuals]
        
        # Train model
        training_metrics = self.forecasting_service.train(
            features_df.drop(columns=["_season"]),
            targets_series,
            seasons=features_df["_season"]
        )
        
        return training_metrics
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate model performance on historical predictions.
        
        Returns:
            Performance metrics dictionary
        """
        predictions_with_actuals = self.prediction_repo.get_all_with_actuals()
        
        if not predictions_with_actuals:
            return {"error": "No predictions with actuals found"}
        
        errors = []
        abs_errors = []
        pct_errors = []
        within_ci = []
        
        for record in predictions_with_actuals:
            predicted = record["predicted_attendance"]
            actual = record["actual_attendance"]
            lower = record["confidence_lower"]
            upper = record["confidence_upper"]
            
            error = predicted - actual
            abs_error = abs(error)
            pct_error = (abs_error / actual) * 100 if actual > 0 else 0
            in_ci = lower <= actual <= upper
            
            errors.append(error)
            abs_errors.append(abs_error)
            pct_errors.append(pct_error)
            within_ci.append(in_ci)
        
        n = len(errors)
        
        return {
            "n_predictions": n,
            "mae": sum(abs_errors) / n,
            "rmse": (sum(e**2 for e in errors) / n) ** 0.5,
            "mape": sum(pct_errors) / n,
            "median_ape": sorted(pct_errors)[n // 2],
            "ci_coverage": sum(within_ci) / n * 100,
            "mean_error": sum(errors) / n,
            "std_error": (sum((e - sum(errors)/n)**2 for e in errors) / n) ** 0.5,
        }
    
    def _determine_horizon(self, game_date: date) -> str:
        """Determine prediction horizon based on days until game.
        
        Args:
            game_date: Date of the game
        
        Returns:
            Horizon string (T_7, T_72, T_24, GAME_DAY)
        """
        days_until = (game_date - date.today()).days
        
        if days_until >= 7:
            return "T_7"
        elif days_until >= 3:
            return "T_72"
        elif days_until >= 1:
            return "T_24"
        else:
            return "GAME_DAY"
    
    def _get_weather_for_game(self, game: Game) -> Dict[str, Any]:
        """Get weather data for a game.
        
        Args:
            game: Game entity
        
        Returns:
            Weather data dictionary
        """
        days_until = (game.game_date - date.today()).days
        
        if days_until < 0:
            # Past game - get historical
            return self.weather_service.get_historical_weather(game.game_date)
        elif days_until <= 7:
            # Within forecast range
            return self.weather_service.get_forecast(game.game_date)
        else:
            # Too far out - use historical averages
            return self.weather_service.get_historical_weather(game.game_date)
    
    def _update_game_weather(
        self,
        game: Game,
        weather_data: Dict[str, Any]
    ) -> Game:
        """Update game with weather data.
        
        Args:
            game: Game entity
            weather_data: Weather data dictionary
        
        Returns:
            Updated game
        """
        weather = weather_data.get("weather", weather_data)
        
        game.temperature_f = weather.get("temperature_f")
        game.feels_like_f = weather.get("feels_like_f")
        game.precipitation_prob = weather.get("precipitation_prob")
        game.wind_speed_mph = weather.get("wind_speed_mph")
        game.weather_condition = weather.get("weather_condition")
        
        if game.temperature_f is not None:
            game.weather_comfort_index = game.calculate_weather_comfort_index()
        
        return game
