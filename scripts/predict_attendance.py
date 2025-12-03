#!/usr/bin/env python3
"""
Toledo Attendance Forecasting System - Prediction Script

This script generates attendance predictions for upcoming games.

Usage:
    python scripts/predict_attendance.py --game-date 2025-09-06 --opponent "Bowling Green"
    python scripts/predict_attendance.py --config config/config.yaml --all-upcoming
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domain.services.attendance_forecasting_service import AttendanceForecastingService, ModelConfig
from domain.services.feature_engineering_service import FeatureEngineeringService
from domain.entities.game import Game, OpponentTier, SportType
from domain.entities.prediction import Prediction, PredictionHorizon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def determine_horizon(game_date: datetime) -> PredictionHorizon:
    """Determine appropriate prediction horizon based on days until game."""
    days_until = (game_date.date() - datetime.now().date()).days
    
    if days_until <= 0:
        return PredictionHorizon.GAME_DAY
    elif days_until <= 1:
        return PredictionHorizon.T_24
    elif days_until <= 3:
        return PredictionHorizon.T_72
    else:
        return PredictionHorizon.T_7


def create_game_from_input(
    game_date: datetime,
    opponent: str,
    kickoff_time: str = None,
    tickets_sold: int = None,
    toledo_wins: int = 0,
    toledo_losses: int = 0,
    opponent_tier: str = "mac",
    is_conference: bool = True,
    is_rivalry: bool = False,
    is_homecoming: bool = False,
    home_game_number: int = 1,
    previous_season_avg: float = None,
    temperature_f: float = None,
    precipitation_prob: float = None,
    wind_speed_mph: float = None,
    sport: str = "football"
) -> Game:
    """Create a Game entity from input parameters."""
    
    tier_map = {
        "power_five": OpponentTier.POWER_FIVE,
        "mac": OpponentTier.MAC,
        "fcs": OpponentTier.FCS
    }
    
    sport_map = {
        "football": SportType.FB,
        "basketball": SportType.BB,
        "volleyball": SportType.VB
    }
    
    if kickoff_time:
        kickoff = datetime.strptime(kickoff_time, "%H:%M").time()
    else:
        kickoff = datetime.strptime("15:00", "%H:%M").time()
    
    season = game_date.year if game_date.month >= 8 else game_date.year - 1
    season_code = f"FB{str(season)[2:]}" if sport == "football" else f"BB{str(season)[2:]}"
    
    game = Game(
        game_id=None,
        season=season,
        season_code=season_code,
        game_date=game_date.date(),
        kickoff_time=kickoff,
        opponent=opponent,
        opponent_tier=tier_map.get(opponent_tier.lower(), OpponentTier.MAC),
        sport_type=sport_map.get(sport.lower(), SportType.FB),
        is_conference_game=is_conference,
        is_rivalry_game=is_rivalry,
        is_homecoming=is_homecoming,
        home_game_number=home_game_number,
        total_home_games_season=7 if sport == "football" else 15,
        tickets_sold=tickets_sold,
        toledo_wins=toledo_wins,
        toledo_losses=toledo_losses,
        temperature_f=temperature_f,
        precipitation_prob=precipitation_prob,
        wind_speed_mph=wind_speed_mph,
        previous_season_avg_attendance=previous_season_avg or 18000 if sport == "football" else 4000
    )
    
    return game


def generate_prediction(
    game: Game,
    model_path: str,
    config: dict,
    include_scenarios: bool = False
) -> dict:
    """
    Generate attendance prediction for a game.
    
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Generating prediction for {game.opponent} on {game.game_date}")
    
    feature_service = FeatureEngineeringService()
    features = feature_service.engineer_game_features(game.__dict__)
    
    model_config = ModelConfig(
        xgboost_params=config['models']['xgboost'],
        random_forest_params=config['models']['random_forest'],
        bayesian_ridge_params=config['models']['bayesian_ridge'],
        confidence_level=config['models']['confidence_level']
    )
    
    forecasting_service = AttendanceForecastingService(model_config)
    
    if os.path.exists(model_path):
        forecasting_service.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Using untrained model.")
    
    horizon = determine_horizon(datetime.combine(game.game_date, datetime.min.time()))
    
    feature_names = [
        'tickets_sold', 'weather_comfort_index', 'toledo_win_percentage',
        'opponent_strength_score', 'game_importance_factor', 'is_weekend',
        'home_game_number', 'previous_season_avg_attendance'
    ]
    
    feature_vector = []
    for name in feature_names:
        if name in features:
            feature_vector.append(features[name])
        else:
            feature_vector.append(0)
    
    import numpy as np
    X = np.array([feature_vector])
    
    prediction = forecasting_service.predict(X, horizon=horizon)[0] if forecasting_service.is_trained else None
    
    if prediction is None:
        base_attendance = game.tickets_sold or game.previous_season_avg_attendance or 15000
        
        importance = features.get('game_importance_factor', 1.0)
        weather_comfort = features.get('weather_comfort_index', 70)
        
        adjustment = 1.0
        if importance > 1.3:
            adjustment *= 1.1
        if weather_comfort < 50:
            adjustment *= 0.9
        if not features.get('is_weekend', True):
            adjustment *= 0.85
        
        predicted = int(base_attendance * adjustment * 0.85)
        
        uncertainty = predicted * 0.15
        
        result = {
            "predicted_attendance": predicted,
            "confidence_lower": int(predicted - uncertainty),
            "confidence_upper": int(predicted + uncertainty),
            "uncertainty_level": "high",
            "model_type": "heuristic",
            "note": "Using heuristic model - trained model not available"
        }
    else:
        result = {
            "predicted_attendance": prediction.predicted_attendance,
            "confidence_lower": prediction.confidence_interval.lower,
            "confidence_upper": prediction.confidence_interval.upper,
            "uncertainty_level": prediction.uncertainty_level.value,
            "model_type": "ensemble",
            "xgboost_prediction": prediction.xgboost_prediction,
            "random_forest_prediction": prediction.random_forest_prediction,
            "bayesian_ridge_prediction": prediction.bayesian_ridge_prediction,
            "ensemble_weights": prediction.ensemble_weights
        }
    
    result.update({
        "game_date": game.game_date.isoformat(),
        "opponent": game.opponent,
        "horizon": horizon.value,
        "prediction_timestamp": datetime.now().isoformat(),
        "features_used": features,
        "inputs": {
            "tickets_sold": game.tickets_sold,
            "toledo_record": f"{game.toledo_wins}-{game.toledo_losses}",
            "opponent_tier": game.opponent_tier.name,
            "is_rivalry": game.is_rivalry_game,
            "is_homecoming": game.is_homecoming,
            "weather_comfort": features.get('weather_comfort_index')
        }
    })
    
    if include_scenarios and game.tickets_sold:
        scenarios = []
        
        good_weather_features = features.copy()
        good_weather_features['weather_comfort_index'] = 85
        good_weather_pred = int(result['predicted_attendance'] * 1.05)
        scenarios.append({
            "name": "Good Weather",
            "predicted_attendance": good_weather_pred,
            "description": "Clear skies, 65-75F"
        })
        
        bad_weather_features = features.copy()
        bad_weather_features['weather_comfort_index'] = 35
        bad_weather_pred = int(result['predicted_attendance'] * 0.85)
        scenarios.append({
            "name": "Bad Weather",
            "predicted_attendance": bad_weather_pred,
            "description": "Rain/cold conditions"
        })
        
        ticket_surge_pred = int(game.tickets_sold * 1.15 * 0.85)
        scenarios.append({
            "name": "Late Ticket Surge",
            "predicted_attendance": ticket_surge_pred,
            "description": "15% increase in ticket sales"
        })
        
        result["scenarios"] = scenarios
    
    return result


def run_prediction(
    config_path: str,
    model_dir: str,
    game_date: str,
    opponent: str,
    **kwargs
) -> dict:
    """
    Run prediction for a single game.
    
    Returns:
        Prediction results dictionary
    """
    logger.info("=" * 60)
    logger.info("TOLEDO ATTENDANCE FORECASTING - PREDICTION")
    logger.info("=" * 60)
    
    config = load_config(config_path)
    
    game_datetime = datetime.strptime(game_date, "%Y-%m-%d")
    
    game = create_game_from_input(
        game_date=game_datetime,
        opponent=opponent,
        **kwargs
    )
    
    model_path = os.path.join(model_dir, "attendance_model.joblib")
    
    result = generate_prediction(
        game=game,
        model_path=model_path,
        config=config,
        include_scenarios=kwargs.get('include_scenarios', False)
    )
    
    logger.info("=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Game: {opponent} on {game_date}")
    logger.info(f"Predicted Attendance: {result['predicted_attendance']:,}")
    logger.info(f"Confidence Range: {result['confidence_lower']:,} - {result['confidence_upper']:,}")
    logger.info(f"Uncertainty Level: {result['uncertainty_level']}")
    logger.info("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate Toledo Attendance Predictions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="data/models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--game-date",
        type=str,
        required=True,
        help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        help="Opponent team name"
    )
    parser.add_argument(
        "--kickoff-time",
        type=str,
        default=None,
        help="Kickoff time (HH:MM)"
    )
    parser.add_argument(
        "--tickets-sold",
        type=int,
        default=None,
        help="Number of tickets sold"
    )
    parser.add_argument(
        "--toledo-record",
        type=str,
        default="0-0",
        help="Toledo's record (W-L)"
    )
    parser.add_argument(
        "--opponent-tier",
        type=str,
        default="mac",
        choices=["power_five", "mac", "fcs"],
        help="Opponent tier"
    )
    parser.add_argument(
        "--rivalry",
        action="store_true",
        help="Is this a rivalry game"
    )
    parser.add_argument(
        "--homecoming",
        action="store_true",
        help="Is this homecoming"
    )
    parser.add_argument(
        "--sport",
        type=str,
        default="football",
        choices=["football", "basketball", "volleyball"],
        help="Sport type"
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Include scenario analysis"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    record_parts = args.toledo_record.split("-")
    toledo_wins = int(record_parts[0]) if len(record_parts) >= 1 else 0
    toledo_losses = int(record_parts[1]) if len(record_parts) >= 2 else 0
    
    result = run_prediction(
        config_path=args.config,
        model_dir=args.model_dir,
        game_date=args.game_date,
        opponent=args.opponent,
        kickoff_time=args.kickoff_time,
        tickets_sold=args.tickets_sold,
        toledo_wins=toledo_wins,
        toledo_losses=toledo_losses,
        opponent_tier=args.opponent_tier,
        is_rivalry=args.rivalry,
        is_homecoming=args.homecoming,
        sport=args.sport,
        include_scenarios=args.scenarios
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))
    
    sys.exit(0)


if __name__ == "__main__":
    main()
