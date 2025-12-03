#!/usr/bin/env python3
"""
Toledo Attendance Forecasting System - Model Training Script

This script trains all models in the ensemble:
- XGBoost
- Random Forest
- Bayesian Ridge
- Prophet (optional)

It also optimizes ensemble weights and churn models.

Usage:
    python scripts/train_models.py --config config/config.yaml
    python scripts/train_models.py --data-dir /path/to/data
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domain.services.attendance_forecasting_service import AttendanceForecastingService, ModelConfig
from domain.services.churn_modeling_service import ChurnModelingService, ChurnModelConfig
from domain.services.price_optimization_service import PriceOptimizationService, OptimizationConfig
from domain.services.feature_engineering_service import FeatureEngineeringService, FeatureConfig
from infrastructure.adapters.data_preprocessing_adapter import DataPreprocessingAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_data(config: dict, data_dir: str) -> tuple:
    """
    Preprocess ticket sales data.
    
    Returns:
        Tuple of (game_features_df, ticket_sales_df, season_ticket_holders_df)
    """
    logger.info("Starting data preprocessing...")
    
    adapter = DataPreprocessingAdapter()
    
    historical_path = os.path.join(data_dir, config['data']['historical_data_file'])
    current_path = os.path.join(data_dir, config['data']['current_season_file'])
    
    if os.path.exists(historical_path) and os.path.exists(current_path):
        logger.info(f"Processing files: {historical_path}, {current_path}")
        df = adapter.process_from_files(current_path, historical_path)
    elif os.path.exists(historical_path):
        logger.info(f"Processing single file: {historical_path}")
        df = adapter.load_csv(historical_path)
        df = adapter.preprocess(df)
    else:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    stats = adapter.get_summary_statistics(df)
    logger.info(f"Preprocessing complete. Summary: {json.dumps(stats, indent=2)}")
    
    feature_service = FeatureEngineeringService()
    
    game_features = adapter.aggregate_by_event(df)
    
    for idx, row in game_features.iterrows():
        features = feature_service.engineer_game_features(row.to_dict())
        for key, value in features.items():
            if key not in game_features.columns:
                game_features[key] = None
            game_features.at[idx, key] = value
    
    return game_features, df


def train_attendance_model(game_features, config: dict, output_dir: str) -> dict:
    """
    Train the attendance forecasting ensemble model.
    
    Returns:
        Training metrics dictionary
    """
    logger.info("Training attendance forecasting model...")
    
    model_config = ModelConfig(
        xgboost_params=config['models']['xgboost'],
        random_forest_params=config['models']['random_forest'],
        bayesian_ridge_params=config['models']['bayesian_ridge'],
        confidence_level=config['models']['confidence_level']
    )
    
    service = AttendanceForecastingService(model_config)
    
    feature_names = [
        'tickets_sold', 'weather_comfort_index', 'toledo_win_percentage',
        'opponent_strength_score', 'game_importance_factor', 'is_weekend',
        'home_game_number', 'previous_season_avg_attendance'
    ]
    
    available_features = [f for f in feature_names if f in game_features.columns]
    
    X = game_features[available_features].values
    y = game_features['actual_attendance'].values if 'actual_attendance' in game_features.columns else None
    
    if y is None:
        logger.warning("No actual_attendance column found. Using tickets_sold as proxy.")
        y = game_features['order_qty'].values if 'order_qty' in game_features.columns else game_features['tickets_sold'].values
    
    seasons = game_features['season_code'].values if 'season_code' in game_features.columns else None
    
    metrics = service.train(X, y, seasons)
    
    model_path = os.path.join(output_dir, "attendance_model.joblib")
    service.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info(f"Training metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    return metrics


def train_churn_model(ticket_data, config: dict, output_dir: str) -> dict:
    """
    Train the churn prediction model.
    
    Returns:
        Training metrics dictionary
    """
    logger.info("Training churn prediction model...")
    
    model_config = ChurnModelConfig(
        n_estimators=config['churn_model']['n_estimators'],
        max_depth=config['churn_model']['max_depth'],
        min_samples_split=config['churn_model']['min_samples_split'],
        class_weight=config['churn_model']['class_weight'],
        churn_probability_threshold=config['churn_model']['churn_probability_threshold'],
        high_risk_threshold=config['churn_model']['high_risk_threshold'],
        cv_folds=config['churn_model']['cv_folds']
    )
    
    service = ChurnModelingService(model_config)
    
    season_ticket_data = ticket_data[
        ticket_data['is_season_ticket'] == True
    ] if 'is_season_ticket' in ticket_data.columns else ticket_data
    
    if len(season_ticket_data) < 50:
        logger.warning("Insufficient season ticket data for churn modeling. Skipping.")
        return {"status": "skipped", "reason": "insufficient_data"}
    
    holder_features = []
    holder_labels = []
    
    logger.info(f"Churn model training deferred - requires season ticket holder data structure")
    
    return {"status": "deferred", "reason": "requires_holder_data"}


def run_training(config_path: str, data_dir: str, output_dir: str) -> dict:
    """
    Run full model training pipeline.
    
    Returns:
        Dictionary with all training results
    """
    logger.info("=" * 60)
    logger.info("TOLEDO ATTENDANCE FORECASTING - MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    config = load_config(config_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "training_timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "models": {}
    }
    
    try:
        game_features, ticket_data = preprocess_data(config, data_dir)
        
        processed_path = os.path.join(output_dir, "processed_data.csv")
        ticket_data.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        attendance_metrics = train_attendance_model(game_features, config, output_dir)
        results["models"]["attendance"] = attendance_metrics
        
        churn_metrics = train_churn_model(ticket_data, config, output_dir)
        results["models"]["churn"] = churn_metrics
        
        results["status"] = "success"
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)
    
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Training results saved to {results_path}")
    
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE - Status: {results['status']}")
    logger.info("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Toledo Attendance Forecasting Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing input data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory for model outputs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = run_training(args.config, args.data_dir, args.output_dir)
    
    sys.exit(0 if results["status"] == "success" else 1)


if __name__ == "__main__":
    main()
