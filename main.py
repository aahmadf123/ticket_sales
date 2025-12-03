#!/usr/bin/env python3
"""
Toledo Attendance Forecasting System - Main Entry Point

This script provides the main entry point for the forecasting system.
It can be run in different modes:
- preprocess: Process raw ticket sales data
- train: Train the forecasting models
- predict: Generate attendance predictions
- optimize: Optimize pricing trajectories
- churn: Analyze season ticket holder churn
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import date
import yaml
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure.adapters.data_preprocessing_adapter import DataPreprocessingAdapter
from src.infrastructure.adapters.weather_api_adapter import WeatherAPIAdapter
from src.infrastructure.repositories.in_memory_repositories import (
    InMemoryGameRepository,
    InMemoryTicketSaleRepository,
    InMemoryPredictionRepository,
    InMemorySeasonTicketHolderRepository,
    InMemoryPricingTrajectoryRepository,
)
from src.domain.services.feature_engineering_service import FeatureEngineeringService
from src.domain.services.attendance_forecasting_service import AttendanceForecastingService
from src.domain.services.churn_modeling_service import ChurnModelingService
from src.domain.services.price_optimization_service import PriceOptimizationService
from src.application.use_cases.forecast_attendance_use_case import ForecastAttendanceUseCase
from src.application.use_cases.optimize_pricing_use_case import OptimizePricingUseCase
from src.application.use_cases.analyze_churn_use_case import AnalyzeChurnUseCase


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(config: dict, args: argparse.Namespace) -> None:
    """Preprocess raw ticket sales data."""
    print("=" * 60)
    print("PREPROCESSING TICKET SALES DATA")
    print("=" * 60)
    
    adapter = DataPreprocessingAdapter()
    
    # Load files
    files_to_process = []
    
    if args.input:
        files_to_process = args.input
    else:
        # Use default files from config
        raw_dir = config["data"]["raw_dir"]
        if os.path.exists(config["data"]["historical_sales"]):
            files_to_process.append(config["data"]["historical_sales"])
        if os.path.exists(config["data"]["current_sales"]):
            files_to_process.append(config["data"]["current_sales"])
    
    if not files_to_process:
        print("No input files found. Please specify input files with --input")
        return
    
    print(f"Processing {len(files_to_process)} file(s)...")
    
    # Process each file
    processed_dfs = []
    for file_path in files_to_process:
        print(f"\nProcessing: {file_path}")
        
        if file_path.endswith(".xlsx"):
            df = adapter.load_excel(file_path)
        else:
            df = adapter.load_csv(file_path)
        
        print(f"  Raw rows: {len(df)}")
        
        processed = adapter.preprocess(df)
        print(f"  Processed rows: {len(processed)}")
        
        processed_dfs.append(processed)
    
    # Merge datasets
    if len(processed_dfs) > 1:
        # Concatenate all processed dataframes
        merged = pd.concat(processed_dfs, ignore_index=True)
        # Apply final filter
        merged = merged[merged["event_pmt"] != 0].reset_index(drop=True)
        print(f"\nMerged dataset: {len(merged)} rows")
    else:
        merged = processed_dfs[0]
    
    # Get summary statistics
    summary = adapter.get_summary_statistics(merged)
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save processed data
    output_path = args.output or os.path.join(
        config["data"]["processed_dir"],
        "processed_ticket_sales.csv"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")


def train_models(config: dict, args: argparse.Namespace) -> None:
    """Train the forecasting models."""
    print("=" * 60)
    print("TRAINING FORECASTING MODELS")
    print("=" * 60)
    
    # Load processed data
    data_path = args.input or os.path.join(
        config["data"]["processed_dir"],
        "processed_ticket_sales.csv"
    )
    
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        print("Run preprocessing first: python main.py preprocess")
        return
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Initialize services
    feature_service = FeatureEngineeringService()
    forecasting_service = AttendanceForecastingService()
    
    # Aggregate by event for game-level features
    adapter = DataPreprocessingAdapter()
    games_df = adapter.aggregate_by_event(df)
    
    print(f"Training on {len(games_df)} games")
    
    # For demo purposes, create synthetic target if not available
    if "actual_attendance" not in games_df.columns:
        print("Note: Using tickets_sold as proxy for actual_attendance")
        games_df["actual_attendance"] = games_df["total_qty"] * 0.85  # 85% scan rate
    
    # Prepare features and target
    features_list = []
    for _, row in games_df.iterrows():
        features = {
            "tickets_sold": row.get("total_qty", 0),
            "total_revenue": row.get("total_revenue", 0),
            "avg_ticket_price": row.get("total_revenue", 0) / max(row.get("total_qty", 1), 1),
            "transaction_count": row.get("transaction_count", 0),
        }
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    targets = games_df["actual_attendance"]
    
    # Add season for CV
    if "season_code" in games_df.columns:
        seasons = games_df["season_code"].apply(lambda x: int(x[-2:]) + 2000 if pd.notna(x) else 2024)
    else:
        seasons = pd.Series([2024] * len(games_df))
    
    # Train models
    print("\nTraining ensemble models...")
    metrics = forecasting_service.train(features_df, targets, seasons)
    
    print("\nTraining Metrics:")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Training MAE: {metrics['training_mae']:.2f}")
    print(f"  Training RMSE: {metrics['training_rmse']:.2f}")
    print(f"  Training MAPE: {metrics['training_mape']:.2f}%")
    
    if "cv_mae" in metrics:
        print(f"\nCross-Validation Metrics:")
        print(f"  CV MAE: {metrics['cv_mae']:.2f}")
        print(f"  CV RMSE: {metrics['cv_rmse']:.2f}")
        print(f"  CV MAPE: {metrics['cv_mape']:.2f}%")
    
    print(f"\nOptimal Ensemble Weights:")
    for model, weight in metrics["ensemble_weights"].items():
        print(f"  {model}: {weight:.3f}")
    
    # Save model
    output_path = args.output or os.path.join(
        config["data"]["models_dir"],
        "attendance_model.joblib"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecasting_service.save_model(output_path)
    print(f"\nModel saved to: {output_path}")


def generate_predictions(config: dict, args: argparse.Namespace) -> None:
    """Generate attendance predictions."""
    print("=" * 60)
    print("GENERATING ATTENDANCE PREDICTIONS")
    print("=" * 60)
    
    # Load model
    model_path = args.model or os.path.join(
        config["data"]["models_dir"],
        "attendance_model.joblib"
    )
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run training first: python main.py train")
        return
    
    forecasting_service = AttendanceForecastingService()
    forecasting_service.load_model(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Example prediction
    print("\nGenerating sample prediction...")
    
    sample_features = pd.DataFrame([{
        "tickets_sold": 15000,
        "total_revenue": 300000,
        "avg_ticket_price": 20,
        "transaction_count": 5000,
    }])
    
    from src.domain.entities.prediction import PredictionHorizon
    predictions = forecasting_service.predict(sample_features, horizon=PredictionHorizon.T_24)
    
    pred = predictions[0]
    print(f"\nPrediction Results:")
    print(f"  Predicted Attendance: {pred.predicted_attendance:,}")
    print(f"  Confidence Interval: {pred.confidence_interval.lower:,} - {pred.confidence_interval.upper:,}")
    print(f"  Uncertainty Level: {pred.uncertainty_level.name}")


def optimize_pricing(config: dict, args: argparse.Namespace) -> None:
    """Optimize ticket pricing trajectories."""
    print("=" * 60)
    print("OPTIMIZING TICKET PRICING")
    print("=" * 60)
    
    # Initialize services
    optimization_service = PriceOptimizationService()
    
    # Example optimization
    section = args.section or "Lower Reserved"
    current_price = args.price or 40.0
    current_season = args.season or 2025
    
    print(f"\nOptimizing pricing for {section}")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Planning Horizon: {config['optimization']['planning_years']} years")
    
    from src.domain.entities.pricing_trajectory import SeatingSection, OptimizationConstraints
    
    # Create constraints from config
    constraints = OptimizationConstraints(
        min_annual_increase=config["optimization"]["min_annual_increase"],
        max_annual_increase=config["optimization"]["max_annual_increase"],
        inflation_floor=config["optimization"]["inflation_floor"],
        max_churn_rate=config["optimization"]["max_acceptable_churn"],
    )
    
    # Determine section enum
    section_lower = section.lower()
    if "club" in section_lower:
        section_enum = SeatingSection.CLUB
    elif "loge" in section_lower:
        section_enum = SeatingSection.LOGE
    elif "lower" in section_lower:
        section_enum = SeatingSection.LOWER_RESERVED
    elif "upper" in section_lower:
        section_enum = SeatingSection.UPPER_RESERVED
    elif "bleacher" in section_lower:
        section_enum = SeatingSection.BLEACHERS
    else:
        section_enum = SeatingSection.LOWER_RESERVED
    
    # Run optimization
    trajectory = optimization_service.create_pricing_trajectory(
        section=section_enum,
        current_price=current_price,
        current_season=current_season,
        planning_years=config["optimization"]["planning_years"],
        constraints=constraints,
        method=args.method or config["optimization"]["default_method"],
    )
    
    print(f"\nOptimal 5-Year Pricing Trajectory:")
    print("-" * 50)
    
    for i, year_price in enumerate(trajectory.yearly_prices):
        increase_pct = (year_price.adjusted_price / current_price - 1) * 100 if i == 0 else \
            (year_price.adjusted_price / trajectory.yearly_prices[i-1].adjusted_price - 1) * 100
        print(f"  Year {i+1} ({year_price.season}): ${year_price.adjusted_price:.2f} "
              f"(+{increase_pct:.1f}%)")
    
    print("-" * 50)
    print(f"  Total Expected Revenue: ${trajectory.total_expected_revenue:,.2f}")
    print(f"  Total Expected Attendance: {trajectory.total_expected_attendance:,.0f}")
    print(f"  Expected 5-Year Retention: {trajectory.expected_retention_rate:.1%}")


def analyze_churn(config: dict, args: argparse.Namespace) -> None:
    """Analyze season ticket holder churn."""
    print("=" * 60)
    print("ANALYZING SEASON TICKET HOLDER CHURN")
    print("=" * 60)
    
    # Initialize services
    churn_service = ChurnModelingService()
    
    print("\nChurn Analysis Configuration:")
    print(f"  High Risk Threshold: {config['churn']['high_risk_threshold']}")
    print(f"  Medium Risk Threshold: {config['churn']['medium_risk_threshold']}")
    print(f"  Price Increase Threshold: {config['churn']['price_increase_threshold']:.0%}")
    
    # Demonstrate churn estimation
    print("\nChurn Estimation by Price Increase:")
    print("-" * 40)
    
    for increase in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        churn = churn_service.estimate_churn_from_price_increase(increase)
        print(f"  {increase:.0%} increase -> {churn:.1%} expected churn")
    
    print("\nNote: Train model with actual holder data for accurate predictions")
    print("Run: python main.py churn --train --input holder_data.csv")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Toledo Attendance Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess --input data.csv
  python main.py train --input processed_data.csv
  python main.py predict --model model.joblib
  python main.py optimize --section "Lower Reserved" --price 40
  python main.py churn --train --input holders.csv
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["preprocess", "train", "predict", "optimize", "churn"],
        help="Operation mode"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--input",
        nargs="*",
        help="Input file(s)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path"
    )
    
    parser.add_argument(
        "--model",
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--section",
        help="Seating section for pricing optimization"
    )
    
    parser.add_argument(
        "--price",
        type=float,
        help="Current ticket price"
    )
    
    parser.add_argument(
        "--season",
        type=int,
        help="Current season year"
    )
    
    parser.add_argument(
        "--method",
        choices=["scipy", "pulp"],
        help="Optimization method"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {args.config}")
        print("Using default configuration...")
        config = {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "models_dir": "data/models",
                "historical_sales": "data/raw/TicketSales_PastData.csv",
                "current_sales": "data/raw/Ticket_Sales_by_Event__Henry_BB_26.csv",
            },
            "optimization": {
                "planning_years": 5,
                "min_annual_increase": 0.00,
                "max_annual_increase": 0.15,
                "inflation_floor": 0.03,
                "max_acceptable_churn": 0.10,
                "default_method": "scipy",
            },
            "churn": {
                "high_risk_threshold": 0.7,
                "medium_risk_threshold": 0.5,
                "price_increase_threshold": 0.05,
            },
        }
    
    # Execute mode
    if args.mode == "preprocess":
        preprocess_data(config, args)
    elif args.mode == "train":
        train_models(config, args)
    elif args.mode == "predict":
        generate_predictions(config, args)
    elif args.mode == "optimize":
        optimize_pricing(config, args)
    elif args.mode == "churn":
        analyze_churn(config, args)


if __name__ == "__main__":
    main()
