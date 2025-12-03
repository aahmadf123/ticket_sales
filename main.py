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
import re
from pathlib import Path
from datetime import date
import yaml
import pandas as pd
import numpy as np

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
    """Train the forecasting models for revenue/demand prediction.
    
    Uses ticket sales data to predict revenue and ticket demand patterns.
    """
    print("=" * 60)
    print("TRAINING FORECASTING MODELS")
    print("=" * 60)
    
    # Initialize adapter
    adapter = DataPreprocessingAdapter()
    
    # Path to processed ticket sales
    ticket_sales_path = args.input[0] if args.input else os.path.join(
        config["data"]["processed_dir"],
        "processed_ticket_sales.csv"
    )
    
    # Check file exists
    if not os.path.exists(ticket_sales_path):
        print(f"Error: Processed data not found at {ticket_sales_path}")
        print("Run preprocessing first: python main.py preprocess")
        return
    
    print(f"Loading ticket sales from: {ticket_sales_path}")
    
    # Create training dataset
    try:
        training_df = adapter.create_training_dataset(ticket_sales_path)
    except Exception as e:
        print(f"Error creating training dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if training_df is None or len(training_df) == 0:
        print("\n❌ Error: No training data available!")
        return
    
    # Target selection: revenue or tickets
    target_type = 'revenue'  # Can also use 'tickets'
    target_col = adapter.get_target_column(target_type)
    
    print(f"\n✓ Created training dataset with {len(training_df)} events")
    print(f"  Target: {target_col}")
    print(f"  Revenue range: ${training_df['total_revenue'].min():,.2f} - ${training_df['total_revenue'].max():,.2f}")
    print(f"  Tickets range: {training_df['total_tickets'].min():,} - {training_df['total_tickets'].max():,}")
    print(f"  Mean revenue: ${training_df['total_revenue'].mean():,.2f}")
    
    # Try to add weather features
    include_weather = False
    try:
        weather_api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        print(f"\nAdding weather features...")
        training_df = adapter.add_weather_features(training_df, api_key=weather_api_key)
        include_weather = True
    except Exception as e:
        print(f"  Warning: Could not add weather features: {e}")
    
    # Get feature columns (including weather if available)
    feature_cols = adapter.get_feature_columns(include_weather=include_weather)
    available_features = [c for c in feature_cols if c in training_df.columns]
    
    print(f"\nUsing {len(available_features)} predictive features:")
    for f in available_features:
        print(f"  - {f}")
    
    # Prepare features and target
    X = training_df[available_features].fillna(0).values
    y = training_df[target_col].values
    feature_names = available_features
    
    # Get season for cross-validation
    # Use season_code directly as grouping variable (e.g., "FB24", "BB25")
    # This ensures proper Leave-One-Season-Out CV
    if 'season_code' in training_df.columns:
        # Use label encoding for seasons
        unique_seasons = training_df['season_code'].unique()
        season_to_int = {s: i for i, s in enumerate(unique_seasons)}
        seasons_array = training_df['season_code'].map(season_to_int).values
        print(f"\n  Found {len(unique_seasons)} unique seasons for CV: {list(unique_seasons)}")
    else:
        seasons_array = np.array([0] * len(training_df))
        print("\n  Warning: No season_code found - CV will use single group")
    
    # Initialize forecasting service
    forecasting_service = AttendanceForecastingService()
    
    # Train models
    print("\nTraining ensemble models...")
    metrics = forecasting_service.train(X, y, feature_names, seasons_array)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Training MAE: {metrics['training_mae']:,.0f} attendees")
    print(f"  Training RMSE: {metrics['training_rmse']:,.0f}")
    print(f"  Training MAPE: {metrics['training_mape']:.2f}%")
    
    if "cv_mae" in metrics:
        print(f"\nCross-Validation Metrics (Leave-One-Season-Out):")
        print(f"  CV MAE: {metrics['cv_mae']:,.0f} attendees")
        print(f"  CV RMSE: {metrics['cv_rmse']:,.0f}")
        print(f"  CV MAPE: {metrics['cv_mape']:.2f}%")
    
    print(f"\nOptimal Ensemble Weights:")
    for model, weight in metrics["ensemble_weights"].items():
        print(f"  {model}: {weight:.3f}")
    
    if "feature_importance" in metrics:
        print(f"\nFeature Importance:")
        importance = metrics["feature_importance"]
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"  {feat}: {imp:.3f}")
    
    # Save model
    output_path = args.output or os.path.join(
        config["data"]["models_dir"],
        "attendance_model.joblib"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecasting_service.save_model(output_path)
    print(f"\n✓ Model saved to: {output_path}")
    
    # Save training data for reference
    training_data_path = os.path.join(
        config["data"]["processed_dir"],
        "training_data_with_attendance.csv"
    )
    if training_df is not None:
        training_df.to_csv(training_data_path, index=False)
        print(f"✓ Training data saved to: {training_data_path}")


def generate_predictions(config: dict, args: argparse.Namespace) -> None:
    """Generate revenue/demand predictions."""
    print("=" * 60)
    print("GENERATING REVENUE/DEMAND PREDICTIONS")
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
    
    # Get feature names from the trained model
    feature_names = forecasting_service.feature_names
    print(f"\nModel expects {len(feature_names)} features:")
    for f in feature_names:
        print(f"  - {f}")
    
    # Example prediction with correct features
    print("\nGenerating sample prediction for a hypothetical event...")
    
    # Build sample features based on what the model was trained with
    sample_features_dict = {}
    for feat in feature_names:
        if feat == 'season_ticket_pct':
            sample_features_dict[feat] = 0.01
        elif feat == 'new_ticket_pct':
            sample_features_dict[feat] = 0.004
        elif feat == 'avg_seat_value_index':
            sample_features_dict[feat] = 1.07
        elif feat == 'revenue_per_ticket':
            sample_features_dict[feat] = 20.0
        elif feat == 'avg_unit_price':
            sample_features_dict[feat] = 21.0
        elif feat == 'game_number':
            sample_features_dict[feat] = 3
        elif feat == 'prior_avg_revenue':
            sample_features_dict[feat] = 5000.0
        elif feat == 'prior_avg_tickets':
            sample_features_dict[feat] = 250.0
        else:
            sample_features_dict[feat] = 0.0
    
    sample_features = pd.DataFrame([sample_features_dict])
    
    # Ensure columns are in the right order
    sample_features = sample_features[feature_names]
    
    from src.domain.entities.prediction import PredictionHorizon
    predictions = forecasting_service.predict(sample_features.values, horizon=PredictionHorizon.T_24)
    
    pred = predictions[0]
    print(f"\nPrediction Results:")
    print(f"  Predicted Value: ${pred.predicted_attendance:,.2f}")
    print(f"  Confidence Interval: ${pred.confidence_interval.lower:,.2f} - ${pred.confidence_interval.upper:,.2f}")
    print(f"  Uncertainty Level: {pred.uncertainty_level.name}")
    
    # Show predictions for different scenarios
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON")
    print("=" * 60)
    
    scenarios = [
        {"name": "Low Price Point", "avg_unit_price": 15.0, "revenue_per_ticket": 14.0},
        {"name": "Medium Price Point", "avg_unit_price": 25.0, "revenue_per_ticket": 23.0},
        {"name": "High Price Point", "avg_unit_price": 40.0, "revenue_per_ticket": 38.0},
    ]
    
    print(f"\n{'Scenario':<25} {'Predicted':>15} {'Confidence Interval':>30}")
    print("-" * 75)
    
    for scenario in scenarios:
        features = sample_features_dict.copy()
        features.update({k: v for k, v in scenario.items() if k in feature_names})
        scenario_df = pd.DataFrame([features])[feature_names]
        preds = forecasting_service.predict(scenario_df.values, horizon=PredictionHorizon.T_24)
        p = preds[0]
        print(f"{scenario['name']:<25} ${p.predicted_attendance:>14,.2f} ${p.confidence_interval.lower:>12,.2f} - ${p.confidence_interval.upper:<12,.2f}")


def optimize_pricing(config: dict, args: argparse.Namespace) -> None:
    """Optimize ticket pricing for EACH seating level.
    
    Outputs specific price recommendations for each seat type for the next 5 years.
    All increases are ABOVE 3% inflation to provide REAL revenue gains.
    """
    print("=" * 60)
    print("TICKET PRICING OPTIMIZATION - 5 YEAR PLAN")
    print("=" * 60)
    
    from src.domain.services.price_optimization_service import OptimizationConfig
    
    # Load processed ticket data
    ticket_sales_path = os.path.join(
        config["data"]["processed_dir"],
        "processed_ticket_sales.csv"
    )
    
    if not os.path.exists(ticket_sales_path):
        print(f"Error: Processed data not found at {ticket_sales_path}")
        print("Run preprocessing first: python main.py preprocess")
        return
    
    print(f"\nLoading ticket data from: {ticket_sales_path}")
    ticket_df = pd.read_csv(ticket_sales_path)
    print(f"  Loaded {len(ticket_df):,} ticket records")
    
    # Create optimization config
    opt_config = OptimizationConfig(
        planning_years=config["optimization"]["planning_years"],
        min_annual_increase=config["optimization"]["min_annual_increase"],
        max_annual_increase=config["optimization"]["max_annual_increase"],
        inflation_rate=config["optimization"]["inflation_floor"],
        max_acceptable_churn=config["optimization"]["max_acceptable_churn"],
        churn_elasticity=config["optimization"].get("churn_elasticity", 2.0),
        baseline_churn=config["optimization"].get("baseline_churn", 0.05),
    )
    
    # Initialize optimization service
    optimization_service = PriceOptimizationService(config=opt_config)
    
    # Try to load trained churn model if available
    churn_model_path = os.path.join(config["data"]["models_dir"], "churn_model.joblib")
    if os.path.exists(churn_model_path):
        print(f"\n  Loading trained churn model from: {churn_model_path}")
        churn_service = ChurnModelingService()
        churn_service.load_model(churn_model_path)
        optimization_service.set_churn_model(churn_service)
    else:
        print(f"\n  No trained churn model found - using default churn formula")
        print(f"    (Train churn model with: python main.py churn --train --input holder_data.csv)")
    
    inflation = opt_config.inflation_rate
    years = opt_config.planning_years
    
    print(f"\n  Key Parameters:")
    print(f"    Inflation: {inflation:.0%} (increases at this rate = NO real gain)")
    print(f"    Max increase: {opt_config.max_annual_increase:.0%}/year (churn constraint)")
    print(f"    Churn threshold: 5% (increases above this cause customer loss)")
    print(f"    Planning horizon: {years} years")
    
    # Run optimization
    results = optimization_service.optimize_all_tickets(ticket_df)
    
    # ========================================
    # LEARNED ELASTICITY BY SEATING LEVEL
    # ========================================
    if results.get('seating_recommendations'):
        print("\n" + "=" * 100)
        print("PRICE ELASTICITY BY SEATING LEVEL (Learned from Data)")
        print("=" * 100)
        print("\n  Elasticity = how much demand drops when price increases")
        print("  More negative = more price sensitive (careful with increases)")
        print("  Less negative = less sensitive (can increase more aggressively)")
        print(f"\n{'Seating Level':<25} {'Elasticity':>12} {'Interpretation':<40}")
        print("-" * 80)
        
        for level, data in sorted(results['seating_recommendations'].items(),
                                   key=lambda x: -x[1]['current_revenue']):
            elasticity = data.get('elasticity', -0.5)
            if elasticity > -0.3:
                interp = "Very inelastic - can raise prices aggressively"
            elif elasticity > -0.6:
                interp = "Inelastic - moderate increases OK"
            elif elasticity > -1.0:
                interp = "Unit elastic - balanced approach needed"
            else:
                interp = "Elastic - be careful with increases"
            
            print(f"  {level[:24]:<25} {elasticity:>10.2f}   {interp:<40}")
        
        # ========================================
        # SEATING LEVEL RECOMMENDATIONS
        # ========================================
        print("\n" + "=" * 100)
        print("RECOMMENDED PRICES BY SEATING LEVEL (5-Year Plan)")
        print("=" * 100)
        print(f"\n{'Seating Level':<20} {'Current':>10} {'Year 1':>10} {'Year 2':>10} {'Year 3':>10} {'Year 4':>10} {'Year 5':>10} {'Real Gain':>12}")
        print("-" * 100)
        
        for level, data in sorted(results['seating_recommendations'].items(), 
                                   key=lambda x: -x[1]['current_revenue']):
            prices = data['optimal_prices']
            real_gain = data.get('real_gain_pct', 0)
            gain_indicator = "✓" if data.get('beats_inflation') else "✗"
            
            # Format price string
            price_str = f"{level[:19]:<20}"
            price_str += f"${prices[0]:>8.2f} "
            for i in range(1, min(6, len(prices))):
                price_str += f"${prices[i]:>8.2f} "
            price_str += f"{gain_indicator} {real_gain:>+.1f}%"
            
            print(price_str)
        
        # Show year-over-year increases (NOW DIFFERENT FOR EACH LEVEL)
        print("\n" + "-" * 100)
        print(f"{'Seating Level':<20} {'Elasticity':>10} {'Yr1 Inc':>10} {'Yr2 Inc':>10} {'Yr3 Inc':>10} {'Yr4 Inc':>10} {'Yr5 Inc':>10}")
        print("-" * 100)
        
        for level, data in sorted(results['seating_recommendations'].items(),
                                   key=lambda x: -x[1]['current_revenue']):
            elasticity = data.get('elasticity', -0.5)
            increases = data['yearly_increases']
            inc_str = f"{level[:19]:<20}{elasticity:>10.2f} "
            for inc in increases[:5]:
                inc_str += f"{inc:>+9.1%} "
            print(inc_str)
    
    # ========================================
    # PRICE TYPE RECOMMENDATIONS (Top 10)
    # ========================================
    if results.get('price_type_recommendations'):
        print("\n" + "=" * 100)
        print("RECOMMENDED PRICES BY TICKET TYPE (Top 10 by Revenue)")
        print("=" * 100)
        print(f"\n{'Price Type':<25} {'Current':>10} {'Year 1':>10} {'Year 2':>10} {'Year 3':>10} {'Year 4':>10} {'Year 5':>10}")
        print("-" * 100)
        
        # Sort by current revenue and take top 10
        sorted_types = sorted(results['price_type_recommendations'].items(),
                              key=lambda x: -x[1]['current_revenue'])[:10]
        
        for ptype, data in sorted_types:
            prices = data['optimal_prices']
            price_str = f"{ptype[:24]:<25}"
            price_str += f"${prices[0]:>8.2f} "
            for i in range(1, min(6, len(prices))):
                price_str += f"${prices[i]:>8.2f} "
            print(price_str)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    current = results.get('current_data_summary', {})
    print(f"\n  Current State:")
    print(f"    Total Revenue: ${current.get('total_revenue', 0):,.2f}")
    print(f"    Total Tickets: {current.get('total_tickets', 0):,}")
    print(f"    Average Price: ${current.get('avg_price', 0):.2f}")
    
    # Calculate projected totals
    if results.get('seating_recommendations'):
        total_5yr_revenue = sum(d['total_5yr_revenue'] for d in results['seating_recommendations'].values())
        avg_retention = np.mean([d['final_retention'] for d in results['seating_recommendations'].values()])
        
        print(f"\n  5-Year Projection (if recommendations followed):")
        print(f"    Projected 5-Year Revenue: ${total_5yr_revenue:,.2f}")
        print(f"    Average Customer Retention: {avg_retention:.1%}")
    
    print(f"\n  Key Insight:")
    print(f"    At 3% inflation, prices must increase by MORE than 3%/year for REAL gains.")
    print(f"    The recommended increases balance revenue growth against customer churn.")


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
    
    baseline_churn = 0.05  # 5% baseline churn rate
    for increase in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        churn = churn_service.estimate_churn_from_price_increase(baseline_churn, increase)
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
        "--method",
        choices=["scipy", "pulp"],
        default="scipy",
        help="Optimization method (scipy for non-linear, pulp for linear programming)"
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
