#!/usr/bin/env python3
"""
Toledo Ticket Sales Forecasting System - Single Command Pipeline

This script runs the COMPLETE pipeline:
1. Preprocess raw data
2. Create training dataset
3. Train forecasting models (with hyperparameter tuning)
4. Run churn analysis (mandatory)
5. Optimize pricing (5-year plan with 3% inflation + predicted)
6. Generate predictions and recommendations

Usage:
    python run.py                    # Run full pipeline with defaults
    python run.py --skip-training    # Skip training, use existing model
    python run.py --config custom.yaml  # Use custom config file

Output:
    - Processed data in data/processed/
    - Trained model in data/models/
    - 5-year pricing recommendations (printed and saved)
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
load_dotenv()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_preprocessing(config: dict, force: bool = False) -> pd.DataFrame:
    """Step 1: Preprocess raw ticket sales data."""
    from src.infrastructure.adapters.data_preprocessing_adapter import DataPreprocessingAdapter

    print_header("STEP 1: DATA PREPROCESSING")

    processed_path = os.path.join(
        config["data"]["processed_dir"],
        "processed_ticket_sales.csv"
    )

    # Check if processed data already exists
    if os.path.exists(processed_path) and not force:
        print(f"  Found existing processed data: {processed_path}")
        print("  Loading existing data (use --force to reprocess)")
        return pd.read_csv(processed_path)

    adapter = DataPreprocessingAdapter()

    # Find raw data files
    files_to_process = []
    if os.path.exists(config["data"]["historical_sales"]):
        files_to_process.append(config["data"]["historical_sales"])
    if os.path.exists(config["data"]["current_sales"]):
        files_to_process.append(config["data"]["current_sales"])

    if not files_to_process:
        print("  ERROR: No raw data files found!")
        print(f"  Expected: {config['data']['historical_sales']}")
        print(f"       or: {config['data']['current_sales']}")
        sys.exit(1)

    print(f"  Processing {len(files_to_process)} file(s)...")

    # Process each file
    processed_dfs = []
    for file_path in files_to_process:
        print(f"    - {file_path}")

        if file_path.endswith(".xlsx"):
            df = adapter.load_excel(file_path)
        else:
            df = adapter.load_csv(file_path)

        processed = adapter.preprocess(df)
        print(f"      Raw: {len(df)} rows -> Processed: {len(processed)} rows")
        processed_dfs.append(processed)

    # Merge datasets
    if len(processed_dfs) > 1:
        merged = pd.concat(processed_dfs, ignore_index=True)
        merged = merged[merged["event_pmt"] != 0].reset_index(drop=True)
    else:
        merged = processed_dfs[0]

    print(f"\n  Total processed records: {len(merged):,}")

    # Save processed data
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    merged.to_csv(processed_path, index=False)
    print(f"  Saved to: {processed_path}")

    return merged


def run_training(config: dict, processed_df: pd.DataFrame, force: bool = False) -> dict:
    """Step 2 & 3: Create training dataset and train models."""
    from src.infrastructure.adapters.data_preprocessing_adapter import DataPreprocessingAdapter
    from src.domain.services.attendance_forecasting_service import AttendanceForecastingService

    print_header("STEP 2: TRAINING FORECASTING MODELS")

    model_path = os.path.join(
        config["data"]["models_dir"],
        "attendance_model.joblib"
    )

    # Check if model already exists
    if os.path.exists(model_path) and not force:
        print(f"  Found existing model: {model_path}")
        print("  Loading existing model (use --force to retrain)")

        forecasting_service = AttendanceForecastingService()
        forecasting_service.load_model(model_path)
        return forecasting_service.training_metrics

    adapter = DataPreprocessingAdapter()

    # Save processed data temporarily for training dataset creation
    temp_path = os.path.join(config["data"]["processed_dir"], "_temp_processed.csv")
    processed_df.to_csv(temp_path, index=False)

    # Create training dataset
    print("  Creating training dataset...")
    try:
        training_df = adapter.create_training_dataset(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if training_df is None or len(training_df) == 0:
        print("  ERROR: No training data available!")
        sys.exit(1)

    print(f"  Training samples: {len(training_df)}")

    # Add weather features if API key available
    weather_api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    if weather_api_key and weather_api_key != 'your_api_key_here':
        print("  Adding weather features...")
        try:
            training_df = adapter.add_weather_features(training_df, api_key=weather_api_key)
            print("    Weather features added successfully")
        except Exception as e:
            print(f"    Warning: Could not add weather features: {e}")

    # Prepare features
    target_col = adapter.get_target_column('revenue')
    feature_cols = adapter.get_feature_columns(include_weather=False)
    available_features = [c for c in feature_cols if c in training_df.columns]

    print(f"  Features: {len(available_features)}")
    print(f"  Target: {target_col}")

    X = training_df[available_features].fillna(0).values
    y = training_df[target_col].values
    feature_names = available_features

    # Get seasons for cross-validation
    if 'season_code' in training_df.columns:
        unique_seasons = training_df['season_code'].unique()
        season_to_int = {s: i for i, s in enumerate(unique_seasons)}
        seasons_array = training_df['season_code'].map(season_to_int).values
        print(f"  Seasons for CV: {list(unique_seasons)}")
    else:
        seasons_array = np.array([0] * len(training_df))

    # Train models
    print("\n  Training ensemble models (this may take a few minutes)...")
    forecasting_service = AttendanceForecastingService()
    metrics = forecasting_service.train(X, y, feature_names, seasons_array)

    print(f"\n  Training Results:")
    print(f"    MAE: {metrics['training_mae']:,.0f}")
    print(f"    RMSE: {metrics['training_rmse']:,.0f}")
    print(f"    MAPE: {metrics['training_mape']:.2f}%")

    if "cv_mae" in metrics:
        print(f"\n  Cross-Validation Results:")
        print(f"    CV MAE: {metrics['cv_mae']:,.0f}")
        print(f"    CV MAPE: {metrics['cv_mape']:.2f}%")

    print(f"\n  Ensemble Weights:")
    for model, weight in metrics["ensemble_weights"].items():
        print(f"    {model}: {weight:.3f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    forecasting_service.save_model(model_path)
    print(f"\n  Model saved to: {model_path}")

    return metrics


def run_churn_analysis(config: dict, processed_df: pd.DataFrame):
    """Step 4: Run churn analysis (mandatory)."""
    from src.domain.services.churn_modeling_service import ChurnModelingService

    print_header("STEP 3: CHURN ANALYSIS")

    churn_service = ChurnModelingService()

    print("  Analyzing churn impact of price increases...")
    print("\n  Churn Estimation by Price Increase:")
    print("  " + "-" * 50)
    print(f"  {'Increase':>12} {'Expected Churn':>18} {'Risk Level':>15}")
    print("  " + "-" * 50)

    baseline_churn = config["optimization"]["baseline_churn"]

    for increase in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
        churn = churn_service.estimate_churn_from_price_increase(baseline_churn, increase)
        if churn <= 0.07:
            risk = "Low"
        elif churn <= 0.12:
            risk = "Medium"
        else:
            risk = "High"
        print(f"  {increase:>11.0%} {churn:>17.1%} {risk:>15}")

    print("  " + "-" * 50)
    print(f"\n  Note: Baseline churn rate = {baseline_churn:.0%}")
    print(f"  Churn accelerates above {config['optimization']['churn_threshold']:.0%} price increase")

    return churn_service


def run_optimization(config: dict, processed_df: pd.DataFrame, churn_service):
    """Step 5: Run price optimization with 5-year plan."""
    from src.domain.services.price_optimization_service import (
        PriceOptimizationService, OptimizationConfig
    )

    print_header("STEP 4: 5-YEAR PRICE OPTIMIZATION")

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

    inflation = opt_config.inflation_rate
    years = opt_config.planning_years

    print(f"\n  Pricing Formula: Year N = (Year N-1 * {1+inflation:.2f}) + predicted_additional")
    print(f"  - 3% inflation is ALWAYS added (baseline)")
    print(f"  - Model predicts ADDITIONAL increase above inflation")
    print(f"  - Max total increase: {opt_config.max_annual_increase:.0%}/year")
    print(f"  - Planning horizon: {years} years")

    # Initialize optimization service
    optimization_service = PriceOptimizationService(config=opt_config)
    optimization_service.set_churn_model(churn_service)

    # Run optimization
    print("\n  Optimizing pricing for each seating level...")
    results = optimization_service.optimize_all_tickets(processed_df)

    # Display results
    if results.get('seating_recommendations'):
        print("\n" + "=" * 100)
        print(" 5-YEAR PRICING RECOMMENDATIONS BY SEATING LEVEL")
        print("=" * 100)

        print(f"\n{'Level':<25} {'Tier':<12} {'Current':>10} {'Year 1':>10} {'Year 2':>10} {'Year 3':>10} {'Year 4':>10} {'Year 5':>10}")
        print("-" * 100)

        for level, data in sorted(results['seating_recommendations'].items(),
                                   key=lambda x: -x[1]['current_revenue']):
            prices = data['optimal_prices']
            tier = data.get('elasticity_tier', 'standard')

            row = f"{level[:24]:<25} {tier:<12}"
            row += f"${prices[0]:>8.2f} "
            for i in range(1, min(6, len(prices))):
                row += f"${prices[i]:>8.2f} "
            print(row)

        # Show breakdown of increases
        print("\n" + "-" * 100)
        print(f"{'Level':<25} {'Elasticity':>10} {'Inflation':>10} {'+ Additional':>12} {'= Total Y1':>12} {'Real Gain':>12}")
        print("-" * 100)

        for level, data in sorted(results['seating_recommendations'].items(),
                                   key=lambda x: -x[1]['current_revenue']):
            elasticity = data.get('elasticity', -0.9)
            additional = data.get('additional_increases', [0])[0] if data.get('additional_increases') else 0
            total = data['yearly_increases'][0] if data.get('yearly_increases') else inflation
            real_gain = data.get('real_gain_pct', 0)

            print(f"{level[:24]:<25} {elasticity:>10.2f} {inflation:>9.0%} {additional:>+11.1%} {total:>11.1%} {real_gain:>+10.1f}%")

    # Summary
    print("\n" + "=" * 100)
    print(" SUMMARY")
    print("=" * 100)

    current = results.get('current_data_summary', {})
    print(f"\n  Current State:")
    print(f"    Total Revenue: ${current.get('total_revenue', 0):,.2f}")
    print(f"    Total Tickets: {current.get('total_tickets', 0):,}")
    print(f"    Average Price: ${current.get('avg_price', 0):.2f}")

    if results.get('seating_recommendations'):
        total_5yr = sum(d['total_5yr_revenue'] for d in results['seating_recommendations'].values())
        avg_retention = np.mean([d['final_retention'] for d in results['seating_recommendations'].values()])

        print(f"\n  5-Year Projection:")
        print(f"    Projected 5-Year Revenue: ${total_5yr:,.2f}")
        print(f"    Average Customer Retention: {avg_retention:.1%}")

    print(f"\n  Key Insight:")
    print(f"    3% inflation = NO REAL GAIN (just maintaining purchasing power)")
    print(f"    The recommended ADDITIONAL increases above 3% = REAL growth")

    return results


def save_results(config: dict, results: dict):
    """Save optimization results to file."""
    output_path = os.path.join(
        config["data"]["processed_dir"],
        f"pricing_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    if results.get('seating_recommendations'):
        rows = []
        for level, data in results['seating_recommendations'].items():
            row = {
                'seating_level': level,
                'elasticity_tier': data.get('elasticity_tier', ''),
                'elasticity': data.get('elasticity', 0),
                'current_price': data['current_price'],
                'current_revenue': data['current_revenue'],
            }
            for i, price in enumerate(data['optimal_prices'][1:], 1):
                row[f'year_{i}_price'] = price
            for i, inc in enumerate(data['yearly_increases'], 1):
                row[f'year_{i}_increase'] = inc
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n  Recommendations saved to: {output_path}")


def main():
    """Main entry point - runs the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Toledo Ticket Sales Forecasting System - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                     # Run full pipeline
    python run.py --skip-training     # Use existing model
    python run.py --force             # Reprocess all data and retrain
    python run.py --config custom.yaml  # Use custom config
        """
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing model"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing and retraining"
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 80)
    print(" TOLEDO TICKET SALES FORECASTING SYSTEM")
    print(" Complete Pipeline Execution")
    print("=" * 80)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        print(f"\n  Configuration: {args.config}")
    except FileNotFoundError:
        print(f"\nERROR: Config file not found: {args.config}")
        sys.exit(1)

    # Check for API key
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        print("\n  Note: OPENWEATHERMAP_API_KEY not set in .env file")
        print("        Weather features will not be included in training")
        print("        Get your free API key at: https://openweathermap.org/api")

    # Step 1: Preprocess data
    processed_df = run_preprocessing(config, force=args.force)

    # Step 2 & 3: Train models (unless skipped)
    if not args.skip_training:
        run_training(config, processed_df, force=args.force)
    else:
        print_header("STEP 2: TRAINING (SKIPPED)")
        print("  Using existing model from previous run")

    # Step 4: Churn analysis (mandatory)
    churn_service = run_churn_analysis(config, processed_df)

    # Step 5: Price optimization
    results = run_optimization(config, processed_df, churn_service)

    # Save results
    save_results(config, results)

    # Done
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE")
    print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
