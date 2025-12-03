#!/usr/bin/env python3
"""
Toledo Attendance Forecasting System - Pricing Optimization Script

This script generates optimal 5-year pricing trajectories for different
seating sections with churn constraints.

Usage:
    python scripts/optimize_pricing.py --section "Lower Reserved" --current-price 100 --target-price 150
    python scripts/optimize_pricing.py --config config/config.yaml --all-sections
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domain.services.price_optimization_service import (
    PriceOptimizationService, 
    OptimizationConfig,
    OptimizationConstraints
)
from domain.entities.pricing_trajectory import (
    PricingTrajectory,
    PricingTier,
    SeatingSection
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def map_section_name(section_name: str) -> SeatingSection:
    """Map section name string to SeatingSection enum."""
    section_map = {
        "club": SeatingSection.CLUB,
        "loge": SeatingSection.LOGE,
        "lower reserved": SeatingSection.LOWER_RESERVED,
        "lower": SeatingSection.LOWER_RESERVED,
        "upper reserved": SeatingSection.UPPER_RESERVED,
        "upper": SeatingSection.UPPER_RESERVED,
        "bleachers": SeatingSection.BLEACHERS,
        "student": SeatingSection.STUDENT,
        "zone a": SeatingSection.LOWER_RESERVED,
        "zone b": SeatingSection.LOWER_RESERVED,
        "zone c": SeatingSection.UPPER_RESERVED
    }
    return section_map.get(section_name.lower(), SeatingSection.LOWER_RESERVED)


def determine_pricing_tier(current_price: float) -> PricingTier:
    """Determine pricing tier based on current price."""
    if current_price >= 75:
        return PricingTier.PREMIUM
    elif current_price >= 35:
        return PricingTier.STANDARD
    else:
        return PricingTier.VALUE


def optimize_section_pricing(
    section_name: str,
    current_price: float,
    target_price: float,
    current_season: int,
    config: dict,
    method: str = "scipy"
) -> dict:
    """
    Optimize pricing for a single section.
    
    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Optimizing pricing for {section_name}")
    logger.info(f"Current: ${current_price:.2f} -> Target: ${target_price:.2f}")
    
    opt_config = OptimizationConfig(
        planning_years=config['pricing']['planning_years'],
        min_annual_increase=config['pricing']['min_annual_increase'],
        max_annual_increase=config['pricing']['max_annual_increase'],
        inflation_floor=config['pricing']['inflation_floor'],
        max_acceptable_churn=config['pricing']['max_acceptable_churn'],
        churn_elasticity=config['pricing']['churn_elasticity'],
        baseline_churn=config['pricing']['baseline_churn'],
        objective_weights=config['pricing']['objective_weights'],
        discount_rate=config['pricing']['discount_rate']
    )
    
    service = PriceOptimizationService(opt_config)
    
    constraints = OptimizationConstraints(
        min_annual_increase=config['pricing']['min_annual_increase'],
        max_annual_increase=config['pricing']['max_annual_increase'],
        inflation_floor=config['pricing']['inflation_floor'],
        max_churn_rate=config['pricing']['max_acceptable_churn'],
        max_cumulative_churn=config['pricing']['max_cumulative_churn'],
        terminal_price_target=target_price
    )
    
    section = map_section_name(section_name)
    tier = determine_pricing_tier(current_price)
    
    trajectory = service.create_pricing_trajectory(
        section=section,
        tier=tier,
        current_price=current_price,
        current_season=current_season,
        constraints=constraints,
        method=method
    )
    
    result = trajectory.to_dict()
    
    result.update({
        "section_name": section_name,
        "optimization_method": method,
        "optimization_timestamp": datetime.now().isoformat(),
        "summary": {
            "current_price": current_price,
            "target_price": target_price,
            "final_price": trajectory.yearly_prices[-1].adjusted_price if trajectory.yearly_prices else target_price,
            "total_expected_revenue": trajectory.total_expected_revenue,
            "expected_retention_rate": trajectory.expected_retention_rate,
            "years": config['pricing']['planning_years']
        }
    })
    
    return result


def optimize_all_sections(
    pricing_data: pd.DataFrame,
    current_season: int,
    config: dict,
    method: str = "scipy"
) -> list:
    """
    Optimize pricing for all sections in the provided data.
    
    Returns:
        List of optimization results
    """
    results = []
    
    for _, row in pricing_data.iterrows():
        section_name = row['section']
        current_price = row['current_price']
        target_price = row.get('target_price', current_price * 1.25)
        
        try:
            result = optimize_section_pricing(
                section_name=section_name,
                current_price=current_price,
                target_price=target_price,
                current_season=current_season,
                config=config,
                method=method
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to optimize {section_name}: {e}")
            results.append({
                "section_name": section_name,
                "status": "failed",
                "error": str(e)
            })
    
    return results


def compare_scenarios(
    section_name: str,
    current_price: float,
    scenarios: list,
    current_season: int,
    config: dict
) -> dict:
    """
    Compare different pricing scenarios.
    
    Parameters:
        section_name: Name of seating section
        current_price: Current ticket price
        scenarios: List of dicts with 'name' and 'target_price'
        current_season: Current season year
        config: Configuration dictionary
    
    Returns:
        Comparison results
    """
    logger.info(f"Comparing {len(scenarios)} pricing scenarios for {section_name}")
    
    opt_config = OptimizationConfig(
        planning_years=config['pricing']['planning_years'],
        min_annual_increase=config['pricing']['min_annual_increase'],
        max_annual_increase=config['pricing']['max_annual_increase'],
        inflation_floor=config['pricing']['inflation_floor'],
        max_acceptable_churn=config['pricing']['max_acceptable_churn'],
        churn_elasticity=config['pricing']['churn_elasticity'],
        baseline_churn=config['pricing']['baseline_churn'],
        objective_weights=config['pricing']['objective_weights'],
        discount_rate=config['pricing']['discount_rate']
    )
    
    service = PriceOptimizationService(opt_config)
    
    section = map_section_name(section_name)
    tier = determine_pricing_tier(current_price)
    
    comparison_input = []
    for scenario in scenarios:
        constraints = OptimizationConstraints(
            min_annual_increase=config['pricing']['min_annual_increase'],
            max_annual_increase=config['pricing']['max_annual_increase'],
            inflation_floor=config['pricing']['inflation_floor'],
            max_churn_rate=config['pricing']['max_acceptable_churn'],
            terminal_price_target=scenario['target_price']
        )
        comparison_input.append({
            "name": scenario['name'],
            "constraints": constraints
        })
    
    comparison = service.compare_scenarios(
        section=section,
        tier=tier,
        current_price=current_price,
        current_season=current_season,
        scenarios=comparison_input
    )
    
    comparison.update({
        "section_name": section_name,
        "current_price": current_price,
        "comparison_timestamp": datetime.now().isoformat()
    })
    
    return comparison


def run_optimization(
    config_path: str,
    section_name: str,
    current_price: float,
    target_price: float,
    current_season: int = None,
    method: str = "scipy",
    compare: bool = False
) -> dict:
    """
    Run pricing optimization.
    
    Returns:
        Optimization results dictionary
    """
    logger.info("=" * 60)
    logger.info("TOLEDO ATTENDANCE FORECASTING - PRICING OPTIMIZATION")
    logger.info("=" * 60)
    
    config = load_config(config_path)
    
    if current_season is None:
        now = datetime.now()
        current_season = now.year if now.month >= 8 else now.year - 1
    
    if compare:
        scenarios = [
            {"name": "Conservative", "target_price": current_price * 1.15},
            {"name": "Moderate", "target_price": target_price},
            {"name": "Aggressive", "target_price": current_price * 1.40}
        ]
        
        result = compare_scenarios(
            section_name=section_name,
            current_price=current_price,
            scenarios=scenarios,
            current_season=current_season,
            config=config
        )
    else:
        result = optimize_section_pricing(
            section_name=section_name,
            current_price=current_price,
            target_price=target_price,
            current_season=current_season,
            config=config,
            method=method
        )
    
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    
    if 'optimal_trajectory' in result:
        logger.info(f"Section: {section_name}")
        logger.info(f"Current Price: ${current_price:.2f}")
        logger.info(f"Target Price: ${target_price:.2f}")
        logger.info(f"Optimal Trajectory:")
        for i, price in enumerate(result.get('optimal_trajectory', [])):
            logger.info(f"  Year {i+1}: ${price:.2f}")
        logger.info(f"Expected Retention: {result.get('expected_retention_rate', 0)*100:.1f}%")
        logger.info(f"Total Expected Revenue: ${result.get('total_expected_revenue', 0):,.2f}")
    
    logger.info("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Toledo Ticket Pricing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        help="Seating section name"
    )
    parser.add_argument(
        "--current-price",
        type=float,
        required=True,
        help="Current ticket price"
    )
    parser.add_argument(
        "--target-price",
        type=float,
        required=True,
        help="Target price at end of planning horizon"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Current season year"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="scipy",
        choices=["scipy", "pulp"],
        help="Optimization method"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple scenarios"
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
    
    result = run_optimization(
        config_path=args.config,
        section_name=args.section,
        current_price=args.current_price,
        target_price=args.target_price,
        current_season=args.season,
        method=args.method,
        compare=args.compare
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
