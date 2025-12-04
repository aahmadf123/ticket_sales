"""Price optimization service using PuLP for multi-year ticket pricing.

Key Understanding:
- 3% inflation is the FLOOR/BASELINE - just treading water, no real gain
- Real revenue growth requires increases ABOVE inflation
- Churn constraints limit how aggressive pricing can be
- Optimization balances revenue growth vs churn risk

Pricing Formula (as requested):
    Year N = (Year N-1 * 1.03) + model_predicted_additional

    This means:
    - Y1 = (current + 3% inflation) + predicted_increase_Y1
    - Y2 = (Y1 + 3%) + predicted_increase_Y2
    - And so on...

    The model predicts the ADDITIONAL increase above inflation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from decimal import Decimal
from dataclasses import dataclass, field
from scipy.optimize import minimize
import yaml
from pathlib import Path

from ..entities.pricing_trajectory import (
    PricingTrajectory, YearlyPrice, OptimizationConstraints,
    PricingTier, SeatingSection
)


@dataclass
class OptimizationConfig:
    """Configuration for price optimization.

    Pricing Formula:
        Year N = (Year N-1 * 1.03) + model_predicted_additional

    Note on inflation_floor (3%):
        This is NOT a target - it's the minimum to maintain purchasing power.
        At 3%, you're just keeping up with inflation (no real growth).
        The model predicts ADDITIONAL increases above this baseline.
    """

    # Time horizon
    planning_years: int = 5

    # Constraints
    min_annual_increase: float = 0.00  # Can freeze prices if needed
    max_annual_increase: float = 0.15  # Churn constraint: max 15% per year
    inflation_rate: float = 0.03  # 3% inflation = baseline, always added

    # Churn model parameters
    max_acceptable_churn: float = 0.10
    churn_elasticity: float = 2.0
    baseline_churn: float = 0.05
    churn_threshold: float = 0.05  # Price increases above this trigger churn

    # Objective weights
    revenue_weight: float = 0.6
    attendance_weight: float = 0.2
    retention_weight: float = 0.2

    # Discount rate for NPV (should be > inflation for real returns)
    discount_rate: float = 0.08

    # Elasticity tier configuration (loaded from config file)
    elasticity_tiers: Dict[str, Dict] = field(default_factory=lambda: {
        "premium": {"base_elasticity": -0.35, "optimal_increase": (0.08, 0.12)},
        "mid_tier": {"base_elasticity": -0.6, "optimal_increase": (0.05, 0.08)},
        "standard": {"base_elasticity": -0.9, "optimal_increase": (0.03, 0.05)},
        "budget": {"base_elasticity": -1.3, "optimal_increase": (0.00, 0.03)},
    })


class PriceOptimizationService:
    """Service for optimizing multi-year ticket pricing trajectories.

    Implements the pricing formula:
        Year N = (Year N-1 * 1.03) + model_predicted_additional

    Uses actual ticket prices from configuration, not revenue (event_pmt).
    Elasticity is determined by ticket type/tier, not just calculated from data.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None, ticket_prices_path: str = None):
        self.config = config or OptimizationConfig()
        self.optimization_results = {}
        self.price_data = None  # Will hold extracted price data from ticket sales
        self.churn_model = None  # Optional trained churn model
        self.actual_ticket_prices = {}  # Loaded from config file
        self.elasticity_keywords = {}  # Mapping keywords to elasticity tiers

        # Load actual ticket prices from config
        if ticket_prices_path:
            self._load_ticket_prices(ticket_prices_path)
        else:
            # Try default location
            default_path = Path(__file__).parent.parent.parent.parent / "config" / "ticket_prices.yaml"
            if default_path.exists():
                self._load_ticket_prices(str(default_path))

    def _load_ticket_prices(self, path: str) -> None:
        """Load actual ticket prices from YAML configuration.

        Args:
            path: Path to ticket_prices.yaml
        """
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            # Store pricing configuration
            self.actual_ticket_prices = config
            self.elasticity_tiers = config.get('elasticity_tiers', {})
            self.elasticity_keywords = config.get('data_mapping', {}).get('elasticity_keywords', {})

            print(f"  Loaded ticket prices from: {path}")
        except Exception as e:
            print(f"  Warning: Could not load ticket prices: {e}")

    def set_churn_model(self, churn_model) -> None:
        """Set a trained churn model for more accurate predictions.

        Args:
            churn_model: Trained ChurnModelingService instance
        """
        self.churn_model = churn_model
        if churn_model and hasattr(churn_model, 'is_trained') and churn_model.is_trained:
            print("  Using trained churn model for predictions")
    
    def load_prices_from_data(self, ticket_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract current pricing information and determine elasticity by ticket type.

        Note: event_pmt is REVENUE, not ticket price.
        Elasticity is determined by ticket type tier (premium, mid_tier, standard, budget),
        then refined by data patterns if available.

        Args:
            ticket_df: Preprocessed ticket sales DataFrame

        Returns:
            Dictionary with pricing summary and tier-based elasticities by pr_level
        """
        pricing_summary = {}

        # By price type
        if 'price_type' in ticket_df.columns and 'unit_price' in ticket_df.columns:
            price_by_type = ticket_df.groupby('price_type').agg({
                'unit_price': ['mean', 'median', 'std', 'count'],
                'order_qty': 'sum',
                'event_pmt': 'sum'
            }).reset_index()
            price_by_type.columns = ['price_type', 'avg_price', 'median_price',
                                      'price_std', 'transaction_count',
                                      'total_tickets', 'total_revenue']
            pricing_summary['by_price_type'] = price_by_type.to_dict('records')

        # By seating level - determine elasticity by TIER, not just price level
        if 'pr_level' in ticket_df.columns and 'unit_price' in ticket_df.columns:
            level_data = []

            for level in ticket_df['pr_level'].unique():
                level_df = ticket_df[ticket_df['pr_level'] == level]

                avg_price = level_df['unit_price'].mean()
                total_tickets = level_df['order_qty'].sum()
                total_revenue = level_df['event_pmt'].sum()

                # Get tier-based elasticity (e.g., suites = premium = -0.35)
                elasticity, tier = self._estimate_elasticity_from_data(level_df, level_name=level)

                level_data.append({
                    'pr_level': level,
                    'avg_price': float(avg_price),
                    'median_price': float(level_df['unit_price'].median()),
                    'price_std': float(level_df['unit_price'].std()) if len(level_df) > 1 else 0,
                    'transaction_count': len(level_df),
                    'total_tickets': int(total_tickets),
                    'total_revenue': float(total_revenue),
                    'elasticity': elasticity,
                    'elasticity_tier': tier,  # premium, mid_tier, standard, budget
                })

            pricing_summary['by_pr_level'] = level_data

        # Overall summary
        pricing_summary['overall'] = {
            'avg_price': float(ticket_df['unit_price'].mean()) if 'unit_price' in ticket_df.columns else 0,
            'median_price': float(ticket_df['unit_price'].median()) if 'unit_price' in ticket_df.columns else 0,
            'total_revenue': float(ticket_df['event_pmt'].sum()) if 'event_pmt' in ticket_df.columns else 0,
            'total_tickets': int(ticket_df['order_qty'].sum()) if 'order_qty' in ticket_df.columns else 0,
        }

        self.price_data = pricing_summary
        return pricing_summary
    
    def _determine_elasticity_tier(self, level_name: str, price_type: str = None) -> str:
        """Determine elasticity tier from ticket type name/keywords.

        Uses keyword matching from config to classify tickets.

        Args:
            level_name: The pr_level or seating level name
            price_type: Optional price_type for additional context

        Returns:
            Elasticity tier string: 'premium', 'mid_tier', 'standard', or 'budget'
        """
        search_text = (level_name or "").lower()
        if price_type:
            search_text += " " + price_type.lower()

        # Check keywords from config
        for tier, keywords in self.elasticity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in search_text:
                    return tier

        # Fallback: guess from common patterns
        if any(kw in search_text for kw in ['suite', 'courtside', 'loge', 'club', 'vip']):
            return 'premium'
        elif any(kw in search_text for kw in ['zone a', 'zone b', 'reserved']):
            return 'mid_tier'
        elif any(kw in search_text for kw in ['zone c', 'upper', 'general']):
            return 'standard'
        elif any(kw in search_text for kw in ['student', 'bleacher', 'ga', 'lawn']):
            return 'budget'

        return 'standard'  # Default

    def _get_elasticity_for_tier(self, tier: str) -> float:
        """Get base elasticity value for a tier.

        Args:
            tier: Elasticity tier name

        Returns:
            Base elasticity value (negative)
        """
        tier_config = self.elasticity_tiers.get(tier, {})
        return tier_config.get('base_elasticity', -0.9)

    def _estimate_elasticity_from_data(self, level_df: pd.DataFrame, level_name: str = None) -> Tuple[float, str]:
        """Estimate price elasticity for a seating level.

        Uses TIER-BASED elasticity from config, refined by data if available.

        Tiers (from ticket_prices.yaml):
        - premium: -0.35 (suites, courtside, loge, club) - very inelastic
        - mid_tier: -0.6 (zone A/B, reserved) - moderately inelastic
        - standard: -0.9 (zone C, upper, general) - near unit elastic
        - budget: -1.3 (student, bleacher, GA) - elastic

        Returns:
            Tuple of (elasticity value, tier name)
        """
        # Step 1: Determine tier from ticket type name
        price_type = level_df['price_type'].iloc[0] if 'price_type' in level_df.columns and len(level_df) > 0 else None
        tier = self._determine_elasticity_tier(level_name or "", price_type)
        base_elasticity = self._get_elasticity_for_tier(tier)

        # Step 2: Refine with data if sufficient variation exists
        if len(level_df) < 5 or 'event_code' not in level_df.columns:
            return base_elasticity, tier

        try:
            event_data = level_df.groupby('event_code').agg({
                'unit_price': 'mean',
                'order_qty': 'sum'
            }).reset_index()

            if len(event_data) < 3:
                return base_elasticity, tier

            prices = event_data['unit_price'].values
            quantities = event_data['order_qty'].values

            # Check if there's enough price variation
            price_cv = prices.std() / (prices.mean() + 0.01)

            if price_cv < 0.05:
                # Not enough price variation - use demand stability as adjustment
                qty_cv = quantities.std() / (quantities.mean() + 0.01)
                if qty_cv < 0.2:
                    # Very stable demand = more inelastic than tier base
                    return max(base_elasticity + 0.15, -0.2), tier
                elif qty_cv > 0.5:
                    # Volatile demand = more elastic than tier base
                    return min(base_elasticity - 0.2, -2.0), tier
                return base_elasticity, tier

            # Calculate data-driven elasticity
            log_prices = np.log(prices + 0.01)
            log_quantities = np.log(quantities + 0.01)

            n = len(log_prices)
            sum_x = log_prices.sum()
            sum_y = log_quantities.sum()
            sum_xy = (log_prices * log_quantities).sum()
            sum_x2 = (log_prices ** 2).sum()

            denominator = n * sum_x2 - sum_x ** 2
            if abs(denominator) < 1e-10:
                return base_elasticity, tier

            calculated_elasticity = (n * sum_xy - sum_x * sum_y) / denominator

            if calculated_elasticity >= 0:
                return base_elasticity, tier

            # Blend: tier base anchors, data adjusts
            # More data = more weight on calculated
            weight = min(len(event_data) / 30, 0.5)  # Max 50% weight on data
            blended = weight * calculated_elasticity + (1 - weight) * base_elasticity

            # Keep within tier's reasonable range
            tier_range = self.elasticity_tiers.get(tier, {}).get('range', [-1.5, -0.3])
            blended = max(tier_range[0], min(tier_range[1], blended))

            return blended, tier

        except Exception:
            return base_elasticity, tier
    
    def optimize_all_tickets(
        self,
        ticket_df: pd.DataFrame,
        forecasting_model=None,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Optimize pricing for EACH seating level using tier-based elasticity.

        Implements the pricing formula:
            Year N = (Year N-1 * 1.03) + model_predicted_additional

        Each seating level gets:
        - Elasticity tier (premium, mid_tier, standard, budget) based on ticket type
        - Tier-appropriate price increase recommendations
        - Breakdown of inflation (3%) vs predicted additional

        Args:
            ticket_df: Preprocessed ticket sales DataFrame with actual prices
            forecasting_model: Trained ML model (optional, for demand prediction)
            constraints: Optional custom constraints

        Returns:
            Detailed pricing recommendations by seating level
        """
        # Extract pricing AND elasticity from data
        pricing_summary = self.load_prices_from_data(ticket_df)

        print(f"\n  Determining elasticity tier for each seating level...")

        # Optimize EACH seating level with ITS OWN tier/elasticity
        seating_recommendations = {}

        if 'by_pr_level' in pricing_summary and pricing_summary['by_pr_level']:
            for level_data in pricing_summary['by_pr_level']:
                level_name = level_data['pr_level']
                current_price = level_data['avg_price']
                total_tickets = level_data['total_tickets']
                elasticity = level_data.get('elasticity', -0.9)
                elasticity_tier = level_data.get('elasticity_tier', 'standard')

                if current_price <= 0 or total_tickets <= 0:
                    continue

                print(f"    {level_name}: tier={elasticity_tier}, elasticity={elasticity:.2f}")

                # Optimize with THIS LEVEL'S elasticity and tier
                level_result = self.optimize_trajectory_scipy(
                    current_price=current_price,
                    base_demand=total_tickets,
                    price_elasticity=elasticity,
                    elasticity_tier=elasticity_tier,
                    constraints=constraints
                )

                if level_result.get('optimization_success'):
                    self._add_real_growth_analysis(level_result, current_price)

                    seating_recommendations[level_name] = {
                        'current_price': current_price,
                        'current_tickets': total_tickets,
                        'current_revenue': level_data['total_revenue'],
                        'elasticity': elasticity,
                        'elasticity_tier': elasticity_tier,
                        'optimal_prices': level_result['optimal_prices'],
                        'yearly_increases': level_result['optimal_increases'],
                        'additional_increases': level_result.get('additional_increases', []),
                        'inflation_component': level_result.get('inflation_component', []),
                        'total_5yr_revenue': level_result['total_revenue'],
                        'final_retention': level_result['final_retention_rate'],
                        'beats_inflation': level_result.get('beats_inflation', False),
                        'real_gain_pct': level_result.get('final_real_gain', 0) * 100,
                    }

        # Also optimize by price type with tier-based elasticity
        price_type_recommendations = {}
        if 'by_price_type' in pricing_summary and pricing_summary['by_price_type']:
            for type_data in pricing_summary['by_price_type']:
                type_name = type_data['price_type']
                current_price = type_data['avg_price']
                total_tickets = type_data['total_tickets']

                if current_price <= 0 or total_tickets <= 0:
                    continue

                # Get tier-based elasticity for this price type
                type_df = ticket_df[ticket_df['price_type'] == type_name]
                elasticity, tier = self._estimate_elasticity_from_data(type_df, level_name=type_name)

                type_result = self.optimize_trajectory_scipy(
                    current_price=current_price,
                    base_demand=total_tickets,
                    price_elasticity=elasticity,
                    elasticity_tier=tier,
                    constraints=constraints
                )

                if type_result.get('optimization_success'):
                    self._add_real_growth_analysis(type_result, current_price)

                    price_type_recommendations[type_name] = {
                        'current_price': current_price,
                        'current_tickets': total_tickets,
                        'current_revenue': type_data['total_revenue'],
                        'elasticity': elasticity,
                        'elasticity_tier': tier,
                        'optimal_prices': type_result['optimal_prices'],
                        'yearly_increases': type_result['optimal_increases'],
                        'additional_increases': type_result.get('additional_increases', []),
                        'total_5yr_revenue': type_result['total_revenue'],
                        'final_retention': type_result['final_retention_rate'],
                        'beats_inflation': type_result.get('beats_inflation', False),
                        'real_gain_pct': type_result.get('final_real_gain', 0) * 100,
                    }
        
        return {
            'seating_recommendations': seating_recommendations,
            'price_type_recommendations': price_type_recommendations,
            'current_data_summary': pricing_summary['overall'],
            'inflation_rate': self.config.inflation_rate,
            'planning_years': self.config.planning_years,
        }
    
    def _add_real_growth_analysis(self, results: Dict[str, Any], initial_price: float):
        """Add analysis of real (inflation-adjusted) growth.
        
        3% inflation means:
        - Year 1: $100 -> $103 is NO REAL GAIN
        - Year 1: $100 -> $105 is only 2% real gain
        """
        inflation = self.config.inflation_rate
        
        if 'optimal_prices' not in results or not results['optimal_prices']:
            return
        
        prices = results['optimal_prices']
        real_growth_analysis = []
        
        cumulative_inflation = 1.0
        for year, price in enumerate(prices[1:], 1):
            cumulative_inflation *= (1 + inflation)
            inflation_adjusted_initial = initial_price * cumulative_inflation
            
            nominal_increase = (price - initial_price) / initial_price
            real_increase = (price - inflation_adjusted_initial) / inflation_adjusted_initial
            
            real_growth_analysis.append({
                'year': year,
                'nominal_price': price,
                'inflation_adjusted_baseline': inflation_adjusted_initial,
                'nominal_increase_from_start': nominal_increase,
                'real_increase_from_baseline': real_increase,
                'beating_inflation': price > inflation_adjusted_initial,
            })
        
        results['real_growth_analysis'] = real_growth_analysis
        
        # Summary
        final_price = prices[-1]
        final_inflation_baseline = initial_price * (1 + inflation) ** len(prices[1:])
        results['final_real_gain'] = (final_price - final_inflation_baseline) / final_inflation_baseline
        results['beats_inflation'] = final_price > final_inflation_baseline
    
    def optimize_trajectory_scipy(
        self,
        current_price: float,
        base_demand: int,
        price_elasticity: float = -0.5,
        elasticity_tier: str = "standard",
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Optimize pricing trajectory with the formula:

            Year N = (Year N-1 * 1.03) + model_predicted_additional

        The optimizer determines the ADDITIONAL increase above inflation.
        This is the key insight: 3% is ALWAYS added (inflation baseline),
        and the model predicts how much MORE to charge on top.

        Args:
            current_price: Current ticket price (from data)
            base_demand: Baseline demand at current price (from data)
            price_elasticity: Price elasticity of demand (from tier/data)
            elasticity_tier: Tier name (premium, mid_tier, standard, budget)
            constraints: Optimization constraints

        Returns:
            Optimization results with:
            - optimal_prices: [current, y1, y2, y3, y4, y5]
            - optimal_increases: total increases (inflation + additional)
            - additional_increases: the model-predicted portion above inflation
        """
        if constraints is None:
            constraints = OptimizationConstraints(
                min_annual_increase=self.config.min_annual_increase,
                max_annual_increase=self.config.max_annual_increase,
                inflation_floor=self.config.inflation_rate,
                max_churn_rate=self.config.max_acceptable_churn,
            )

        n_years = self.config.planning_years
        inflation = self.config.inflation_rate  # 3% baseline

        # Get optimal additional increase range from tier
        tier_config = self.config.elasticity_tiers.get(elasticity_tier, {})
        optimal_range = tier_config.get('optimal_increase', (0.03, 0.06))

        # Start with tier-suggested additional increase
        initial_additional = (optimal_range[0] + optimal_range[1]) / 2

        # Optimizer finds the ADDITIONAL increase above inflation
        # Bounds for additional: [0%, max_increase - inflation]
        max_additional = constraints.max_annual_increase - inflation

        def objective(additional_increases):
            """Objective: maximize REAL revenue.

            Price formula: P_n = P_{n-1} * (1 + inflation) + P_{n-1} * additional
                         = P_{n-1} * (1 + inflation + additional)
            """
            prices = [current_price]
            for additional in additional_increases:
                total_increase = inflation + additional  # 3% + predicted
                prices.append(prices[-1] * (1 + total_increase))

            total_real_revenue = 0
            total_attendance = 0
            cumulative_retention = 1.0

            prev_price = current_price
            for year, price in enumerate(prices[1:], 1):
                total_increase_pct = (price - prev_price) / prev_price

                # Churn from total price increase
                churn = self._calculate_churn(total_increase_pct)
                cumulative_retention *= (1 - churn)

                # Demand change from elasticity
                price_ratio = price / current_price
                demand_multiplier = price_ratio ** price_elasticity
                demand = max(1, int(base_demand * demand_multiplier * cumulative_retention))

                # Revenue (discounted to present value)
                nominal_revenue = price * demand
                real_discount_factor = (1 + self.config.discount_rate) ** year
                inflation_factor = (1 + inflation) ** year
                real_revenue = nominal_revenue / (real_discount_factor * inflation_factor)

                total_real_revenue += real_revenue
                total_attendance += demand

                prev_price = price

            # Objective: maximize real revenue with secondary objectives
            objective_value = -(
                0.7 * total_real_revenue / (current_price * base_demand) +
                0.2 * total_attendance / (base_demand * n_years) +
                0.1 * cumulative_retention
            )

            return objective_value

        # Bounds for ADDITIONAL increases (above inflation)
        # Additional can be 0% to (max_total - inflation)
        bounds = [
            (0.0, max_additional)
            for _ in range(n_years)
        ]

        # Start with tier-suggested additional increase
        x0 = [initial_additional] * n_years

        # Optimize the ADDITIONAL increases
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False}
        )

        # Extract optimized additional increases
        optimal_additional = result.x if result.success else x0

        # Calculate total increases (inflation + additional)
        optimal_total_increases = [inflation + add for add in optimal_additional]

        # Calculate prices with the formula: P_n = P_{n-1} * (1 + total_increase)
        optimal_prices = [current_price]
        for total_inc in optimal_total_increases:
            optimal_prices.append(optimal_prices[-1] * (1 + total_inc))

        # Calculate metrics
        metrics = self._calculate_trajectory_metrics(
            optimal_prices, base_demand, price_elasticity, current_price
        )

        self.optimization_results = {
            "optimal_prices": optimal_prices,
            "optimal_increases": optimal_total_increases,  # Total = inflation + additional
            "additional_increases": list(optimal_additional),  # Model-predicted portion
            "inflation_component": [inflation] * n_years,  # Always 3%
            "total_revenue": metrics["total_revenue"],
            "total_attendance": metrics["total_attendance"],
            "final_retention_rate": metrics["final_retention_rate"],
            "yearly_metrics": metrics["yearly_metrics"],
            "optimization_success": result.success,
            "price_elasticity_used": price_elasticity,
            "elasticity_tier": elasticity_tier,
            "inflation_rate": inflation,
        }

        return self.optimization_results
    
    def _calculate_optimal_increase_from_elasticity(
        self,
        elasticity: float,
        min_increase: float,
        max_increase: float,
        inflation: float
    ) -> float:
        """Calculate optimal price increase based on price elasticity.
        
        Economics:
        - Elasticity = -0.3 (very inelastic): 10% price increase → only 3% demand drop
          Revenue change: +10% - 3% = +7% → CAN BE AGGRESSIVE
        - Elasticity = -1.0 (unit elastic): 10% price increase → 10% demand drop
          Revenue change: +10% - 10% = 0% → NEUTRAL
        - Elasticity = -1.5 (elastic): 10% price increase → 15% demand drop
          Revenue change: +10% - 15% = -5% → BE CAREFUL
        
        Returns:
            Optimal annual price increase
        """
        # The revenue-maximizing price increase given elasticity
        # Revenue = P * Q, where Q = Q0 * (P/P0)^elasticity
        # dRevenue/dP = 0 when increase = -1/(elasticity) - 1
        # But this is theoretical max, we also consider churn
        
        if elasticity >= -0.1:
            elasticity = -0.1  # Can't be zero or positive
        
        # Theoretical optimal: more inelastic = higher optimal increase
        # At elasticity = -0.5: theoretical optimal is very high, but limited by churn
        # At elasticity = -1.5: theoretical optimal is lower
        
        # Map elasticity to optimal increase:
        # elasticity -0.3 → increase ~12% (aggressive, inelastic)
        # elasticity -0.5 → increase ~10%
        # elasticity -0.8 → increase ~7%
        # elasticity -1.0 → increase ~5% (unit elastic)
        # elasticity -1.5 → increase ~4% (elastic, careful)
        # elasticity -2.0 → increase ~3.5% (very elastic)
        
        # Linear interpolation based on elasticity
        # Range: elasticity [-2.0, -0.3] → increase [3.5%, 12%]
        elasticity_clamped = max(-2.0, min(-0.3, elasticity))
        
        # Map [-2.0, -0.3] to [0.035, 0.12]
        # y = mx + b where x is elasticity
        # At x=-2.0: y=0.035, At x=-0.3: y=0.12
        # slope = (0.12 - 0.035) / (-0.3 - (-2.0)) = 0.085 / 1.7 = 0.05
        # y - 0.035 = 0.05 * (x - (-2.0))
        # y = 0.05 * x + 0.1 + 0.035 = 0.05x + 0.135
        
        optimal = 0.05 * elasticity_clamped + 0.135
        
        # Must be above inflation to have real gains
        optimal = max(optimal, inflation + 0.01)
        
        # Respect bounds
        optimal = max(min_increase, min(max_increase, optimal))
        
        return optimal
    
    def optimize_trajectory_pulp(
        self,
        current_price: float,
        base_demand: int,
        price_elasticity: float = -0.5,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Optimize pricing trajectory using PuLP (linear programming).
        
        For linear approximation, we discretize the price increases
        and formulate as a mixed-integer program.
        
        Args:
            current_price: Current ticket price
            base_demand: Baseline demand at current price
            price_elasticity: Price elasticity of demand
            constraints: Optimization constraints
        
        Returns:
            Optimization results dictionary
        """
        try:
            import pulp
        except ImportError:
            raise ImportError("PuLP is required for this optimization method")
        
        if constraints is None:
            constraints = OptimizationConstraints(
                min_annual_increase=self.config.min_annual_increase,
                max_annual_increase=self.config.max_annual_increase,
                inflation_floor=self.config.inflation_floor,
                max_churn_rate=self.config.max_acceptable_churn,
            )
        
        n_years = self.config.planning_years
        
        # Discretize price increases (0% to 15% in 1% increments)
        increase_options = np.arange(0, 0.16, 0.01)
        n_options = len(increase_options)
        
        # Create problem
        prob = pulp.LpProblem("TicketPriceOptimization", pulp.LpMaximize)
        
        # Decision variables: binary selection of price increase for each year
        x = pulp.LpVariable.dicts(
            "x",
            ((y, i) for y in range(n_years) for i in range(n_options)),
            cat="Binary"
        )
        
        # Continuous variables for prices
        prices = pulp.LpVariable.dicts(
            "price",
            range(n_years + 1),
            lowBound=0
        )
        
        # Set initial price
        prob += prices[0] == current_price
        
        # Only one increase option per year
        for y in range(n_years):
            prob += pulp.lpSum(x[y, i] for i in range(n_options)) == 1
        
        # Price update constraint
        for y in range(n_years):
            prob += prices[y + 1] == prices[y] * (
                1 + pulp.lpSum(increase_options[i] * x[y, i] for i in range(n_options))
            )
        
        # Inflation floor constraint
        for y in range(n_years):
            prob += pulp.lpSum(
                increase_options[i] * x[y, i] for i in range(n_options)
            ) >= constraints.inflation_floor
        
        # Churn constraint (linearized)
        for y in range(n_years):
            # Approximate: churn increases linearly above 5%
            prob += pulp.lpSum(
                self._calculate_churn(increase_options[i]) * x[y, i]
                for i in range(n_options)
            ) <= constraints.max_churn_rate
        
        # Objective: maximize sum of prices (simplified proxy for revenue)
        # In reality, we would need to linearize the demand function
        prob += pulp.lpSum(
            prices[y] / ((1 + self.config.discount_rate) ** y)
            for y in range(1, n_years + 1)
        )
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] == "Optimal":
            # Extract optimal prices
            optimal_prices = [pulp.value(prices[y]) for y in range(n_years + 1)]
            optimal_increases = [
                (optimal_prices[y + 1] - optimal_prices[y]) / optimal_prices[y]
                for y in range(n_years)
            ]
            
            # Calculate metrics
            metrics = self._calculate_trajectory_metrics(
                optimal_prices, base_demand, price_elasticity, current_price
            )
            
            self.optimization_results = {
                "optimal_prices": optimal_prices,
                "optimal_increases": optimal_increases,
                "total_revenue": metrics["total_revenue"],
                "total_attendance": metrics["total_attendance"],
                "final_retention_rate": metrics["final_retention_rate"],
                "yearly_metrics": metrics["yearly_metrics"],
                "optimization_success": True,
                "optimization_method": "PuLP",
            }
        else:
            self.optimization_results = {
                "optimal_prices": None,
                "optimization_success": False,
                "optimization_message": f"Status: {pulp.LpStatus[prob.status]}",
            }
        
        return self.optimization_results
    
    def _calculate_churn(self, price_increase_pct: float) -> float:
        """Calculate churn rate from price increase.
        
        Uses trained churn model if available, otherwise falls back to
        simple elasticity model.
        
        Args:
            price_increase_pct: Annual price increase as decimal
        
        Returns:
            Expected churn rate
        """
        # Use trained churn model if available
        if self.churn_model is not None and hasattr(self.churn_model, 'estimate_churn_from_price_increase'):
            return self.churn_model.estimate_churn_from_price_increase(
                current_churn_rate=self.config.baseline_churn,
                price_increase_pct=price_increase_pct,
                price_elasticity=self.config.churn_elasticity
            )
        
        # Fallback: simple elasticity model
        threshold = self.config.churn_threshold  # Default 5%
        
        if price_increase_pct <= threshold:
            return self.config.baseline_churn
        
        excess = price_increase_pct - threshold
        churn = self.config.baseline_churn + excess * self.config.churn_elasticity
        
        return min(churn, 0.30)  # Cap at 30%
    
    def _calculate_trajectory_metrics(
        self,
        prices: List[float],
        base_demand: int,
        price_elasticity: float,
        initial_price: float
    ) -> Dict[str, Any]:
        """Calculate metrics for a price trajectory.
        
        Args:
            prices: List of prices by year
            base_demand: Baseline demand
            price_elasticity: Price elasticity of demand
            initial_price: Initial price for reference
        
        Returns:
            Metrics dictionary
        """
        total_revenue = 0
        total_attendance = 0
        cumulative_retention = 1.0
        yearly_metrics = []
        
        prev_price = prices[0]
        
        for year, price in enumerate(prices[1:], 1):
            # Calculate increase
            increase_pct = (price - prev_price) / prev_price
            
            # Calculate churn
            churn = self._calculate_churn(increase_pct)
            cumulative_retention *= (1 - churn)
            
            # Calculate demand
            price_ratio = price / initial_price
            demand_multiplier = price_ratio ** price_elasticity
            demand = int(base_demand * demand_multiplier * cumulative_retention)
            
            # Calculate revenue
            revenue = price * demand
            discounted_revenue = revenue / ((1 + self.config.discount_rate) ** year)
            
            yearly_metrics.append({
                "year": year,
                "price": price,
                "price_increase_pct": increase_pct,
                "churn_rate": churn,
                "cumulative_retention": cumulative_retention,
                "demand": demand,
                "revenue": revenue,
                "discounted_revenue": discounted_revenue,
            })
            
            total_revenue += discounted_revenue
            total_attendance += demand
            
            prev_price = price
        
        return {
            "total_revenue": total_revenue,
            "total_attendance": total_attendance,
            "final_retention_rate": cumulative_retention,
            "yearly_metrics": yearly_metrics,
        }
    
    def create_pricing_trajectory(
        self,
        section: SeatingSection,
        current_price: float,
        current_season: str,
        base_demand: int,
        price_elasticity: float = -0.5,
        method: str = "scipy"
    ) -> PricingTrajectory:
        """Create an optimized pricing trajectory.
        
        Args:
            section: Stadium section
            current_price: Current price
            current_season: Current season code
            base_demand: Baseline demand
            price_elasticity: Price elasticity
            method: Optimization method ('scipy' or 'pulp')
        
        Returns:
            PricingTrajectory entity
        """
        # Run optimization
        if method == "pulp":
            results = self.optimize_trajectory_pulp(
                current_price, base_demand, price_elasticity
            )
        else:
            results = self.optimize_trajectory_scipy(
                current_price, base_demand, price_elasticity
            )
        
        if not results.get("optimization_success", False):
            # Return trajectory with default inflation
            return self._create_default_trajectory(
                section, current_price, current_season
            )
        
        # Create trajectory entity
        trajectory = PricingTrajectory(
            section=section,
            pricing_tier=self._determine_pricing_tier(current_price),
            current_price=Decimal(str(current_price)),
            current_season=current_season,
            planning_horizon_years=self.config.planning_years,
            base_inflation_rate=self.config.inflation_floor,
            min_annual_increase=self.config.min_annual_increase,
            max_annual_increase=self.config.max_annual_increase,
            max_churn_rate=self.config.max_acceptable_churn,
        )
        
        # Set optimal trajectory
        trajectory.optimal_trajectory = [
            Decimal(str(p)) for p in results["optimal_prices"]
        ]
        
        # Create yearly prices
        yearly_metrics = results.get("yearly_metrics", [])
        trajectory.yearly_prices = []
        
        for metrics in yearly_metrics:
            yearly_price = YearlyPrice(
                year=metrics["year"],
                base_price=Decimal(str(metrics["price"])),
                adjusted_price=Decimal(str(metrics["price"])),
                expected_demand=metrics["demand"],
                expected_revenue=Decimal(str(metrics["revenue"])),
            )
            trajectory.yearly_prices.append(yearly_price)
        
        # Set totals
        trajectory.total_expected_revenue = Decimal(str(results["total_revenue"]))
        trajectory.total_expected_attendance = results["total_attendance"]
        trajectory.expected_retention_rate = results["final_retention_rate"]
        
        return trajectory
    
    def _determine_pricing_tier(self, price: float) -> PricingTier:
        """Determine pricing tier from price."""
        if price > 75:
            return PricingTier.PREMIUM
        elif price > 35:
            return PricingTier.STANDARD
        else:
            return PricingTier.VALUE
    
    def _create_default_trajectory(
        self,
        section: SeatingSection,
        current_price: float,
        current_season: str
    ) -> PricingTrajectory:
        """Create default trajectory with inflation only."""
        trajectory = PricingTrajectory(
            section=section,
            pricing_tier=self._determine_pricing_tier(current_price),
            current_price=Decimal(str(current_price)),
            current_season=current_season,
            planning_horizon_years=self.config.planning_years,
        )
        
        return trajectory
    
    def compare_scenarios(
        self,
        current_price: float,
        base_demand: int,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare different pricing scenarios.
        
        Args:
            current_price: Current price
            base_demand: Baseline demand
            scenarios: List of scenario definitions
        
        Returns:
            Comparison results
        """
        comparison = []
        
        for scenario in scenarios:
            name = scenario.get("name", "Unnamed")
            increases = scenario.get("increases", [0.05] * self.config.planning_years)
            
            prices = [current_price]
            for inc in increases:
                prices.append(prices[-1] * (1 + inc))
            
            metrics = self._calculate_trajectory_metrics(
                prices, base_demand, scenario.get("elasticity", -0.5), current_price
            )
            
            comparison.append({
                "scenario_name": name,
                "prices": prices,
                "total_revenue": metrics["total_revenue"],
                "total_attendance": metrics["total_attendance"],
                "final_retention_rate": metrics["final_retention_rate"],
            })
        
        return {
            "scenarios": comparison,
            "best_revenue": max(comparison, key=lambda x: x["total_revenue"]),
            "best_attendance": max(comparison, key=lambda x: x["total_attendance"]),
            "best_retention": max(comparison, key=lambda x: x["final_retention_rate"]),
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of last optimization."""
        if not self.optimization_results:
            return {"status": "No optimization run"}
        
        return {
            "success": self.optimization_results.get("optimization_success", False),
            "optimal_prices": self.optimization_results.get("optimal_prices"),
            "optimal_increases": self.optimization_results.get("optimal_increases"),
            "total_npv_revenue": self.optimization_results.get("total_revenue"),
            "final_retention": self.optimization_results.get("final_retention_rate"),
        }
