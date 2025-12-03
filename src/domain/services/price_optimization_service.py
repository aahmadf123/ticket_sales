"""Price optimization service using PuLP for multi-year ticket pricing.

Key Understanding:
- 3% inflation is the FLOOR/BASELINE - just treading water, no real gain
- Real revenue growth requires increases ABOVE inflation
- Churn constraints limit how aggressive pricing can be
- Optimization balances revenue growth vs churn risk
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from decimal import Decimal
from dataclasses import dataclass
from scipy.optimize import minimize

from ..entities.pricing_trajectory import (
    PricingTrajectory, YearlyPrice, OptimizationConstraints,
    PricingTier, SeatingSection
)


@dataclass
class OptimizationConfig:
    """Configuration for price optimization.
    
    Note on inflation_floor (3%):
        This is NOT a target - it's the minimum to maintain purchasing power.
        At 3%, you're just keeping up with inflation (no real growth).
        Real revenue growth requires increases ABOVE 3%.
    """
    
    # Time horizon
    planning_years: int = 5
    
    # Constraints
    min_annual_increase: float = 0.00  # Can freeze prices if needed
    max_annual_increase: float = 0.15  # Churn constraint: max 15% per year
    inflation_rate: float = 0.03  # 3% inflation = no real gain, just treading water
    
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


class PriceOptimizationService:
    """Service for optimizing multi-year ticket pricing trajectories.
    
    Works with ALL ticket types from the data, not section-by-section.
    Prices come from actual ticket sales data, not hardcoded values.
    Can use a trained churn model for more accurate churn predictions.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_results = {}
        self.price_data = None  # Will hold extracted price data from ticket sales
        self.churn_model = None  # Optional trained churn model
    
    def set_churn_model(self, churn_model) -> None:
        """Set a trained churn model for more accurate predictions.
        
        Args:
            churn_model: Trained ChurnModelingService instance
        """
        self.churn_model = churn_model
        if churn_model and hasattr(churn_model, 'is_trained') and churn_model.is_trained:
            print("  Using trained churn model for predictions")
    
    def load_prices_from_data(self, ticket_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract current pricing information and calculate price elasticity from data.
        
        Price elasticity is LEARNED from the data - how does quantity change when price changes?
        Different seating levels will have DIFFERENT elasticities.
        
        Args:
            ticket_df: Preprocessed ticket sales DataFrame
        
        Returns:
            Dictionary with pricing summary and learned elasticities by pr_level
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
        
        # By seating level - LEARN elasticity for each level
        if 'pr_level' in ticket_df.columns and 'unit_price' in ticket_df.columns:
            level_data = []
            
            for level in ticket_df['pr_level'].unique():
                level_df = ticket_df[ticket_df['pr_level'] == level]
                
                avg_price = level_df['unit_price'].mean()
                total_tickets = level_df['order_qty'].sum()
                total_revenue = level_df['event_pmt'].sum()
                
                # Calculate price elasticity from variance in the data
                # Higher price variance with stable quantity = INELASTIC (can raise prices)
                # Price changes causing big quantity swings = ELASTIC (be careful)
                elasticity = self._estimate_elasticity_from_data(level_df)
                
                level_data.append({
                    'pr_level': level,
                    'avg_price': float(avg_price),
                    'median_price': float(level_df['unit_price'].median()),
                    'price_std': float(level_df['unit_price'].std()) if len(level_df) > 1 else 0,
                    'transaction_count': len(level_df),
                    'total_tickets': int(total_tickets),
                    'total_revenue': float(total_revenue),
                    'elasticity': elasticity,  # LEARNED from data
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
    
    def _estimate_elasticity_from_data(self, level_df: pd.DataFrame) -> float:
        """Estimate price elasticity for a seating level from historical data.
        
        Uses multiple signals:
        1. Price-quantity correlation (if data varies enough)
        2. Price level as proxy (premium seats tend to be more inelastic)
        3. Demand consistency (stable demand = inelastic)
        
        Returns:
            Estimated price elasticity (negative value, e.g., -0.3 to -1.5)
            More negative = more elastic (price sensitive)
            Less negative = more inelastic (can raise prices with less demand loss)
        """
        # Default varies by price level (proxy for seat quality)
        avg_price = level_df['unit_price'].mean() if 'unit_price' in level_df.columns else 20.0
        
        # Premium seats (higher price) tend to be more inelastic
        # Budget seats tend to be more elastic
        if avg_price >= 40:
            base_elasticity = -0.4  # Premium - inelastic
        elif avg_price >= 25:
            base_elasticity = -0.6  # Mid-tier
        elif avg_price >= 15:
            base_elasticity = -0.9  # Standard
        else:
            base_elasticity = -1.2  # Budget - elastic
        
        if len(level_df) < 5 or 'event_code' not in level_df.columns:
            return base_elasticity
        
        # Try to estimate from actual price-quantity variation
        try:
            event_data = level_df.groupby('event_code').agg({
                'unit_price': 'mean',
                'order_qty': 'sum'
            }).reset_index()
            
            if len(event_data) < 3:
                return base_elasticity
            
            prices = event_data['unit_price'].values
            quantities = event_data['order_qty'].values
            
            # Check if there's enough price variation to estimate
            price_cv = prices.std() / (prices.mean() + 0.01)  # Coefficient of variation
            
            if price_cv < 0.05:  # Less than 5% price variation - can't estimate
                # Use demand stability as proxy instead
                qty_cv = quantities.std() / (quantities.mean() + 0.01)
                if qty_cv < 0.2:
                    # Very stable demand = inelastic
                    return max(base_elasticity + 0.2, -0.3)
                elif qty_cv > 0.5:
                    # Volatile demand = elastic
                    return min(base_elasticity - 0.3, -1.5)
                return base_elasticity
            
            # Enough price variation - calculate elasticity
            log_prices = np.log(prices + 0.01)
            log_quantities = np.log(quantities + 0.01)
            
            # Correlation-based elasticity
            n = len(log_prices)
            sum_x = log_prices.sum()
            sum_y = log_quantities.sum()
            sum_xy = (log_prices * log_quantities).sum()
            sum_x2 = (log_prices ** 2).sum()
            
            denominator = n * sum_x2 - sum_x ** 2
            if abs(denominator) < 1e-10:
                return base_elasticity
            
            calculated_elasticity = (n * sum_xy - sum_x * sum_y) / denominator
            
            # If calculated elasticity is positive (unusual), use base
            if calculated_elasticity >= 0:
                return base_elasticity
            
            # Blend calculated with base (data-driven but regularized)
            # Weight by sample size
            weight = min(len(event_data) / 20, 0.7)  # Max 70% weight on data
            blended = weight * calculated_elasticity + (1 - weight) * base_elasticity
            
            # Bound to reasonable range
            return max(-2.0, min(-0.3, blended))
            
        except:
            return base_elasticity
    
    def optimize_all_tickets(
        self,
        ticket_df: pd.DataFrame,
        forecasting_model=None,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Optimize pricing for EACH seating level using LEARNED price elasticity.
        
        Each seating level gets a DIFFERENT optimal increase based on:
        - Its own price elasticity (learned from data)
        - Its demand patterns
        - Churn constraints
        
        Args:
            ticket_df: Preprocessed ticket sales DataFrame with actual prices
            forecasting_model: Trained ML model (optional, for demand prediction)
            constraints: Optional custom constraints
        
        Returns:
            Detailed pricing recommendations by seating level
        """
        # Extract pricing AND elasticity from data
        pricing_summary = self.load_prices_from_data(ticket_df)
        
        print(f"\n  Learning price elasticity for each seating level...")
        
        # Optimize EACH seating level with ITS OWN elasticity
        seating_recommendations = {}
        
        if 'by_pr_level' in pricing_summary and pricing_summary['by_pr_level']:
            for level_data in pricing_summary['by_pr_level']:
                level_name = level_data['pr_level']
                current_price = level_data['avg_price']
                total_tickets = level_data['total_tickets']
                elasticity = level_data.get('elasticity', -0.5)  # LEARNED from data
                
                if current_price <= 0 or total_tickets <= 0:
                    continue
                
                print(f"    {level_name}: elasticity = {elasticity:.2f}")
                
                # Optimize with THIS LEVEL'S elasticity
                level_result = self.optimize_trajectory_scipy(
                    current_price=current_price,
                    base_demand=total_tickets,
                    price_elasticity=elasticity,  # DIFFERENT for each level
                    constraints=constraints
                )
                
                if level_result.get('optimization_success'):
                    self._add_real_growth_analysis(level_result, current_price)
                    
                    seating_recommendations[level_name] = {
                        'current_price': current_price,
                        'current_tickets': total_tickets,
                        'current_revenue': level_data['total_revenue'],
                        'elasticity': elasticity,
                        'optimal_prices': level_result['optimal_prices'],
                        'yearly_increases': level_result['optimal_increases'],
                        'total_5yr_revenue': level_result['total_revenue'],
                        'final_retention': level_result['final_retention_rate'],
                        'beats_inflation': level_result.get('beats_inflation', False),
                        'real_gain_pct': level_result.get('final_real_gain', 0) * 100,
                    }
        
        # Also optimize by price type with learned elasticity
        price_type_recommendations = {}
        if 'by_price_type' in pricing_summary and pricing_summary['by_price_type']:
            # Calculate elasticity for price types too
            for type_data in pricing_summary['by_price_type']:
                type_name = type_data['price_type']
                current_price = type_data['avg_price']
                total_tickets = type_data['total_tickets']
                
                if current_price <= 0 or total_tickets <= 0:
                    continue
                
                # Get elasticity for this price type
                type_df = ticket_df[ticket_df['price_type'] == type_name]
                elasticity = self._estimate_elasticity_from_data(type_df)
                
                type_result = self.optimize_trajectory_scipy(
                    current_price=current_price,
                    base_demand=total_tickets,
                    price_elasticity=elasticity,
                    constraints=constraints
                )
                
                if type_result.get('optimization_success'):
                    self._add_real_growth_analysis(type_result, current_price)
                    
                    price_type_recommendations[type_name] = {
                        'current_price': current_price,
                        'current_tickets': total_tickets,
                        'current_revenue': type_data['total_revenue'],
                        'elasticity': elasticity,
                        'optimal_prices': type_result['optimal_prices'],
                        'yearly_increases': type_result['optimal_increases'],
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
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Optimize pricing trajectory where ELASTICITY drives the optimal increase.
        
        Key insight from economics:
        - INELASTIC demand (elasticity close to 0): Can increase prices aggressively
          because quantity doesn't drop much. Revenue = P * Q, if Q barely changes, higher P = more revenue.
        - ELASTIC demand (elasticity < -1): Must be careful with increases
          because quantity drops faster than price rises. Could LOSE revenue.
        
        Optimal price increase formula (Lerner Index):
        For revenue maximization: optimal_markup = -1 / elasticity
        But constrained by churn and practical limits.
        
        Args:
            current_price: Current ticket price (from data)
            base_demand: Baseline demand at current price (from data)
            price_elasticity: Price elasticity of demand (LEARNED from data)
            constraints: Optimization constraints
        
        Returns:
            Optimization results dictionary with DIFFERENT increases based on elasticity
        """
        if constraints is None:
            constraints = OptimizationConstraints(
                min_annual_increase=self.config.min_annual_increase,
                max_annual_increase=self.config.max_annual_increase,
                inflation_floor=self.config.inflation_rate,
                max_churn_rate=self.config.max_acceptable_churn,
            )
        
        n_years = self.config.planning_years
        inflation = self.config.inflation_rate
        
        # Calculate the OPTIMAL annual increase based on elasticity
        # This is the key differentiator - different elasticities = different optimal increases
        optimal_annual_increase = self._calculate_optimal_increase_from_elasticity(
            price_elasticity,
            constraints.min_annual_increase,
            constraints.max_annual_increase,
            inflation
        )
        
        # Now optimize with this as the starting point, but let optimizer fine-tune
        def objective(price_increases):
            """Objective: maximize REAL revenue considering elasticity-driven demand."""
            prices = [current_price]
            for inc in price_increases:
                prices.append(prices[-1] * (1 + inc))
            
            total_real_revenue = 0
            total_attendance = 0
            cumulative_retention = 1.0
            
            prev_price = current_price
            for year, price in enumerate(prices[1:], 1):
                increase_pct = (price - prev_price) / prev_price
                
                # Churn from price increase
                churn = self._calculate_churn(increase_pct)
                cumulative_retention *= (1 - churn)
                
                # Demand change from elasticity - THIS IS WHERE ELASTICITY MATTERS
                # Q_new = Q_old * (P_new / P_old)^elasticity
                price_ratio = price / current_price
                demand_multiplier = price_ratio ** price_elasticity
                demand = max(1, int(base_demand * demand_multiplier * cumulative_retention))
                
                # Revenue
                nominal_revenue = price * demand
                real_discount_factor = (1 + self.config.discount_rate) ** year
                inflation_factor = (1 + inflation) ** year
                real_revenue = nominal_revenue / (real_discount_factor * inflation_factor)
                
                total_real_revenue += real_revenue
                total_attendance += demand
                
                prev_price = price
            
            # Objective: maximize real revenue (primary) with attendance and retention as secondary
            objective_value = -(
                0.7 * total_real_revenue / (current_price * base_demand) +  # Normalized revenue
                0.2 * total_attendance / (base_demand * n_years) +  # Normalized attendance
                0.1 * cumulative_retention  # Retention rate
            )
            
            return objective_value
        
        # Bounds
        bounds = [
            (constraints.min_annual_increase, constraints.max_annual_increase)
            for _ in range(n_years)
        ]
        
        # Start with the elasticity-optimal increase
        x0 = [optimal_annual_increase] * n_years
        
        # Optimize - but the starting point already reflects elasticity
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False}
        )
        
        # Use the optimized result
        optimal_increases = result.x if result.success else x0
        
        optimal_prices = [current_price]
        for inc in optimal_increases:
            optimal_prices.append(optimal_prices[-1] * (1 + inc))
        
        # Calculate metrics
        metrics = self._calculate_trajectory_metrics(
            optimal_prices, base_demand, price_elasticity, current_price
        )
        
        self.optimization_results = {
            "optimal_prices": optimal_prices,
            "optimal_increases": list(optimal_increases),
            "total_revenue": metrics["total_revenue"],
            "total_attendance": metrics["total_attendance"],
            "final_retention_rate": metrics["final_retention_rate"],
            "yearly_metrics": metrics["yearly_metrics"],
            "optimization_success": result.success,
            "price_elasticity_used": price_elasticity,
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
