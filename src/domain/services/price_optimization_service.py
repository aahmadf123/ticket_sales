"""Price optimization service using PuLP for multi-year ticket pricing."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from decimal import Decimal
from dataclasses import dataclass
from scipy.optimize import minimize

from ..entities.pricing_trajectory import (
    PricingTrajectory, YearlyPrice, OptimizationConstraints,
    PricingTier, SeatingSection
)


@dataclass
class OptimizationConfig:
    """Configuration for price optimization."""
    
    # Time horizon
    planning_years: int = 5
    
    # Constraints
    min_annual_increase: float = 0.00
    max_annual_increase: float = 0.15  # Churn constraint: max 15% per year
    inflation_floor: float = 0.03
    
    # Churn model parameters
    max_acceptable_churn: float = 0.10
    churn_elasticity: float = 2.0
    baseline_churn: float = 0.05
    
    # Objective weights
    revenue_weight: float = 0.6
    attendance_weight: float = 0.2
    retention_weight: float = 0.2
    
    # Discount rate for NPV
    discount_rate: float = 0.08


class PriceOptimizationService:
    """Service for optimizing multi-year ticket pricing trajectories."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_results = {}
    
    def optimize_trajectory_scipy(
        self,
        current_price: float,
        base_demand: int,
        price_elasticity: float = -0.5,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, Any]:
        """Optimize pricing trajectory using scipy optimization.
        
        Args:
            current_price: Current ticket price
            base_demand: Baseline demand at current price
            price_elasticity: Price elasticity of demand
            constraints: Optimization constraints
        
        Returns:
            Optimization results dictionary
        """
        if constraints is None:
            constraints = OptimizationConstraints(
                min_annual_increase=self.config.min_annual_increase,
                max_annual_increase=self.config.max_annual_increase,
                inflation_floor=self.config.inflation_floor,
                max_churn_rate=self.config.max_acceptable_churn,
            )
        
        n_years = self.config.planning_years
        
        def objective(price_increases):
            """Objective: maximize total discounted revenue minus churn cost."""
            prices = [current_price]
            for inc in price_increases:
                prices.append(prices[-1] * (1 + inc))
            
            total_revenue = 0
            total_attendance = 0
            cumulative_retention = 1.0
            
            prev_price = current_price
            for year, price in enumerate(prices[1:], 1):
                # Calculate churn from price increase
                increase_pct = (price - prev_price) / prev_price
                churn = self._calculate_churn(increase_pct)
                cumulative_retention *= (1 - churn)
                
                # Calculate demand at this price (with elasticity)
                price_ratio = price / current_price
                demand_multiplier = price_ratio ** price_elasticity
                demand = int(base_demand * demand_multiplier * cumulative_retention)
                
                # Calculate discounted revenue
                revenue = price * demand
                discounted_revenue = revenue / ((1 + self.config.discount_rate) ** year)
                
                total_revenue += discounted_revenue
                total_attendance += demand
                
                prev_price = price
            
            # Combined objective (negative because we minimize)
            objective_value = -(
                self.config.revenue_weight * total_revenue / 1000000 +
                self.config.attendance_weight * total_attendance / 10000 +
                self.config.retention_weight * cumulative_retention * 100
            )
            
            return objective_value
        
        # Bounds: annual increase between min and max
        bounds = [
            (constraints.min_annual_increase, constraints.max_annual_increase)
            for _ in range(n_years)
        ]
        
        # Constraint: inflation floor
        def inflation_floor_constraint(price_increases):
            """Ensure minimum inflation adjustment."""
            return np.array([inc - constraints.inflation_floor for inc in price_increases])
        
        # Constraint: churn ceiling
        def churn_constraint(price_increases):
            """Ensure churn doesn't exceed maximum."""
            churn_rates = [self._calculate_churn(inc) for inc in price_increases]
            return np.array([constraints.max_churn_rate - cr for cr in churn_rates])
        
        scipy_constraints = [
            {"type": "ineq", "fun": churn_constraint},
        ]
        
        # Initial guess: inflation rate each year
        x0 = [self.config.inflation_floor] * n_years
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 1000, "disp": False}
        )
        
        # Build results
        optimal_increases = result.x
        optimal_prices = [current_price]
        for inc in optimal_increases:
            optimal_prices.append(optimal_prices[-1] * (1 + inc))
        
        # Calculate metrics for optimal trajectory
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
            "optimization_message": result.message,
        }
        
        return self.optimization_results
    
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
        
        Args:
            price_increase_pct: Annual price increase as decimal
        
        Returns:
            Expected churn rate
        """
        threshold = 0.05  # 5% threshold
        
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
