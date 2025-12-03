"""Pricing trajectory entity for multi-year ticket price optimization."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from decimal import Decimal
from enum import Enum


class PricingTier(Enum):
    """Ticket pricing tier."""
    PREMIUM = 1
    STANDARD = 2
    VALUE = 3


class SeatingSection(Enum):
    """Stadium seating sections."""
    CLUB = "Club"
    LOGE = "Loge"
    LOWER_RESERVED = "Lower Reserved"
    UPPER_RESERVED = "Upper Reserved"
    BLEACHERS = "Bleachers"
    STUDENT = "Student"


@dataclass
class YearlyPrice:
    """Price for a single year in the trajectory."""
    
    year: int
    season: str = ""
    base_price: Decimal = Decimal("0.00")
    adjusted_price: Decimal = Decimal("0.00")
    
    # Adjustments
    inflation_adjustment: Decimal = Decimal("0.00")
    demand_adjustment: Decimal = Decimal("0.00")
    competitive_adjustment: Decimal = Decimal("0.00")
    
    # Constraints
    min_price: Decimal = Decimal("0.00")
    max_price: Decimal = Decimal("1000.00")
    
    # Metrics
    expected_demand: int = 0
    expected_revenue: Decimal = Decimal("0.00")
    price_elasticity: float = -0.5
    
    def calculate_adjusted_price(self) -> Decimal:
        """Calculate final adjusted price."""
        total_adjustment = (
            self.inflation_adjustment 
            + self.demand_adjustment 
            + self.competitive_adjustment
        )
        self.adjusted_price = max(
            self.min_price,
            min(self.max_price, self.base_price + total_adjustment)
        )
        return self.adjusted_price
    
    def calculate_expected_revenue(self) -> Decimal:
        """Calculate expected revenue at this price."""
        self.expected_revenue = self.adjusted_price * self.expected_demand
        return self.expected_revenue


@dataclass
class PricingTrajectory:
    """Represents a multi-year pricing trajectory for ticket optimization."""
    
    trajectory_id: Optional[int] = None
    section: SeatingSection = SeatingSection.LOWER_RESERVED
    pricing_tier: PricingTier = PricingTier.STANDARD
    
    # Current state
    current_price: Decimal = Decimal("0.00")
    current_season: str = ""
    
    # Trajectory parameters
    planning_horizon_years: int = 5
    base_inflation_rate: float = 0.03
    
    # Yearly prices
    yearly_prices: List[YearlyPrice] = field(default_factory=list)
    
    # Constraints
    min_annual_increase: float = 0.00    # Minimum 0% increase per year
    max_annual_increase: float = 0.15    # Maximum 15% increase per year (churn constraint)
    inflation_floor: float = 0.03        # Minimum 3% inflation floor
    terminal_price_target: Optional[Decimal] = None
    
    # Churn constraints
    max_churn_rate: float = 0.10         # Maximum acceptable churn rate
    churn_elasticity: float = 2.0        # How much churn increases per price increase
    
    # Optimization results
    optimal_trajectory: List[Decimal] = field(default_factory=list)
    total_expected_revenue: Decimal = Decimal("0.00")
    total_expected_attendance: int = 0
    expected_retention_rate: float = 0.90
    
    # Model weights for ensemble prediction
    model_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize trajectory."""
        if not self.yearly_prices and self.current_price > 0:
            self._initialize_trajectory()
    
    def _initialize_trajectory(self):
        """Initialize yearly prices based on current price."""
        self.yearly_prices = []
        
        for year in range(self.planning_horizon_years):
            year_num = year + 1
            season = self._get_season_code(year_num)
            
            # Calculate base price with inflation
            base_price = self.current_price * Decimal(str(
                (1 + self.base_inflation_rate) ** year_num
            ))
            
            yearly_price = YearlyPrice(
                year=year_num,
                season=season,
                base_price=base_price,
                min_price=self.current_price * Decimal("0.90"),
                max_price=self.current_price * Decimal(str(
                    (1 + self.max_annual_increase) ** year_num
                )),
            )
            
            self.yearly_prices.append(yearly_price)
    
    def _get_season_code(self, years_ahead: int) -> str:
        """Generate season code for future year."""
        # Parse current season (e.g., "25-26" or "FB25")
        if "-" in self.current_season:
            start_year = int(self.current_season.split("-")[0])
            future_year = start_year + years_ahead
            return f"{future_year}-{future_year + 1}"
        else:
            # Handle format like "FB25"
            sport = self.current_season[:2]
            year = int(self.current_season[2:])
            return f"{sport}{year + years_ahead}"
    
    def calculate_churn_from_price_increase(self, price_increase_pct: float) -> float:
        """Calculate expected churn rate from price increase."""
        # Churn increases exponentially with price increases above threshold
        threshold = 0.05  # 5% threshold before churn kicks in
        
        if price_increase_pct <= threshold:
            return 0.02  # Baseline churn
        
        excess_increase = price_increase_pct - threshold
        churn = 0.02 + (excess_increase * self.churn_elasticity)
        
        return min(churn, 0.30)  # Cap at 30% churn
    
    def validate_trajectory(self) -> Dict[str, List[str]]:
        """Validate trajectory against constraints."""
        errors = []
        warnings = []
        
        prev_price = self.current_price
        
        for yp in self.yearly_prices:
            # Check annual increase constraint
            if prev_price > 0:
                increase_pct = float((yp.adjusted_price - prev_price) / prev_price)
                
                if increase_pct > self.max_annual_increase:
                    errors.append(
                        f"Year {yp.year}: Annual increase {increase_pct:.1%} exceeds "
                        f"maximum {self.max_annual_increase:.1%}"
                    )
                
                if increase_pct < self.inflation_floor:
                    warnings.append(
                        f"Year {yp.year}: Annual increase {increase_pct:.1%} below "
                        f"inflation floor {self.inflation_floor:.1%}"
                    )
                
                # Check churn constraint
                expected_churn = self.calculate_churn_from_price_increase(increase_pct)
                if expected_churn > self.max_churn_rate:
                    warnings.append(
                        f"Year {yp.year}: Expected churn {expected_churn:.1%} exceeds "
                        f"target {self.max_churn_rate:.1%}"
                    )
            
            prev_price = yp.adjusted_price
        
        # Check terminal constraint
        if self.terminal_price_target and self.yearly_prices:
            final_price = self.yearly_prices[-1].adjusted_price
            if final_price < self.terminal_price_target:
                warnings.append(
                    f"Final price {final_price} below terminal target {self.terminal_price_target}"
                )
        
        return {"errors": errors, "warnings": warnings}
    
    def calculate_total_metrics(self):
        """Calculate total trajectory metrics."""
        self.total_expected_revenue = Decimal("0.00")
        self.total_expected_attendance = 0
        
        cumulative_retention = 1.0
        prev_price = self.current_price
        
        for yp in self.yearly_prices:
            # Calculate churn impact
            if prev_price > 0:
                increase_pct = float((yp.adjusted_price - prev_price) / prev_price)
                churn = self.calculate_churn_from_price_increase(increase_pct)
                cumulative_retention *= (1 - churn)
            
            # Adjust demand for churn
            adjusted_demand = int(yp.expected_demand * cumulative_retention)
            yp.expected_demand = adjusted_demand
            
            # Calculate revenue
            yp.calculate_expected_revenue()
            self.total_expected_revenue += yp.expected_revenue
            self.total_expected_attendance += adjusted_demand
            
            prev_price = yp.adjusted_price
        
        self.expected_retention_rate = cumulative_retention
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "trajectory_id": self.trajectory_id,
            "section": self.section.value,
            "pricing_tier": self.pricing_tier.value,
            "current_price": float(self.current_price),
            "current_season": self.current_season,
            "planning_horizon_years": self.planning_horizon_years,
            "base_inflation_rate": self.base_inflation_rate,
            "min_annual_increase": self.min_annual_increase,
            "max_annual_increase": self.max_annual_increase,
            "inflation_floor": self.inflation_floor,
            "max_churn_rate": self.max_churn_rate,
            "optimal_trajectory": [float(p) for p in self.optimal_trajectory],
            "total_expected_revenue": float(self.total_expected_revenue),
            "total_expected_attendance": self.total_expected_attendance,
            "expected_retention_rate": self.expected_retention_rate,
            "yearly_prices": [
                {
                    "year": yp.year,
                    "season": yp.season,
                    "base_price": float(yp.base_price),
                    "adjusted_price": float(yp.adjusted_price),
                    "expected_demand": yp.expected_demand,
                    "expected_revenue": float(yp.expected_revenue),
                }
                for yp in self.yearly_prices
            ],
        }


@dataclass
class OptimizationConstraints:
    """Constraints for pricing optimization."""
    
    # Annual increase constraints
    min_annual_increase: float = 0.00
    max_annual_increase: float = 0.15
    inflation_floor: float = 0.03
    
    # Churn constraints
    max_churn_rate: float = 0.10
    max_cumulative_churn: float = 0.30
    
    # Revenue constraints
    min_annual_revenue: Decimal = Decimal("0.00")
    min_total_revenue: Decimal = Decimal("0.00")
    
    # Terminal constraints
    terminal_price_target: Optional[Decimal] = None
    terminal_revenue_target: Optional[Decimal] = None
    
    # Attendance constraints
    min_attendance: int = 0
    target_attendance: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "min_annual_increase": self.min_annual_increase,
            "max_annual_increase": self.max_annual_increase,
            "inflation_floor": self.inflation_floor,
            "max_churn_rate": self.max_churn_rate,
            "max_cumulative_churn": self.max_cumulative_churn,
            "min_annual_revenue": float(self.min_annual_revenue),
            "min_total_revenue": float(self.min_total_revenue),
            "terminal_price_target": float(self.terminal_price_target) if self.terminal_price_target else None,
            "terminal_revenue_target": float(self.terminal_revenue_target) if self.terminal_revenue_target else None,
            "min_attendance": self.min_attendance,
            "target_attendance": self.target_attendance,
        }
