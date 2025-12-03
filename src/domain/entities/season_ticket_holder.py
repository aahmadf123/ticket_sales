"""Season ticket holder entity for churn modeling."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict
from enum import Enum
from decimal import Decimal


class HolderStatus(Enum):
    """Status of season ticket holder."""
    ACTIVE = "active"
    CHURNED = "churned"
    AT_RISK = "at_risk"
    NEW = "new"


class TenureCategory(Enum):
    """Tenure category based on years as holder."""
    NEW = "new"           # 0-1 years
    DEVELOPING = "developing"  # 2-3 years
    ESTABLISHED = "established"  # 4-7 years
    LOYAL = "loyal"       # 8+ years


@dataclass
class SeasonTicketHolder:
    """Represents a season ticket holder for churn analysis."""
    
    holder_id: Optional[int] = None
    account_id: str = ""
    holder_name: str = ""
    
    # Tenure information
    first_season: str = ""
    seasons_held: int = 0
    tenure_category: TenureCategory = TenureCategory.NEW
    
    # Current season information
    current_season: str = ""
    status: HolderStatus = HolderStatus.ACTIVE
    
    # Ticket information
    num_seats: int = 0
    seat_locations: List[str] = field(default_factory=list)
    seat_value_total: Decimal = Decimal("0.00")
    avg_seat_value: Decimal = Decimal("0.00")
    
    # Financial metrics
    total_revenue_lifetime: Decimal = Decimal("0.00")
    current_season_revenue: Decimal = Decimal("0.00")
    previous_season_revenue: Decimal = Decimal("0.00")
    revenue_change_pct: float = 0.0
    
    # Engagement metrics
    games_attended_current: int = 0
    games_attended_previous: int = 0
    attendance_rate: float = 0.0
    avg_scan_time_minutes: float = 0.0
    
    # Upgrade/downgrade history
    upgrades_count: int = 0
    downgrades_count: int = 0
    seats_added_count: int = 0
    seats_removed_count: int = 0
    
    # Churn indicators
    churn_risk_score: float = 0.0
    churn_probability: float = 0.0
    retention_probability: float = 1.0
    
    # Contact information (for marketing)
    email: str = ""
    phone: str = ""
    preferred_contact: str = "email"
    
    def __post_init__(self):
        """Calculate derived fields."""
        self._calculate_tenure_category()
        self._calculate_avg_seat_value()
        self._calculate_revenue_change()
    
    def _calculate_tenure_category(self):
        """Determine tenure category based on seasons held."""
        if self.seasons_held <= 1:
            self.tenure_category = TenureCategory.NEW
        elif self.seasons_held <= 3:
            self.tenure_category = TenureCategory.DEVELOPING
        elif self.seasons_held <= 7:
            self.tenure_category = TenureCategory.ESTABLISHED
        else:
            self.tenure_category = TenureCategory.LOYAL
    
    def _calculate_avg_seat_value(self):
        """Calculate average seat value."""
        if self.num_seats > 0:
            self.avg_seat_value = self.seat_value_total / self.num_seats
    
    def _calculate_revenue_change(self):
        """Calculate revenue change percentage."""
        if self.previous_season_revenue > 0:
            self.revenue_change_pct = float(
                (self.current_season_revenue - self.previous_season_revenue) 
                / self.previous_season_revenue * 100
            )
    
    def calculate_churn_features(self) -> Dict[str, float]:
        """Calculate features for churn prediction model."""
        return {
            "seasons_held": self.seasons_held,
            "tenure_category_new": 1 if self.tenure_category == TenureCategory.NEW else 0,
            "tenure_category_developing": 1 if self.tenure_category == TenureCategory.DEVELOPING else 0,
            "tenure_category_established": 1 if self.tenure_category == TenureCategory.ESTABLISHED else 0,
            "tenure_category_loyal": 1 if self.tenure_category == TenureCategory.LOYAL else 0,
            "num_seats": self.num_seats,
            "avg_seat_value": float(self.avg_seat_value),
            "total_revenue_lifetime": float(self.total_revenue_lifetime),
            "revenue_change_pct": self.revenue_change_pct,
            "attendance_rate": self.attendance_rate,
            "games_attended_current": self.games_attended_current,
            "upgrades_count": self.upgrades_count,
            "downgrades_count": self.downgrades_count,
            "seats_added_count": self.seats_added_count,
            "seats_removed_count": self.seats_removed_count,
            "net_seat_change": self.seats_added_count - self.seats_removed_count,
            "upgrade_downgrade_ratio": self.upgrades_count / max(1, self.downgrades_count),
        }
    
    def calculate_retention_value(self, discount_rate: float = 0.1) -> float:
        """Calculate expected lifetime value if retained."""
        # Simple NPV calculation
        expected_annual_revenue = float(self.current_season_revenue)
        expected_years = 3.0  # Average expected additional years based on tenure
        
        if self.tenure_category == TenureCategory.NEW:
            expected_years = 2.0
        elif self.tenure_category == TenureCategory.DEVELOPING:
            expected_years = 3.0
        elif self.tenure_category == TenureCategory.ESTABLISHED:
            expected_years = 5.0
        else:  # LOYAL
            expected_years = 7.0
        
        # Calculate NPV
        npv = 0.0
        for year in range(1, int(expected_years) + 1):
            npv += expected_annual_revenue / ((1 + discount_rate) ** year)
        
        return npv * self.retention_probability
    
    def is_at_risk(self) -> bool:
        """Determine if holder is at risk of churning."""
        risk_indicators = 0
        
        if self.revenue_change_pct < -20:
            risk_indicators += 1
        if self.attendance_rate < 0.5:
            risk_indicators += 1
        if self.downgrades_count > self.upgrades_count:
            risk_indicators += 1
        if self.seats_removed_count > 0:
            risk_indicators += 1
        if self.tenure_category == TenureCategory.NEW:
            risk_indicators += 0.5
        
        return risk_indicators >= 2
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "holder_id": self.holder_id,
            "account_id": self.account_id,
            "holder_name": self.holder_name,
            "first_season": self.first_season,
            "seasons_held": self.seasons_held,
            "tenure_category": self.tenure_category.value,
            "current_season": self.current_season,
            "status": self.status.value,
            "num_seats": self.num_seats,
            "seat_value_total": float(self.seat_value_total),
            "avg_seat_value": float(self.avg_seat_value),
            "total_revenue_lifetime": float(self.total_revenue_lifetime),
            "current_season_revenue": float(self.current_season_revenue),
            "revenue_change_pct": self.revenue_change_pct,
            "attendance_rate": self.attendance_rate,
            "churn_risk_score": self.churn_risk_score,
            "churn_probability": self.churn_probability,
            "retention_probability": self.retention_probability,
        }
