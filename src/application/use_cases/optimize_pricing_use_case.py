"""Use case for optimizing ticket pricing trajectories."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from src.domain.entities.pricing_trajectory import (
    PricingTrajectory,
    SeatingSection,
    PricingTier,
    OptimizationConstraints,
)
from src.domain.entities.season_ticket_holder import SeasonTicketHolder
from src.domain.services.price_optimization_service import PriceOptimizationService
from src.domain.services.churn_modeling_service import ChurnModelingService
from src.application.ports.ports import (
    PricingTrajectoryRepositoryPort,
    SeasonTicketHolderRepositoryPort,
)


@dataclass
class PricingScenario:
    """A pricing scenario for comparison."""
    
    name: str
    constraints: OptimizationConstraints
    trajectory: Optional[PricingTrajectory] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PricingOptimizationResult:
    """Result of pricing optimization."""
    
    section: str
    tier: str
    current_price: float
    optimal_trajectory: PricingTrajectory
    alternative_scenarios: List[PricingScenario]
    churn_analysis: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section": self.section,
            "tier": self.tier,
            "current_price": self.current_price,
            "optimal_trajectory": self.optimal_trajectory.to_dict(),
            "alternative_scenarios": [
                {
                    "name": s.name,
                    "metrics": s.metrics,
                    "trajectory": s.trajectory.to_dict() if s.trajectory else None,
                }
                for s in self.alternative_scenarios
            ],
            "churn_analysis": self.churn_analysis,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


class OptimizePricingUseCase:
    """Use case for optimizing multi-year ticket pricing.
    
    Orchestrates:
    1. Current pricing analysis
    2. Churn risk modeling
    3. Trajectory optimization with constraints
    4. Scenario comparison
    5. Recommendation generation
    """
    
    def __init__(
        self,
        trajectory_repository: PricingTrajectoryRepositoryPort,
        holder_repository: SeasonTicketHolderRepositoryPort,
        optimization_service: PriceOptimizationService,
        churn_service: ChurnModelingService,
    ):
        """Initialize use case.
        
        Args:
            trajectory_repository: Repository for pricing trajectories
            holder_repository: Repository for season ticket holders
            optimization_service: Service for price optimization
            churn_service: Service for churn modeling
        """
        self.trajectory_repo = trajectory_repository
        self.holder_repo = holder_repository
        self.optimization_service = optimization_service
        self.churn_service = churn_service
    
    def optimize_section_pricing(
        self,
        section: str,
        current_price: float,
        current_season: int,
        planning_years: int = 5,
        constraints: Optional[OptimizationConstraints] = None,
        method: str = "scipy"
    ) -> PricingOptimizationResult:
        """Optimize pricing for a seating section.
        
        Args:
            section: Seating section name
            current_price: Current ticket price
            current_season: Current season year
            planning_years: Number of years to plan
            constraints: Optimization constraints (uses defaults if None)
            method: Optimization method ("scipy" or "pulp")
        
        Returns:
            PricingOptimizationResult with optimal trajectory
        """
        # Determine section enum and tier
        section_enum = self._parse_section(section)
        tier = self._determine_tier(current_price)
        
        # Get current holders for this section
        active_holders = self.holder_repo.get_active_holders(current_season)
        section_holders = [
            h for h in active_holders
            if section.lower() in " ".join(h.seat_locations).lower()
        ]
        
        # Analyze churn risk
        churn_analysis = self._analyze_section_churn(section_holders)
        
        # Use default constraints if not provided
        if constraints is None:
            constraints = self._create_default_constraints(churn_analysis)
        
        # Create and optimize trajectory
        optimal_trajectory = self.optimization_service.create_pricing_trajectory(
            section=section_enum,
            current_price=current_price,
            current_season=current_season,
            planning_years=planning_years,
            constraints=constraints,
            method=method,
        )
        
        # Generate alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(
            section_enum=section_enum,
            current_price=current_price,
            current_season=current_season,
            planning_years=planning_years,
            base_constraints=constraints,
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            optimal_trajectory=optimal_trajectory,
            churn_analysis=churn_analysis,
            alternative_scenarios=alternative_scenarios,
        )
        
        # Save trajectory
        self.trajectory_repo.save(optimal_trajectory)
        
        return PricingOptimizationResult(
            section=section,
            tier=tier.name,
            current_price=current_price,
            optimal_trajectory=optimal_trajectory,
            alternative_scenarios=alternative_scenarios,
            churn_analysis=churn_analysis,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )
    
    def optimize_all_sections(
        self,
        pricing_config: Dict[str, float],
        current_season: int,
        planning_years: int = 5
    ) -> List[PricingOptimizationResult]:
        """Optimize pricing for all sections.
        
        Args:
            pricing_config: Dictionary mapping section names to current prices
            current_season: Current season year
            planning_years: Number of years to plan
        
        Returns:
            List of PricingOptimizationResults
        """
        results = []
        
        for section, price in pricing_config.items():
            try:
                result = self.optimize_section_pricing(
                    section=section,
                    current_price=price,
                    current_season=current_season,
                    planning_years=planning_years,
                )
                results.append(result)
            except Exception as e:
                print(f"Error optimizing {section}: {e}")
                continue
        
        return results
    
    def compare_pricing_strategies(
        self,
        section: str,
        current_price: float,
        current_season: int,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple pricing strategies.
        
        Args:
            section: Seating section name
            current_price: Current ticket price
            current_season: Current season year
            strategies: List of strategy configurations
        
        Returns:
            Comparison results
        """
        section_enum = self._parse_section(section)
        
        results = []
        for strategy in strategies:
            constraints = OptimizationConstraints(
                min_annual_increase=strategy.get("min_increase", 0.00),
                max_annual_increase=strategy.get("max_increase", 0.15),
                inflation_floor=strategy.get("inflation_floor", 0.03),
                max_churn_rate=strategy.get("max_churn", 0.10),
            )
            
            trajectory = self.optimization_service.create_pricing_trajectory(
                section=section_enum,
                current_price=current_price,
                current_season=current_season,
                constraints=constraints,
            )
            
            results.append({
                "name": strategy.get("name", "Unnamed"),
                "constraints": {
                    "max_increase": constraints.max_annual_increase,
                    "inflation_floor": constraints.inflation_floor,
                    "max_churn": constraints.max_churn_rate,
                },
                "total_revenue": trajectory.total_expected_revenue,
                "total_attendance": trajectory.total_expected_attendance,
                "retention_rate": trajectory.expected_retention_rate,
                "final_price": trajectory.yearly_prices[-1].adjusted_price
                    if trajectory.yearly_prices else current_price,
            })
        
        # Find best by different criteria
        best_revenue = max(results, key=lambda x: x["total_revenue"])
        best_attendance = max(results, key=lambda x: x["total_attendance"])
        best_retention = max(results, key=lambda x: x["retention_rate"])
        
        return {
            "strategies": results,
            "best_revenue": best_revenue["name"],
            "best_attendance": best_attendance["name"],
            "best_retention": best_retention["name"],
            "recommendation": self._select_best_strategy(results),
        }
    
    def simulate_price_change(
        self,
        section: str,
        current_price: float,
        proposed_increase_pct: float,
        current_season: int
    ) -> Dict[str, Any]:
        """Simulate the impact of a price change.
        
        Args:
            section: Seating section name
            current_price: Current ticket price
            proposed_increase_pct: Proposed percentage increase
            current_season: Current season year
        
        Returns:
            Simulation results
        """
        # Get holders for this section
        active_holders = self.holder_repo.get_active_holders(current_season)
        section_holders = [
            h for h in active_holders
            if section.lower() in " ".join(h.seat_locations).lower()
        ]
        
        # Calculate expected churn
        expected_churn = self.churn_service.estimate_churn_from_price_increase(
            proposed_increase_pct
        )
        
        # Calculate current metrics
        current_seats = sum(h.num_seats for h in section_holders)
        current_revenue = current_price * current_seats
        
        # Calculate post-change metrics
        retained_seats = int(current_seats * (1 - expected_churn))
        new_price = current_price * (1 + proposed_increase_pct)
        new_revenue = new_price * retained_seats
        
        # Revenue change
        revenue_change = new_revenue - current_revenue
        revenue_change_pct = (revenue_change / current_revenue) * 100 if current_revenue > 0 else 0
        
        return {
            "section": section,
            "current_price": current_price,
            "proposed_price": new_price,
            "price_increase_pct": proposed_increase_pct * 100,
            "current_seats": current_seats,
            "expected_churn_rate": expected_churn * 100,
            "expected_retained_seats": retained_seats,
            "expected_lost_seats": current_seats - retained_seats,
            "current_revenue": current_revenue,
            "projected_revenue": new_revenue,
            "revenue_change": revenue_change,
            "revenue_change_pct": revenue_change_pct,
            "recommendation": "Proceed" if revenue_change > 0 else "Reconsider",
        }
    
    def _parse_section(self, section: str) -> SeatingSection:
        """Parse section string to enum.
        
        Args:
            section: Section name string
        
        Returns:
            SeatingSection enum
        """
        section_lower = section.lower()
        
        if "club" in section_lower:
            return SeatingSection.CLUB
        elif "loge" in section_lower:
            return SeatingSection.LOGE
        elif "lower" in section_lower:
            return SeatingSection.LOWER_RESERVED
        elif "upper" in section_lower:
            return SeatingSection.UPPER_RESERVED
        elif "bleacher" in section_lower:
            return SeatingSection.BLEACHERS
        elif "student" in section_lower:
            return SeatingSection.STUDENT
        else:
            return SeatingSection.LOWER_RESERVED
    
    def _determine_tier(self, price: float) -> PricingTier:
        """Determine pricing tier from price.
        
        Args:
            price: Ticket price
        
        Returns:
            PricingTier enum
        """
        if price > 75:
            return PricingTier.PREMIUM
        elif price > 35:
            return PricingTier.STANDARD
        else:
            return PricingTier.VALUE
    
    def _analyze_section_churn(
        self,
        holders: List[SeasonTicketHolder]
    ) -> Dict[str, Any]:
        """Analyze churn risk for section holders.
        
        Args:
            holders: List of season ticket holders
        
        Returns:
            Churn analysis dictionary
        """
        if not holders:
            return {
                "total_holders": 0,
                "avg_churn_probability": 0.0,
                "at_risk_count": 0,
                "at_risk_revenue": 0.0,
            }
        
        # Get churn probabilities
        churn_probs = []
        at_risk_revenue = 0.0
        at_risk_count = 0
        
        for holder in holders:
            prob = holder.churn_probability or 0.0
            churn_probs.append(prob)
            
            if prob >= 0.5:
                at_risk_count += 1
                at_risk_revenue += holder.current_season_revenue or 0.0
        
        return {
            "total_holders": len(holders),
            "total_seats": sum(h.num_seats for h in holders),
            "avg_churn_probability": sum(churn_probs) / len(churn_probs),
            "max_churn_probability": max(churn_probs),
            "at_risk_count": at_risk_count,
            "at_risk_pct": at_risk_count / len(holders) * 100,
            "at_risk_revenue": at_risk_revenue,
            "total_revenue": sum(h.current_season_revenue or 0 for h in holders),
        }
    
    def _create_default_constraints(
        self,
        churn_analysis: Dict[str, Any]
    ) -> OptimizationConstraints:
        """Create default constraints based on churn analysis.
        
        Args:
            churn_analysis: Churn analysis dictionary
        
        Returns:
            OptimizationConstraints
        """
        # Adjust max churn based on current risk
        current_risk = churn_analysis.get("avg_churn_probability", 0.0)
        
        if current_risk > 0.3:
            max_churn = 0.05  # Conservative if already high risk
            max_increase = 0.08
        elif current_risk > 0.2:
            max_churn = 0.08
            max_increase = 0.10
        else:
            max_churn = 0.10
            max_increase = 0.15
        
        return OptimizationConstraints(
            min_annual_increase=0.00,
            max_annual_increase=max_increase,
            inflation_floor=0.03,
            max_churn_rate=max_churn,
            max_cumulative_churn=0.30,
        )
    
    def _generate_alternative_scenarios(
        self,
        section_enum: SeatingSection,
        current_price: float,
        current_season: int,
        planning_years: int,
        base_constraints: OptimizationConstraints,
    ) -> List[PricingScenario]:
        """Generate alternative pricing scenarios.
        
        Args:
            section_enum: Seating section
            current_price: Current price
            current_season: Current season
            planning_years: Planning horizon
            base_constraints: Base constraints
        
        Returns:
            List of PricingScenarios
        """
        scenarios = []
        
        # Conservative scenario
        conservative = OptimizationConstraints(
            min_annual_increase=0.00,
            max_annual_increase=0.08,
            inflation_floor=0.03,
            max_churn_rate=0.05,
        )
        
        traj = self.optimization_service.create_pricing_trajectory(
            section=section_enum,
            current_price=current_price,
            current_season=current_season,
            planning_years=planning_years,
            constraints=conservative,
        )
        
        scenarios.append(PricingScenario(
            name="Conservative",
            constraints=conservative,
            trajectory=traj,
            metrics={
                "total_revenue": traj.total_expected_revenue,
                "total_attendance": traj.total_expected_attendance,
                "retention_rate": traj.expected_retention_rate,
            },
        ))
        
        # Aggressive scenario
        aggressive = OptimizationConstraints(
            min_annual_increase=0.05,
            max_annual_increase=0.15,
            inflation_floor=0.05,
            max_churn_rate=0.12,
        )
        
        traj = self.optimization_service.create_pricing_trajectory(
            section=section_enum,
            current_price=current_price,
            current_season=current_season,
            planning_years=planning_years,
            constraints=aggressive,
        )
        
        scenarios.append(PricingScenario(
            name="Aggressive",
            constraints=aggressive,
            trajectory=traj,
            metrics={
                "total_revenue": traj.total_expected_revenue,
                "total_attendance": traj.total_expected_attendance,
                "retention_rate": traj.expected_retention_rate,
            },
        ))
        
        # Inflation-only scenario
        inflation_only = OptimizationConstraints(
            min_annual_increase=0.03,
            max_annual_increase=0.03,
            inflation_floor=0.03,
            max_churn_rate=0.05,
        )
        
        traj = self.optimization_service.create_pricing_trajectory(
            section=section_enum,
            current_price=current_price,
            current_season=current_season,
            planning_years=planning_years,
            constraints=inflation_only,
        )
        
        scenarios.append(PricingScenario(
            name="Inflation Only",
            constraints=inflation_only,
            trajectory=traj,
            metrics={
                "total_revenue": traj.total_expected_revenue,
                "total_attendance": traj.total_expected_attendance,
                "retention_rate": traj.expected_retention_rate,
            },
        ))
        
        return scenarios
    
    def _generate_recommendations(
        self,
        optimal_trajectory: PricingTrajectory,
        churn_analysis: Dict[str, Any],
        alternative_scenarios: List[PricingScenario],
    ) -> List[str]:
        """Generate pricing recommendations.
        
        Args:
            optimal_trajectory: Optimal trajectory
            churn_analysis: Churn analysis
            alternative_scenarios: Alternative scenarios
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check churn risk
        if churn_analysis.get("avg_churn_probability", 0) > 0.25:
            recommendations.append(
                "High churn risk detected. Consider customer retention initiatives "
                "before implementing price increases."
            )
        
        # Check optimal vs alternatives
        optimal_revenue = optimal_trajectory.total_expected_revenue
        
        for scenario in alternative_scenarios:
            if scenario.metrics.get("total_revenue", 0) > optimal_revenue * 1.1:
                recommendations.append(
                    f"The '{scenario.name}' scenario offers 10%+ higher revenue. "
                    f"Consider if the additional churn risk is acceptable."
                )
        
        # First year recommendation
        if optimal_trajectory.yearly_prices:
            first_year = optimal_trajectory.yearly_prices[0]
            increase_pct = (first_year.adjusted_price / optimal_trajectory.current_price - 1) * 100
            
            if increase_pct > 5:
                recommendations.append(
                    f"Recommended first-year increase of {increase_pct:.1f}% is above "
                    f"the 5% churn threshold. Monitor renewals closely."
                )
        
        # Retention focus
        if optimal_trajectory.expected_retention_rate < 0.85:
            recommendations.append(
                f"Expected 5-year retention of {optimal_trajectory.expected_retention_rate:.1%} "
                f"is below 85% target. Consider loyalty incentives for long-term holders."
            )
        
        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Optimal pricing trajectory balances revenue growth with customer retention. "
                "Implement with standard monitoring protocols."
            )
        
        return recommendations
    
    def _select_best_strategy(
        self,
        results: List[Dict[str, Any]]
    ) -> str:
        """Select best overall strategy.
        
        Args:
            results: Strategy comparison results
        
        Returns:
            Name of recommended strategy
        """
        # Score each strategy
        scored = []
        
        for r in results:
            # Normalize metrics (higher is better)
            revenue_score = r["total_revenue"] / max(x["total_revenue"] for x in results)
            retention_score = r["retention_rate"]
            
            # Weighted score (60% revenue, 40% retention)
            score = 0.6 * revenue_score + 0.4 * retention_score
            scored.append((r["name"], score))
        
        # Return highest scoring
        return max(scored, key=lambda x: x[1])[0]
