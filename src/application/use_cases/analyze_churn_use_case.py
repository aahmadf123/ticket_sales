"""Use case for analyzing season ticket holder churn."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd

from src.domain.entities.season_ticket_holder import (
    SeasonTicketHolder,
    HolderStatus,
    TenureCategory,
)
from src.domain.services.churn_modeling_service import ChurnModelingService
from src.application.ports.ports import (
    SeasonTicketHolderRepositoryPort,
    NotificationPort,
)


@dataclass
class HolderAnalysis:
    """Analysis result for a single holder."""
    
    holder_id: str
    account_id: str
    holder_name: str
    tenure_years: int
    tenure_category: str
    current_status: str
    churn_probability: float
    retention_probability: float
    risk_level: str
    retention_value: float
    risk_factors: List[Dict[str, Any]]
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "holder_id": self.holder_id,
            "account_id": self.account_id,
            "holder_name": self.holder_name,
            "tenure_years": self.tenure_years,
            "tenure_category": self.tenure_category,
            "current_status": self.current_status,
            "churn_probability": self.churn_probability,
            "retention_probability": self.retention_probability,
            "risk_level": self.risk_level,
            "retention_value": self.retention_value,
            "risk_factors": self.risk_factors,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class PortfolioAnalysis:
    """Analysis of entire holder portfolio."""
    
    season: int
    total_holders: int
    total_seats: int
    total_revenue: float
    risk_distribution: Dict[str, int]
    expected_churn_count: int
    expected_churn_rate: float
    at_risk_revenue: float
    total_retention_value: float
    tenure_distribution: Dict[str, int]
    top_risk_holders: List[HolderAnalysis]
    segment_analysis: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "season": self.season,
            "total_holders": self.total_holders,
            "total_seats": self.total_seats,
            "total_revenue": self.total_revenue,
            "risk_distribution": self.risk_distribution,
            "expected_churn_count": self.expected_churn_count,
            "expected_churn_rate": self.expected_churn_rate,
            "at_risk_revenue": self.at_risk_revenue,
            "total_retention_value": self.total_retention_value,
            "tenure_distribution": self.tenure_distribution,
            "top_risk_holders": [h.to_dict() for h in self.top_risk_holders],
            "segment_analysis": self.segment_analysis,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


class AnalyzeChurnUseCase:
    """Use case for analyzing season ticket holder churn risk.
    
    Orchestrates:
    1. Individual holder analysis
    2. Portfolio-wide risk assessment
    3. Segment analysis
    4. Retention recommendations
    5. Alert generation
    """
    
    def __init__(
        self,
        holder_repository: SeasonTicketHolderRepositoryPort,
        churn_service: ChurnModelingService,
        notification_port: Optional[NotificationPort] = None,
    ):
        """Initialize use case.
        
        Args:
            holder_repository: Repository for season ticket holders
            churn_service: Service for churn modeling
            notification_port: Optional notification service
        """
        self.holder_repo = holder_repository
        self.churn_service = churn_service
        self.notification_port = notification_port
    
    def analyze_holder(self, holder_id: str) -> HolderAnalysis:
        """Analyze churn risk for a single holder.
        
        Args:
            holder_id: ID of the holder
        
        Returns:
            HolderAnalysis with risk assessment
        
        Raises:
            ValueError: If holder not found
        """
        holder = self.holder_repo.get_by_id(holder_id)
        if holder is None:
            raise ValueError(f"Holder not found: {holder_id}")
        
        # Get analysis from churn service
        analysis = self.churn_service.analyze_holder(holder)
        
        return HolderAnalysis(
            holder_id=holder.holder_id,
            account_id=holder.account_id,
            holder_name=holder.holder_name,
            tenure_years=holder.seasons_held,
            tenure_category=holder.tenure_category.name,
            current_status=holder.status.name,
            churn_probability=analysis["churn_probability"],
            retention_probability=analysis["retention_probability"],
            risk_level=analysis["risk_level"],
            retention_value=analysis["retention_value"],
            risk_factors=analysis["risk_factors"],
            recommended_actions=analysis["recommended_actions"],
        )
    
    def analyze_portfolio(self, season: int) -> PortfolioAnalysis:
        """Analyze entire holder portfolio for a season.
        
        Args:
            season: Season year to analyze
        
        Returns:
            PortfolioAnalysis with comprehensive metrics
        """
        # Get all active holders
        holders = self.holder_repo.get_active_holders(season)
        
        if not holders:
            return self._empty_portfolio_analysis(season)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.churn_service.calculate_portfolio_metrics(holders)
        
        # Analyze individual holders
        holder_analyses = []
        for holder in holders:
            try:
                analysis = self.churn_service.analyze_holder(holder)
                holder_analyses.append((holder, analysis))
            except Exception as e:
                print(f"Error analyzing holder {holder.holder_id}: {e}")
                continue
        
        # Get top risk holders
        top_risk = sorted(
            holder_analyses,
            key=lambda x: x[1]["churn_probability"],
            reverse=True
        )[:10]
        
        top_risk_analyses = [
            HolderAnalysis(
                holder_id=h.holder_id,
                account_id=h.account_id,
                holder_name=h.holder_name,
                tenure_years=h.seasons_held,
                tenure_category=h.tenure_category.name,
                current_status=h.status.name,
                churn_probability=a["churn_probability"],
                retention_probability=a["retention_probability"],
                risk_level=a["risk_level"],
                retention_value=a["retention_value"],
                risk_factors=a["risk_factors"],
                recommended_actions=a["recommended_actions"],
            )
            for h, a in top_risk
        ]
        
        # Calculate tenure distribution
        tenure_dist = self._calculate_tenure_distribution(holders)
        
        # Segment analysis
        segment_analysis = self._analyze_segments(holders, holder_analyses)
        
        # Generate recommendations
        recommendations = self._generate_portfolio_recommendations(
            portfolio_metrics, segment_analysis, top_risk_analyses
        )
        
        return PortfolioAnalysis(
            season=season,
            total_holders=portfolio_metrics["total_holders"],
            total_seats=sum(h.num_seats for h in holders),
            total_revenue=sum(h.current_season_revenue or 0 for h in holders),
            risk_distribution=portfolio_metrics["risk_distribution"],
            expected_churn_count=portfolio_metrics["expected_churn_count"],
            expected_churn_rate=portfolio_metrics["expected_churn_rate"],
            at_risk_revenue=portfolio_metrics["at_risk_revenue"],
            total_retention_value=portfolio_metrics["total_retention_value"],
            tenure_distribution=tenure_dist,
            top_risk_holders=top_risk_analyses,
            segment_analysis=segment_analysis,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )
    
    def identify_at_risk_holders(
        self,
        season: int,
        threshold: float = 0.5
    ) -> List[HolderAnalysis]:
        """Identify holders at high churn risk.
        
        Args:
            season: Season year
            threshold: Churn probability threshold
        
        Returns:
            List of HolderAnalysis for at-risk holders
        """
        at_risk = self.holder_repo.get_at_risk_holders(threshold)
        
        analyses = []
        for holder in at_risk:
            try:
                analysis = self.analyze_holder(holder.holder_id)
                analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing holder {holder.holder_id}: {e}")
                continue
        
        return sorted(analyses, key=lambda x: x.churn_probability, reverse=True)
    
    def train_churn_model(
        self,
        historical_holders: Optional[List[SeasonTicketHolder]] = None,
        include_churned: bool = True
    ) -> Dict[str, Any]:
        """Train or retrain the churn prediction model.
        
        Args:
            historical_holders: Historical holder data with churn outcomes
                               If None, loads from repository
            include_churned: Whether to include churned holders
        
        Returns:
            Training metrics dictionary
        """
        if historical_holders is None:
            # Load active and churned holders from multiple seasons
            historical_holders = []
            
            for season in range(2020, 2025):
                active = self.holder_repo.get_active_holders(season)
                historical_holders.extend(active)
                
                if include_churned:
                    churned = self.holder_repo.get_churned_holders(season)
                    historical_holders.extend(churned)
        
        if len(historical_holders) < 50:
            raise ValueError(
                f"Insufficient training data: {len(historical_holders)} holders. "
                f"Need at least 50 holders."
            )
        
        # Prepare features and labels
        features_list = []
        labels = []
        
        for holder in historical_holders:
            features = holder.calculate_churn_features()
            features_list.append(features)
            
            # Label: 1 if churned, 0 if active
            label = 1 if holder.status == HolderStatus.CHURNED else 0
            labels.append(label)
        
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels)
        
        # Train model
        training_metrics = self.churn_service.train(features_df, labels_series)
        
        return training_metrics
    
    def send_risk_alerts(
        self,
        season: int,
        threshold: float = 0.7,
        recipients: List[str] = None
    ) -> Dict[str, Any]:
        """Send alerts for high-risk holders.
        
        Args:
            season: Season year
            threshold: Risk threshold for alerts
            recipients: Email recipients
        
        Returns:
            Alert status dictionary
        """
        if self.notification_port is None:
            return {"error": "Notification port not configured"}
        
        if recipients is None:
            recipients = []
        
        # Get high-risk holders
        high_risk = self.identify_at_risk_holders(season, threshold)
        
        alerts_sent = 0
        for holder_analysis in high_risk:
            holder = self.holder_repo.get_by_id(holder_analysis.holder_id)
            
            if holder and self.notification_port.send_churn_alert(holder, recipients):
                alerts_sent += 1
        
        return {
            "high_risk_count": len(high_risk),
            "alerts_sent": alerts_sent,
            "recipients": recipients,
        }
    
    def get_retention_recommendations(
        self,
        holder_id: str
    ) -> Dict[str, Any]:
        """Get detailed retention recommendations for a holder.
        
        Args:
            holder_id: Holder ID
        
        Returns:
            Detailed recommendations
        """
        analysis = self.analyze_holder(holder_id)
        holder = self.holder_repo.get_by_id(holder_id)
        
        recommendations = {
            "holder": {
                "id": holder_id,
                "name": analysis.holder_name,
                "tenure": analysis.tenure_years,
                "risk_level": analysis.risk_level,
            },
            "immediate_actions": [],
            "long_term_strategies": [],
            "estimated_impact": {},
        }
        
        # Generate immediate actions based on risk level
        if analysis.risk_level == "high_risk":
            recommendations["immediate_actions"] = [
                "Schedule personal outreach call within 48 hours",
                "Offer flexible payment plan options",
                "Provide exclusive loyalty benefit or discount",
                "Assign dedicated account manager",
            ]
        elif analysis.risk_level == "medium_risk":
            recommendations["immediate_actions"] = [
                "Send personalized renewal reminder with benefits summary",
                "Offer early renewal incentive",
                "Invite to exclusive holder event",
            ]
        else:
            recommendations["immediate_actions"] = [
                "Include in regular communication cadence",
                "Recognize loyalty in upcoming communications",
            ]
        
        # Long-term strategies based on risk factors
        for factor in analysis.risk_factors:
            factor_name = factor.get("factor", "")
            
            if "attendance" in factor_name.lower():
                recommendations["long_term_strategies"].append(
                    "Implement seat relocation program to improve game experience"
                )
            elif "revenue" in factor_name.lower():
                recommendations["long_term_strategies"].append(
                    "Develop tiered pricing options for budget flexibility"
                )
            elif "tenure" in factor_name.lower():
                recommendations["long_term_strategies"].append(
                    "Create onboarding program with new holder mentorship"
                )
        
        # Estimated impact
        if analysis.churn_probability > 0.5:
            retention_lift = 0.20  # Assume 20% reduction in churn with intervention
        else:
            retention_lift = 0.10
        
        recommendations["estimated_impact"] = {
            "current_churn_probability": analysis.churn_probability,
            "estimated_post_intervention": analysis.churn_probability * (1 - retention_lift),
            "retention_value_at_risk": analysis.retention_value,
            "potential_value_saved": analysis.retention_value * retention_lift,
        }
        
        return recommendations
    
    def _empty_portfolio_analysis(self, season: int) -> PortfolioAnalysis:
        """Create empty portfolio analysis.
        
        Args:
            season: Season year
        
        Returns:
            Empty PortfolioAnalysis
        """
        return PortfolioAnalysis(
            season=season,
            total_holders=0,
            total_seats=0,
            total_revenue=0.0,
            risk_distribution={"safe": 0, "low_risk": 0, "medium_risk": 0, "high_risk": 0},
            expected_churn_count=0,
            expected_churn_rate=0.0,
            at_risk_revenue=0.0,
            total_retention_value=0.0,
            tenure_distribution={},
            top_risk_holders=[],
            segment_analysis={},
            recommendations=["No active holders found for this season"],
            generated_at=datetime.now(),
        )
    
    def _calculate_tenure_distribution(
        self,
        holders: List[SeasonTicketHolder]
    ) -> Dict[str, int]:
        """Calculate tenure category distribution.
        
        Args:
            holders: List of holders
        
        Returns:
            Dictionary of tenure category counts
        """
        distribution = {
            "NEW": 0,
            "DEVELOPING": 0,
            "ESTABLISHED": 0,
            "LOYAL": 0,
        }
        
        for holder in holders:
            category = holder.tenure_category.name
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def _analyze_segments(
        self,
        holders: List[SeasonTicketHolder],
        holder_analyses: List[tuple]
    ) -> Dict[str, Any]:
        """Analyze risk by segments.
        
        Args:
            holders: List of holders
            holder_analyses: List of (holder, analysis) tuples
        
        Returns:
            Segment analysis dictionary
        """
        # By tenure
        tenure_risk = {}
        for holder, analysis in holder_analyses:
            category = holder.tenure_category.name
            if category not in tenure_risk:
                tenure_risk[category] = []
            tenure_risk[category].append(analysis["churn_probability"])
        
        tenure_avg_risk = {
            cat: sum(probs) / len(probs) if probs else 0
            for cat, probs in tenure_risk.items()
        }
        
        # By seat count
        seat_segments = {"1-2 seats": [], "3-4 seats": [], "5+ seats": []}
        for holder, analysis in holder_analyses:
            if holder.num_seats <= 2:
                seat_segments["1-2 seats"].append(analysis["churn_probability"])
            elif holder.num_seats <= 4:
                seat_segments["3-4 seats"].append(analysis["churn_probability"])
            else:
                seat_segments["5+ seats"].append(analysis["churn_probability"])
        
        seat_avg_risk = {
            seg: sum(probs) / len(probs) if probs else 0
            for seg, probs in seat_segments.items()
        }
        
        # By revenue
        revenue_segments = {"<$500": [], "$500-$1000": [], ">$1000": []}
        for holder, analysis in holder_analyses:
            revenue = holder.current_season_revenue or 0
            if revenue < 500:
                revenue_segments["<$500"].append(analysis["churn_probability"])
            elif revenue < 1000:
                revenue_segments["$500-$1000"].append(analysis["churn_probability"])
            else:
                revenue_segments[">$1000"].append(analysis["churn_probability"])
        
        revenue_avg_risk = {
            seg: sum(probs) / len(probs) if probs else 0
            for seg, probs in revenue_segments.items()
        }
        
        return {
            "by_tenure": tenure_avg_risk,
            "by_seat_count": seat_avg_risk,
            "by_revenue": revenue_avg_risk,
            "highest_risk_segment": max(tenure_avg_risk, key=tenure_avg_risk.get)
                if tenure_avg_risk else None,
        }
    
    def _generate_portfolio_recommendations(
        self,
        portfolio_metrics: Dict[str, Any],
        segment_analysis: Dict[str, Any],
        top_risk: List[HolderAnalysis]
    ) -> List[str]:
        """Generate portfolio-level recommendations.
        
        Args:
            portfolio_metrics: Portfolio metrics
            segment_analysis: Segment analysis
            top_risk: Top risk holders
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Overall churn rate
        churn_rate = portfolio_metrics.get("expected_churn_rate", 0)
        if churn_rate > 0.15:
            recommendations.append(
                f"Expected churn rate of {churn_rate:.1%} exceeds 15% threshold. "
                f"Implement comprehensive retention program immediately."
            )
        elif churn_rate > 0.10:
            recommendations.append(
                f"Expected churn rate of {churn_rate:.1%} is elevated. "
                f"Focus on proactive outreach to medium-risk holders."
            )
        
        # High-risk segment
        highest_risk = segment_analysis.get("highest_risk_segment")
        if highest_risk:
            recommendations.append(
                f"Highest risk segment: {highest_risk}. "
                f"Develop targeted retention strategies for this group."
            )
        
        # New holder focus
        tenure_risk = segment_analysis.get("by_tenure", {})
        if tenure_risk.get("NEW", 0) > 0.3:
            recommendations.append(
                "New holders show high churn risk. "
                "Strengthen onboarding program and first-year engagement."
            )
        
        # Top risk holders
        if top_risk:
            high_value_at_risk = [h for h in top_risk if h.retention_value > 5000]
            if high_value_at_risk:
                recommendations.append(
                    f"{len(high_value_at_risk)} high-value holders at risk. "
                    f"Assign dedicated account managers for personal outreach."
                )
        
        # At-risk revenue
        at_risk_rev = portfolio_metrics.get("at_risk_revenue", 0)
        if at_risk_rev > 50000:
            recommendations.append(
                f"${at_risk_rev:,.0f} in revenue at risk from potential churn. "
                f"Prioritize retention initiatives for ROI."
            )
        
        if not recommendations:
            recommendations.append(
                "Portfolio health is stable. Maintain regular engagement cadence."
            )
        
        return recommendations
