"""Prediction entities for attendance forecasting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    T_7 = "T-7"       # 7 days before
    T_72 = "T-72"     # 72 hours before
    T_24 = "T-24"     # 24 hours before
    GAME_DAY = "game_day"


class UncertaintyLevel(Enum):
    """Level of uncertainty in prediction."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PredictionInterval:
    """Represents a confidence interval for a prediction."""
    lower: int
    upper: int
    confidence_level: float = 0.80
    
    @property
    def width(self) -> int:
        """Calculate interval width."""
        return self.upper - self.lower
    
    @property
    def midpoint(self) -> float:
        """Calculate interval midpoint."""
        return (self.lower + self.upper) / 2
    
    def contains(self, value: int) -> bool:
        """Check if value falls within interval."""
        return self.lower <= value <= self.upper


@dataclass
class Prediction:
    """Represents an attendance prediction."""
    
    prediction_id: Optional[int] = None
    game_id: int = 0
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    horizon: PredictionHorizon = PredictionHorizon.T_7
    days_before_game: int = 7
    model_version: str = ""
    
    # Point prediction
    predicted_attendance: int = 0
    
    # Confidence interval
    confidence_interval: Optional[PredictionInterval] = None
    
    # Individual model predictions
    xgboost_prediction: Optional[int] = None
    random_forest_prediction: Optional[int] = None
    bayesian_ridge_prediction: Optional[int] = None
    prophet_prediction: Optional[int] = None
    
    # Ensemble weights used
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    
    # Features used
    features_used: Dict[str, float] = field(default_factory=dict)
    
    # Uncertainty metrics
    prediction_std: float = 0.0
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MODERATE
    
    # Scenario predictions (for TBD kickoff times)
    scenario_predictions: List[Dict] = field(default_factory=list)
    
    # Notes
    notes: str = ""
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.confidence_interval is None:
            # Create default confidence interval
            uncertainty = int(self.prediction_std * 1.96)
            self.confidence_interval = PredictionInterval(
                lower=max(0, self.predicted_attendance - uncertainty),
                upper=self.predicted_attendance + uncertainty,
                confidence_level=0.80
            )
        
        self._determine_uncertainty_level()
    
    def _determine_uncertainty_level(self):
        """Determine uncertainty level based on interval width."""
        if self.confidence_interval is None:
            self.uncertainty_level = UncertaintyLevel.HIGH
            return
        
        width_ratio = self.confidence_interval.width / max(1, self.predicted_attendance)
        
        if width_ratio < 0.10:
            self.uncertainty_level = UncertaintyLevel.LOW
        elif width_ratio < 0.20:
            self.uncertainty_level = UncertaintyLevel.MODERATE
        elif width_ratio < 0.35:
            self.uncertainty_level = UncertaintyLevel.HIGH
        else:
            self.uncertainty_level = UncertaintyLevel.VERY_HIGH
    
    def get_model_spread(self) -> int:
        """Calculate spread between individual model predictions."""
        predictions = []
        if self.xgboost_prediction is not None:
            predictions.append(self.xgboost_prediction)
        if self.random_forest_prediction is not None:
            predictions.append(self.random_forest_prediction)
        if self.bayesian_ridge_prediction is not None:
            predictions.append(self.bayesian_ridge_prediction)
        if self.prophet_prediction is not None:
            predictions.append(self.prophet_prediction)
        
        if len(predictions) < 2:
            return 0
        
        return max(predictions) - min(predictions)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage/transmission."""
        return {
            "prediction_id": self.prediction_id,
            "game_id": self.game_id,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "horizon": self.horizon.value,
            "days_before_game": self.days_before_game,
            "model_version": self.model_version,
            "predicted_attendance": self.predicted_attendance,
            "confidence_lower": self.confidence_interval.lower if self.confidence_interval else 0,
            "confidence_upper": self.confidence_interval.upper if self.confidence_interval else 0,
            "xgboost_prediction": self.xgboost_prediction,
            "random_forest_prediction": self.random_forest_prediction,
            "bayesian_ridge_prediction": self.bayesian_ridge_prediction,
            "prophet_prediction": self.prophet_prediction,
            "ensemble_weights": self.ensemble_weights,
            "prediction_std": self.prediction_std,
            "uncertainty_level": self.uncertainty_level.value,
            "notes": self.notes,
        }


@dataclass
class PredictionError:
    """Represents the error between prediction and actual attendance."""
    
    error_id: Optional[int] = None
    game_id: int = 0
    prediction_id: int = 0
    actual_attendance: int = 0
    predicted_attendance: int = 0
    
    # Error metrics
    error: int = 0
    absolute_error: int = 0
    percentage_error: float = 0.0
    
    # Confidence interval coverage
    within_confidence_interval: bool = False
    
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate error metrics."""
        self.error = self.predicted_attendance - self.actual_attendance
        self.absolute_error = abs(self.error)
        if self.actual_attendance > 0:
            self.percentage_error = (self.absolute_error / self.actual_attendance) * 100
    
    def is_over_prediction(self) -> bool:
        """Check if prediction was too high."""
        return self.error > 0
    
    def is_under_prediction(self) -> bool:
        """Check if prediction was too low."""
        return self.error < 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "game_id": self.game_id,
            "prediction_id": self.prediction_id,
            "actual_attendance": self.actual_attendance,
            "predicted_attendance": self.predicted_attendance,
            "error": self.error,
            "absolute_error": self.absolute_error,
            "percentage_error": self.percentage_error,
            "within_confidence_interval": self.within_confidence_interval,
            "calculated_at": self.calculated_at.isoformat(),
        }
