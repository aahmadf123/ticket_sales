"""Game entity representing a single athletic event."""

from dataclasses import dataclass, field
from datetime import date, time
from typing import Optional
from enum import Enum


class OpponentTier(Enum):
    """Classification of opponent strength."""
    POWER_FIVE = 1
    MAC = 2
    FCS = 3


class SportType(Enum):
    """Type of sport."""
    FOOTBALL = "FB"
    BASKETBALL = "BB"
    VOLLEYBALL = "VB"


@dataclass
class Game:
    """Represents a single game/event."""
    
    game_id: Optional[int] = None
    season: str = ""
    season_code: str = ""
    game_date: Optional[date] = None
    kickoff_time: Optional[time] = None
    kickoff_time_tbd: bool = True
    opponent: str = ""
    opponent_tier: OpponentTier = OpponentTier.MAC
    sport_type: SportType = SportType.FOOTBALL
    is_conference_game: bool = False
    is_rivalry_game: bool = False
    is_homecoming: bool = False
    is_senior_day: bool = False
    special_event: Optional[str] = None
    home_game_number: int = 0
    total_home_games_season: int = 0
    
    # Ticket information
    tickets_sold: int = 0
    tickets_distributed: int = 0
    
    # Team performance
    toledo_wins: int = 0
    toledo_losses: int = 0
    toledo_win_percentage: float = 0.0
    opponent_wins: int = 0
    opponent_losses: int = 0
    opponent_ranking: Optional[int] = None
    
    # Weather data
    temperature_f: Optional[float] = None
    feels_like_f: Optional[float] = None
    precipitation_prob: float = 0.0
    precipitation_amount: float = 0.0
    wind_speed_mph: float = 0.0
    weather_condition: str = ""
    weather_comfort_index: float = 50.0
    
    # Previous season context
    previous_season_avg_attendance: float = 0.0
    
    # Actual outcomes (populated post-game)
    actual_attendance: Optional[int] = None
    actual_scanned_count: Optional[int] = None
    total_scan_rate: Optional[float] = None
    pre_kickoff_scan_rate: Optional[float] = None
    
    def calculate_win_percentage(self) -> float:
        """Calculate current win percentage."""
        total_games = self.toledo_wins + self.toledo_losses
        if total_games == 0:
            return 0.0
        return self.toledo_wins / total_games
    
    def calculate_weather_comfort_index(self) -> float:
        """Calculate weather comfort index based on conditions."""
        if self.temperature_f is None:
            return 50.0
        
        comfort = 100.0
        
        # Temperature penalty
        if self.temperature_f < 40:
            comfort -= (40 - self.temperature_f) * 1.5
        elif self.temperature_f > 85:
            comfort -= (self.temperature_f - 85) * 1.5
        
        # Wind penalty
        if self.wind_speed_mph > 20:
            comfort -= (self.wind_speed_mph - 20) * 0.5
        
        # Precipitation penalty
        comfort -= self.precipitation_prob * 30
        comfort -= self.precipitation_amount * 20
        
        return max(0.0, min(100.0, comfort))
    
    def get_opponent_strength_score(self) -> float:
        """Calculate opponent strength score."""
        tier_scores = {
            OpponentTier.POWER_FIVE: 1.0,
            OpponentTier.MAC: 0.6,
            OpponentTier.FCS: 0.3,
        }
        
        base_score = tier_scores.get(self.opponent_tier, 0.5)
        
        # Add ranking bonus
        if self.opponent_ranking is not None:
            rank_bonus = (26 - self.opponent_ranking) / 25 * 0.3
            base_score += max(0, rank_bonus)
        
        # Add win percentage factor
        opponent_total = self.opponent_wins + self.opponent_losses
        if opponent_total > 0:
            opp_win_pct = self.opponent_wins / opponent_total
            base_score += opp_win_pct * 0.2
        
        return min(1.5, base_score)
    
    def get_game_importance_factor(self) -> float:
        """Calculate game importance factor."""
        importance = 1.0
        
        if self.is_rivalry_game:
            importance += 0.4
        if self.is_homecoming:
            importance += 0.3
        if self.is_senior_day:
            importance += 0.2
        if self.special_event:
            importance += 0.15
        if self.is_conference_game:
            importance += 0.1
        
        return importance
    
    def is_weekend_game(self) -> bool:
        """Check if game is on weekend."""
        if self.game_date is None:
            return True
        return self.game_date.weekday() >= 5
    
    def to_feature_dict(self) -> dict:
        """Convert game to feature dictionary for ML models."""
        return {
            "tickets_sold": self.tickets_sold,
            "tickets_distributed": self.tickets_distributed,
            "toledo_win_percentage": self.calculate_win_percentage(),
            "opponent_tier": self.opponent_tier.value,
            "opponent_strength_score": self.get_opponent_strength_score(),
            "is_weekend": 1 if self.is_weekend_game() else 0,
            "is_rivalry": 1 if self.is_rivalry_game else 0,
            "is_homecoming": 1 if self.is_homecoming else 0,
            "is_senior_day": 1 if self.is_senior_day else 0,
            "is_conference_game": 1 if self.is_conference_game else 0,
            "has_special_event": 1 if self.special_event else 0,
            "home_game_number": self.home_game_number,
            "total_home_games": self.total_home_games_season,
            "season_progress": self.home_game_number / max(1, self.total_home_games_season),
            "weather_comfort_index": self.calculate_weather_comfort_index(),
            "temperature_f": self.temperature_f or 60.0,
            "precipitation_prob": self.precipitation_prob,
            "wind_speed_mph": self.wind_speed_mph,
            "game_importance_factor": self.get_game_importance_factor(),
            "previous_season_avg_attendance": self.previous_season_avg_attendance,
        }
