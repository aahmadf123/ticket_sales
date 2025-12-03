"""Feature engineering service for ML models."""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Weather thresholds
    cold_temp_threshold: float = 40.0
    hot_temp_threshold: float = 85.0
    high_wind_threshold: float = 20.0
    
    # Comfort index weights
    temp_weight: float = 0.4
    wind_weight: float = 0.2
    precip_weight: float = 0.3
    humidity_weight: float = 0.1
    
    # Opponent scoring
    power_five_score: float = 1.0
    mac_score: float = 0.6
    fcs_score: float = 0.3


class FeatureEngineeringService:
    """Service for creating and transforming features for ML models."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def calculate_weather_comfort_index(
        self,
        temperature: float,
        wind_speed: float = 0.0,
        precipitation_prob: float = 0.0,
        humidity: float = 50.0
    ) -> float:
        """Calculate composite weather comfort index (0-100)."""
        comfort = 100.0
        
        # Temperature penalty
        if temperature < self.config.cold_temp_threshold:
            temp_penalty = (self.config.cold_temp_threshold - temperature) * 1.5
            comfort -= temp_penalty * self.config.temp_weight * 2
        elif temperature > self.config.hot_temp_threshold:
            temp_penalty = (temperature - self.config.hot_temp_threshold) * 1.5
            comfort -= temp_penalty * self.config.temp_weight * 2
        
        # Wind penalty
        if wind_speed > self.config.high_wind_threshold:
            wind_penalty = (wind_speed - self.config.high_wind_threshold) * 0.5
            comfort -= wind_penalty * self.config.wind_weight * 100
        
        # Precipitation penalty
        comfort -= precipitation_prob * self.config.precip_weight * 100
        
        # Humidity penalty (if extreme)
        if humidity > 80:
            comfort -= (humidity - 80) * self.config.humidity_weight * 2
        
        return max(0.0, min(100.0, comfort))
    
    def calculate_opponent_strength_score(
        self,
        opponent_tier: int,
        opponent_win_pct: float = 0.5,
        opponent_ranking: Optional[int] = None
    ) -> float:
        """Calculate composite opponent strength score (0-1.5)."""
        tier_scores = {
            1: self.config.power_five_score,
            2: self.config.mac_score,
            3: self.config.fcs_score,
        }
        
        base_score = tier_scores.get(opponent_tier, 0.5)
        
        # Add ranking bonus
        if opponent_ranking is not None and opponent_ranking > 0:
            rank_bonus = (26 - opponent_ranking) / 25 * 0.3
            base_score += max(0, rank_bonus)
        
        # Add win percentage factor
        base_score += opponent_win_pct * 0.2
        
        return min(1.5, base_score)
    
    def calculate_game_importance_factor(
        self,
        is_rivalry: bool = False,
        is_homecoming: bool = False,
        is_senior_day: bool = False,
        is_conference: bool = False,
        has_special_event: bool = False,
        bowl_eligible: bool = False
    ) -> float:
        """Calculate game importance factor (1.0-2.0)."""
        importance = 1.0
        
        if is_rivalry:
            importance += 0.4
        if is_homecoming:
            importance += 0.3
        if is_senior_day:
            importance += 0.2
        if has_special_event:
            importance += 0.15
        if is_conference:
            importance += 0.1
        if bowl_eligible:
            importance += 0.1
        
        return min(2.0, importance)
    
    def calculate_momentum_index(
        self,
        recent_results: List[int],
        weights: Optional[List[float]] = None
    ) -> float:
        """Calculate team momentum index based on recent results.
        
        Args:
            recent_results: List of recent game results (1=win, 0=loss), most recent first
            weights: Optional weights for each game (default: more recent = more weight)
        
        Returns:
            Momentum index (-1 to 1)
        """
        if not recent_results:
            return 0.0
        
        if weights is None:
            # Default: exponentially decreasing weights
            n = len(recent_results)
            weights = [0.5 ** i for i in range(n)]
        
        # Normalize weights
        total_weight = sum(weights[:len(recent_results)])
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted average (convert 0/1 to -1/1)
        momentum = sum(
            (r * 2 - 1) * w 
            for r, w in zip(recent_results, weights)
        ) / total_weight
        
        return momentum
    
    def calculate_seat_value_index(self, pr_level: str) -> float:
        """Calculate seat value index based on price level."""
        pr_level_lower = pr_level.lower()
        
        if "club" in pr_level_lower or "courtside" in pr_level_lower:
            return 1.5
        elif "loge" in pr_level_lower or "zone a" in pr_level_lower:
            return 1.3
        elif "zone b" in pr_level_lower or "lower" in pr_level_lower:
            return 1.1
        elif "rocket fund" in pr_level_lower:
            return 1.2
        elif "upper" in pr_level_lower:
            return 0.9
        elif "bleacher" in pr_level_lower:
            return 0.8
        elif "student" in pr_level_lower:
            return 0.7
        else:
            return 1.0
    
    def determine_pricing_tier(self, unit_price: float) -> int:
        """Determine pricing tier based on unit price.
        
        Returns:
            1 = Premium, 2 = Standard, 3 = Value
        """
        if unit_price > 50:
            return 1
        elif unit_price > 20:
            return 2
        else:
            return 3
    
    def engineer_ticket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from ticket sales data.
        
        Args:
            df: DataFrame with ticket sales data
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Calculate unit price
        df["unit_price"] = df.apply(
            lambda row: row["event_pmt"] / row["order_qty"] 
            if row["order_qty"] > 0 else 0,
            axis=1
        )
        
        # Calculate seat value index
        if "pr_level" in df.columns:
            df["seat_value_index"] = df["pr_level"].apply(self.calculate_seat_value_index)
        
        # Determine pricing tier
        df["pricing_tier"] = df["unit_price"].apply(self.determine_pricing_tier)
        
        # Binary indicators
        if "class_name" in df.columns:
            df["is_new_ticket"] = df["class_name"].str.contains(
                "(?i)new", regex=True, na=False
            ).astype(int)
            
            df["is_season_ticket"] = df["class_name"].str.contains(
                "(?i)season ticket", regex=True, na=False
            ).astype(int)
            
            df["is_comp"] = df["class_name"].str.contains(
                "(?i)comp", regex=True, na=False
            ).astype(int)
        
        if "price_type" in df.columns:
            df["is_alumni_ticket"] = df["price_type"].str.contains(
                "(?i)alumni", regex=True, na=False
            ).astype(int)
            
            df["is_transfer"] = df["price_type"].str.contains(
                "(?i)transfer", regex=True, na=False
            ).astype(int)
        
        # Extract class category
        if "class_name" in df.columns:
            df["class_category"] = df["class_name"].apply(
                lambda x: x.split(" - ")[-1] if " - " in str(x) else "Other"
            )
        
        # Revenue indicator
        df["is_revenue_generating"] = (df["event_pmt"] > 0).astype(int)
        
        return df
    
    def engineer_game_features(self, game_data: Dict) -> Dict[str, float]:
        """Engineer features for a single game.
        
        Args:
            game_data: Dictionary with raw game data
        
        Returns:
            Dictionary with engineered features
        """
        features = {}
        
        # Basic features
        features["tickets_sold"] = game_data.get("tickets_sold", 0)
        features["tickets_distributed"] = game_data.get("tickets_distributed", 0)
        
        # Weather features
        temp = game_data.get("temperature_f", 60.0)
        wind = game_data.get("wind_speed_mph", 0.0)
        precip = game_data.get("precipitation_prob", 0.0)
        humidity = game_data.get("humidity", 50.0)
        
        features["temperature_f"] = temp
        features["wind_speed_mph"] = wind
        features["precipitation_prob"] = precip
        features["weather_comfort_index"] = self.calculate_weather_comfort_index(
            temp, wind, precip, humidity
        )
        
        # Team performance features
        toledo_wins = game_data.get("toledo_wins", 0)
        toledo_losses = game_data.get("toledo_losses", 0)
        total_games = toledo_wins + toledo_losses
        
        features["toledo_win_percentage"] = (
            toledo_wins / total_games if total_games > 0 else 0.5
        )
        
        # Opponent features
        opp_wins = game_data.get("opponent_wins", 0)
        opp_losses = game_data.get("opponent_losses", 0)
        opp_total = opp_wins + opp_losses
        opp_win_pct = opp_wins / opp_total if opp_total > 0 else 0.5
        
        features["opponent_tier"] = game_data.get("opponent_tier", 2)
        features["opponent_strength_score"] = self.calculate_opponent_strength_score(
            game_data.get("opponent_tier", 2),
            opp_win_pct,
            game_data.get("opponent_ranking")
        )
        
        # Game type features
        features["is_weekend"] = 1 if game_data.get("is_weekend", True) else 0
        features["is_rivalry"] = 1 if game_data.get("is_rivalry_game", False) else 0
        features["is_homecoming"] = 1 if game_data.get("is_homecoming", False) else 0
        features["is_senior_day"] = 1 if game_data.get("is_senior_day", False) else 0
        features["is_conference_game"] = 1 if game_data.get("is_conference_game", False) else 0
        features["has_special_event"] = 1 if game_data.get("special_event") else 0
        
        # Game importance
        features["game_importance_factor"] = self.calculate_game_importance_factor(
            is_rivalry=game_data.get("is_rivalry_game", False),
            is_homecoming=game_data.get("is_homecoming", False),
            is_senior_day=game_data.get("is_senior_day", False),
            is_conference=game_data.get("is_conference_game", False),
            has_special_event=bool(game_data.get("special_event")),
        )
        
        # Season progression
        home_game_num = game_data.get("home_game_number", 1)
        total_home_games = game_data.get("total_home_games_season", 7)
        
        features["home_game_number"] = home_game_num
        features["total_home_games"] = total_home_games
        features["season_progress"] = home_game_num / max(1, total_home_games)
        
        # Historical context
        features["previous_season_avg_attendance"] = game_data.get(
            "previous_season_avg_attendance", 15000
        )
        
        return features
    
    def create_prophet_features(
        self,
        df: pd.DataFrame,
        date_col: str = "game_date",
        target_col: str = "actual_attendance"
    ) -> pd.DataFrame:
        """Create features for Facebook Prophet model.
        
        Args:
            df: DataFrame with game data
            date_col: Column name for date
            target_col: Column name for target variable
        
        Returns:
            DataFrame formatted for Prophet (ds, y, and regressors)
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = pd.to_datetime(df[date_col])
        prophet_df["y"] = df[target_col]
        
        # Add regressors
        regressor_cols = [
            "weather_comfort_index",
            "opponent_strength_score",
            "game_importance_factor",
            "is_weekend",
            "toledo_win_percentage",
        ]
        
        for col in regressor_cols:
            if col in df.columns:
                prophet_df[col] = df[col]
        
        return prophet_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in standard order."""
        return [
            "tickets_sold",
            "tickets_distributed",
            "toledo_win_percentage",
            "opponent_tier",
            "opponent_strength_score",
            "is_weekend",
            "is_rivalry",
            "is_homecoming",
            "is_senior_day",
            "is_conference_game",
            "has_special_event",
            "home_game_number",
            "total_home_games",
            "season_progress",
            "weather_comfort_index",
            "temperature_f",
            "precipitation_prob",
            "wind_speed_mph",
            "game_importance_factor",
            "previous_season_avg_attendance",
        ]
