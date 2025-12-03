"""
Tests for Toledo Attendance Forecasting System

Run with: pytest tests/ -v
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import date, datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_weather_comfort_index_optimal(self):
        """Test weather comfort index with optimal conditions."""
        from domain.services.feature_engineering_service import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        comfort = service.calculate_weather_comfort_index(
            temperature_f=70.0,
            wind_speed_mph=5.0,
            precipitation_prob=0.0,
            humidity=50.0
        )
        
        assert comfort >= 90.0, "Optimal conditions should have high comfort"
        assert comfort <= 100.0, "Comfort should not exceed 100"
    
    def test_weather_comfort_index_cold(self):
        """Test weather comfort index with cold conditions."""
        from domain.services.feature_engineering_service import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        comfort = service.calculate_weather_comfort_index(
            temperature_f=35.0,
            wind_speed_mph=15.0,
            precipitation_prob=0.2,
            humidity=70.0
        )
        
        assert comfort < 70.0, "Cold windy conditions should have lower comfort"
        assert comfort >= 0.0, "Comfort should not be negative"
    
    def test_opponent_strength_score(self):
        """Test opponent strength score calculation."""
        from domain.services.feature_engineering_service import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        power_five_score = service.calculate_opponent_strength_score(
            opponent_tier=1,
            opponent_ranking=15,
            opponent_win_pct=0.75
        )
        
        mac_score = service.calculate_opponent_strength_score(
            opponent_tier=2,
            opponent_ranking=None,
            opponent_win_pct=0.5
        )
        
        assert power_five_score > mac_score, "Power 5 should have higher score than MAC"
    
    def test_game_importance_factor_rivalry(self):
        """Test game importance factor for rivalry games."""
        from domain.services.feature_engineering_service import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        rivalry_importance = service.calculate_game_importance_factor(
            is_rivalry=True,
            is_homecoming=False,
            is_senior_day=False,
            special_event=None,
            is_conference=True,
            toledo_bowl_eligible=False
        )
        
        regular_importance = service.calculate_game_importance_factor(
            is_rivalry=False,
            is_homecoming=False,
            is_senior_day=False,
            special_event=None,
            is_conference=False,
            toledo_bowl_eligible=False
        )
        
        assert rivalry_importance > regular_importance, "Rivalry should be more important"
        assert rivalry_importance <= 2.0, "Importance should cap at 2.0"


class TestGameEntity:
    """Test Game entity functionality."""
    
    def test_game_creation(self):
        """Test basic game creation."""
        from domain.entities.game import Game, OpponentTier, SportType
        
        game = Game(
            game_id=1,
            season=2025,
            season_code="FB25",
            game_date=date(2025, 9, 6),
            opponent="Bowling Green",
            opponent_tier=OpponentTier.MAC,
            sport_type=SportType.FB
        )
        
        assert game.opponent == "Bowling Green"
        assert game.opponent_tier == OpponentTier.MAC
    
    def test_win_percentage_calculation(self):
        """Test win percentage calculation."""
        from domain.entities.game import Game, OpponentTier, SportType
        
        game = Game(
            game_id=1,
            season=2025,
            season_code="FB25",
            game_date=date(2025, 10, 15),
            opponent="Ohio",
            opponent_tier=OpponentTier.MAC,
            sport_type=SportType.FB,
            toledo_wins=5,
            toledo_losses=2
        )
        
        win_pct = game.calculate_win_percentage()
        
        assert abs(win_pct - 0.714) < 0.01, "Win percentage should be 5/7"
    
    def test_is_weekend_saturday(self):
        """Test weekend detection for Saturday games."""
        from domain.entities.game import Game, OpponentTier, SportType
        
        game = Game(
            game_id=1,
            season=2025,
            season_code="FB25",
            game_date=date(2025, 9, 6),
            opponent="Test",
            opponent_tier=OpponentTier.MAC,
            sport_type=SportType.FB
        )
        
        assert game.is_weekend_game() == True, "September 6, 2025 is a Saturday"


class TestPredictionEntity:
    """Test Prediction entity functionality."""
    
    def test_prediction_interval(self):
        """Test prediction interval calculations."""
        from domain.entities.prediction import PredictionInterval
        
        interval = PredictionInterval(lower=14000, upper=16000, confidence_level=0.80)
        
        assert interval.width == 2000
        assert interval.midpoint == 15000
    
    def test_prediction_interval_contains(self):
        """Test contains method for prediction interval."""
        from domain.entities.prediction import PredictionInterval
        
        interval = PredictionInterval(lower=14000, upper=16000)
        
        assert interval.contains(15000) == True
        assert interval.contains(13000) == False
        assert interval.contains(17000) == False


class TestPricingTrajectory:
    """Test pricing trajectory functionality."""
    
    def test_churn_calculation_below_threshold(self):
        """Test churn calculation below threshold."""
        from domain.entities.pricing_trajectory import PricingTrajectory, SeatingSection, PricingTier
        
        trajectory = PricingTrajectory(
            section=SeatingSection.LOWER_RESERVED,
            pricing_tier=PricingTier.STANDARD,
            current_price=100.0,
            current_season=2025
        )
        
        churn = trajectory.calculate_churn_from_price_increase(0.03)
        
        assert churn < 0.05, "Small price increase should have minimal churn impact"
    
    def test_churn_calculation_above_threshold(self):
        """Test churn calculation above threshold."""
        from domain.entities.pricing_trajectory import PricingTrajectory, SeatingSection, PricingTier
        
        trajectory = PricingTrajectory(
            section=SeatingSection.LOWER_RESERVED,
            pricing_tier=PricingTier.STANDARD,
            current_price=100.0,
            current_season=2025
        )
        
        churn_low = trajectory.calculate_churn_from_price_increase(0.05)
        churn_high = trajectory.calculate_churn_from_price_increase(0.15)
        
        assert churn_high > churn_low, "Higher price increase should have more churn"
        assert churn_high <= 0.30, "Churn should be capped at 30%"


class TestDataPreprocessingAdapter:
    """Test data preprocessing adapter."""
    
    def test_column_standardization(self):
        """Test column name standardization."""
        from infrastructure.adapters.data_preprocessing_adapter import DataPreprocessingAdapter
        import pandas as pd
        
        adapter = DataPreprocessingAdapter()
        
        df = pd.DataFrame({
            'Season.Code': ['FB25'],
            'Event.Code': ['F01'],
            'Order.Qty..Total.': ['100'],
            'Event.Pmt..Total.': ['1,500']
        })
        
        result = adapter.standardize_columns(df)
        
        assert 'season_code' in result.columns
        assert 'event_code' in result.columns
        assert 'order_qty' in result.columns
        assert 'event_pmt' in result.columns
    
    def test_class_category_extraction(self):
        """Test class category extraction."""
        from infrastructure.adapters.data_preprocessing_adapter import DataPreprocessingAdapter
        
        adapter = DataPreprocessingAdapter()
        
        assert adapter._extract_class_category("Season Tickets - New") == "New"
        assert adapter._extract_class_category("Season Tickets - Renewal") == "Renewal"
        assert adapter._extract_class_category("Single Game") == "Other"


class TestOptimizationService:
    """Test price optimization service."""
    
    def test_churn_calculation(self):
        """Test churn calculation in optimization."""
        from domain.services.price_optimization_service import PriceOptimizationService
        
        service = PriceOptimizationService()
        
        churn_low = service._calculate_churn(0.03)
        churn_high = service._calculate_churn(0.12)
        
        assert churn_low < churn_high, "Higher increase should have higher churn"
        assert churn_low >= 0.05, "Should have at least baseline churn"
    
    def test_trajectory_metrics(self):
        """Test trajectory metrics calculation."""
        from domain.services.price_optimization_service import PriceOptimizationService
        
        service = PriceOptimizationService()
        
        trajectory = [100, 105, 110, 115, 120]
        increases = [0.05, 0.0476, 0.0455, 0.0435]
        
        metrics = service._calculate_trajectory_metrics(
            trajectory=trajectory,
            annual_increases=increases,
            base_demand=1000,
            price_elasticity=-0.5
        )
        
        assert 'total_revenue' in metrics
        assert 'total_attendance' in metrics
        assert 'final_retention_rate' in metrics
        assert metrics['total_revenue'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
