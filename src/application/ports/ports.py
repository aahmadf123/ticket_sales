"""Application ports - interfaces for the hexagonal architecture."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import date
import pandas as pd

from src.domain.entities.game import Game
from src.domain.entities.ticket_sale import TicketSale
from src.domain.entities.prediction import Prediction
from src.domain.entities.season_ticket_holder import SeasonTicketHolder
from src.domain.entities.pricing_trajectory import PricingTrajectory


class GameRepositoryPort(ABC):
    """Port for game data persistence."""
    
    @abstractmethod
    def save(self, game: Game) -> str:
        """Save a game and return its ID."""
        pass
    
    @abstractmethod
    def get_by_id(self, game_id: str) -> Optional[Game]:
        """Get a game by ID."""
        pass
    
    @abstractmethod
    def get_by_date_opponent(self, game_date: date, opponent: str) -> Optional[Game]:
        """Get a game by date and opponent."""
        pass
    
    @abstractmethod
    def get_by_season(self, season: int) -> List[Game]:
        """Get all games for a season."""
        pass
    
    @abstractmethod
    def get_upcoming(self, days_ahead: int = 30) -> List[Game]:
        """Get upcoming games."""
        pass
    
    @abstractmethod
    def get_past_games(self, limit: int = 100) -> List[Game]:
        """Get past games with actual attendance."""
        pass
    
    @abstractmethod
    def update(self, game: Game) -> bool:
        """Update a game."""
        pass
    
    @abstractmethod
    def delete(self, game_id: str) -> bool:
        """Delete a game."""
        pass


class TicketSaleRepositoryPort(ABC):
    """Port for ticket sale data persistence."""
    
    @abstractmethod
    def save(self, ticket_sale: TicketSale) -> str:
        """Save a ticket sale and return its ID."""
        pass
    
    @abstractmethod
    def save_batch(self, ticket_sales: List[TicketSale]) -> int:
        """Save multiple ticket sales, return count saved."""
        pass
    
    @abstractmethod
    def get_by_event(self, event_code: str) -> List[TicketSale]:
        """Get all ticket sales for an event."""
        pass
    
    @abstractmethod
    def get_by_season(self, season_code: str) -> List[TicketSale]:
        """Get all ticket sales for a season."""
        pass
    
    @abstractmethod
    def get_revenue_by_event(self, event_code: str) -> float:
        """Get total revenue for an event."""
        pass
    
    @abstractmethod
    def get_tickets_sold_by_event(self, event_code: str) -> int:
        """Get total tickets sold for an event."""
        pass
    
    @abstractmethod
    def aggregate_by_event(self) -> pd.DataFrame:
        """Aggregate ticket sales by event."""
        pass
    
    @abstractmethod
    def aggregate_by_season(self) -> pd.DataFrame:
        """Aggregate ticket sales by season."""
        pass


class PredictionRepositoryPort(ABC):
    """Port for prediction data persistence."""
    
    @abstractmethod
    def save(self, prediction: Prediction) -> str:
        """Save a prediction and return its ID."""
        pass
    
    @abstractmethod
    def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
        """Get a prediction by ID."""
        pass
    
    @abstractmethod
    def get_by_game(self, game_id: str) -> List[Prediction]:
        """Get all predictions for a game."""
        pass
    
    @abstractmethod
    def get_latest_by_game(self, game_id: str) -> Optional[Prediction]:
        """Get the most recent prediction for a game."""
        pass
    
    @abstractmethod
    def get_by_horizon(self, game_id: str, horizon: str) -> List[Prediction]:
        """Get predictions for a game at a specific horizon."""
        pass
    
    @abstractmethod
    def get_all_with_actuals(self) -> List[Dict[str, Any]]:
        """Get all predictions that have corresponding actuals."""
        pass


class SeasonTicketHolderRepositoryPort(ABC):
    """Port for season ticket holder data persistence."""
    
    @abstractmethod
    def save(self, holder: SeasonTicketHolder) -> str:
        """Save a holder and return its ID."""
        pass
    
    @abstractmethod
    def get_by_id(self, holder_id: str) -> Optional[SeasonTicketHolder]:
        """Get a holder by ID."""
        pass
    
    @abstractmethod
    def get_by_account(self, account_id: str) -> Optional[SeasonTicketHolder]:
        """Get a holder by account ID."""
        pass
    
    @abstractmethod
    def get_active_holders(self, season: int) -> List[SeasonTicketHolder]:
        """Get all active holders for a season."""
        pass
    
    @abstractmethod
    def get_at_risk_holders(self, threshold: float = 0.5) -> List[SeasonTicketHolder]:
        """Get holders with churn probability above threshold."""
        pass
    
    @abstractmethod
    def get_churned_holders(self, season: int) -> List[SeasonTicketHolder]:
        """Get holders who churned in a season."""
        pass
    
    @abstractmethod
    def update(self, holder: SeasonTicketHolder) -> bool:
        """Update a holder."""
        pass


class PricingTrajectoryRepositoryPort(ABC):
    """Port for pricing trajectory data persistence."""
    
    @abstractmethod
    def save(self, trajectory: PricingTrajectory) -> str:
        """Save a trajectory and return its ID."""
        pass
    
    @abstractmethod
    def get_by_id(self, trajectory_id: str) -> Optional[PricingTrajectory]:
        """Get a trajectory by ID."""
        pass
    
    @abstractmethod
    def get_by_section(self, section: str) -> List[PricingTrajectory]:
        """Get all trajectories for a seating section."""
        pass
    
    @abstractmethod
    def get_active_trajectories(self) -> List[PricingTrajectory]:
        """Get all active pricing trajectories."""
        pass
    
    @abstractmethod
    def update(self, trajectory: PricingTrajectory) -> bool:
        """Update a trajectory."""
        pass


class WeatherServicePort(ABC):
    """Port for weather data retrieval."""
    
    @abstractmethod
    def get_current_weather(self) -> Dict[str, Any]:
        """Get current weather conditions."""
        pass
    
    @abstractmethod
    def get_forecast(self, game_date: date) -> Dict[str, Any]:
        """Get weather forecast for a date."""
        pass
    
    @abstractmethod
    def get_historical_weather(self, game_date: date) -> Dict[str, Any]:
        """Get historical weather for a past date."""
        pass
    
    @abstractmethod
    def calculate_comfort_index(self, weather_data: Dict[str, Any]) -> float:
        """Calculate weather comfort index."""
        pass


class DataPreprocessingPort(ABC):
    """Port for data preprocessing."""
    
    @abstractmethod
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        pass
    
    @abstractmethod
    def load_excel(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Load data from Excel file."""
        pass
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full preprocessing pipeline."""
        pass
    
    @abstractmethod
    def merge_datasets(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple datasets."""
        pass
    
    @abstractmethod
    def to_ticket_sales(self, df: pd.DataFrame) -> List[TicketSale]:
        """Convert DataFrame to TicketSale entities."""
        pass


class ModelPersistencePort(ABC):
    """Port for ML model persistence."""
    
    @abstractmethod
    def save_model(self, model: Any, model_name: str, version: str) -> str:
        """Save a trained model and return path."""
        pass
    
    @abstractmethod
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a trained model."""
        pass
    
    @abstractmethod
    def list_versions(self, model_name: str) -> List[str]:
        """List available versions for a model."""
        pass
    
    @abstractmethod
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a model version."""
        pass


class NotificationPort(ABC):
    """Port for sending notifications."""
    
    @abstractmethod
    def send_prediction_alert(
        self,
        game: Game,
        prediction: Prediction,
        recipients: List[str]
    ) -> bool:
        """Send prediction alert to recipients."""
        pass
    
    @abstractmethod
    def send_churn_alert(
        self,
        holder: SeasonTicketHolder,
        recipients: List[str]
    ) -> bool:
        """Send churn risk alert."""
        pass
    
    @abstractmethod
    def send_data_entry_reminder(
        self,
        games_needing_data: List[Game],
        recipients: List[str]
    ) -> bool:
        """Send reminder for games needing attendance data."""
        pass
