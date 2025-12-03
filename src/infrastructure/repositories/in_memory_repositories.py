"""In-memory repository implementations for testing and development."""

from typing import Dict, List, Optional, Any
from datetime import date, datetime
import uuid
import pandas as pd

from src.application.ports.ports import (
    GameRepositoryPort,
    TicketSaleRepositoryPort,
    PredictionRepositoryPort,
    SeasonTicketHolderRepositoryPort,
    PricingTrajectoryRepositoryPort,
)
from src.domain.entities.game import Game
from src.domain.entities.ticket_sale import TicketSale
from src.domain.entities.prediction import Prediction
from src.domain.entities.season_ticket_holder import SeasonTicketHolder
from src.domain.entities.pricing_trajectory import PricingTrajectory


class InMemoryGameRepository(GameRepositoryPort):
    """In-memory implementation of game repository."""
    
    def __init__(self):
        """Initialize repository."""
        self._games: Dict[str, Game] = {}
    
    def save(self, game: Game) -> str:
        """Save a game and return its ID."""
        if game.game_id is None:
            game.game_id = str(uuid.uuid4())
        self._games[game.game_id] = game
        return game.game_id
    
    def get_by_id(self, game_id: str) -> Optional[Game]:
        """Get a game by ID."""
        return self._games.get(game_id)
    
    def get_by_date_opponent(self, game_date: date, opponent: str) -> Optional[Game]:
        """Get a game by date and opponent."""
        for game in self._games.values():
            if game.game_date == game_date and game.opponent == opponent:
                return game
        return None
    
    def get_by_season(self, season: int) -> List[Game]:
        """Get all games for a season."""
        return [g for g in self._games.values() if g.season == season]
    
    def get_upcoming(self, days_ahead: int = 30) -> List[Game]:
        """Get upcoming games."""
        today = date.today()
        cutoff = date.fromordinal(today.toordinal() + days_ahead)
        
        upcoming = [
            g for g in self._games.values()
            if g.game_date >= today and g.game_date <= cutoff
        ]
        return sorted(upcoming, key=lambda x: x.game_date)
    
    def get_past_games(self, limit: int = 100) -> List[Game]:
        """Get past games with actual attendance."""
        today = date.today()
        past = [
            g for g in self._games.values()
            if g.game_date < today and g.actual_attendance is not None
        ]
        past.sort(key=lambda x: x.game_date, reverse=True)
        return past[:limit]
    
    def update(self, game: Game) -> bool:
        """Update a game."""
        if game.game_id in self._games:
            self._games[game.game_id] = game
            return True
        return False
    
    def delete(self, game_id: str) -> bool:
        """Delete a game."""
        if game_id in self._games:
            del self._games[game_id]
            return True
        return False
    
    def load_from_dataframe(self, df: pd.DataFrame) -> int:
        """Load games from DataFrame.
        
        Args:
            df: DataFrame with game data
        
        Returns:
            Number of games loaded
        """
        count = 0
        for _, row in df.iterrows():
            game = Game(
                game_id=str(uuid.uuid4()),
                season=row.get("season"),
                season_code=row.get("season_code"),
                game_date=pd.to_datetime(row.get("game_date")).date()
                    if pd.notna(row.get("game_date")) else None,
                opponent=row.get("opponent"),
                tickets_sold=row.get("tickets_sold"),
                actual_attendance=row.get("actual_attendance"),
            )
            self.save(game)
            count += 1
        return count


class InMemoryTicketSaleRepository(TicketSaleRepositoryPort):
    """In-memory implementation of ticket sale repository."""
    
    def __init__(self):
        """Initialize repository."""
        self._sales: Dict[str, TicketSale] = {}
    
    def save(self, ticket_sale: TicketSale) -> str:
        """Save a ticket sale and return its ID."""
        if ticket_sale.sale_id is None:
            ticket_sale.sale_id = str(uuid.uuid4())
        self._sales[ticket_sale.sale_id] = ticket_sale
        return ticket_sale.sale_id
    
    def save_batch(self, ticket_sales: List[TicketSale]) -> int:
        """Save multiple ticket sales, return count saved."""
        count = 0
        for sale in ticket_sales:
            self.save(sale)
            count += 1
        return count
    
    def get_by_event(self, event_code: str) -> List[TicketSale]:
        """Get all ticket sales for an event."""
        return [s for s in self._sales.values() if s.event_code == event_code]
    
    def get_by_season(self, season_code: str) -> List[TicketSale]:
        """Get all ticket sales for a season."""
        return [s for s in self._sales.values() if s.season_code == season_code]
    
    def get_revenue_by_event(self, event_code: str) -> float:
        """Get total revenue for an event."""
        sales = self.get_by_event(event_code)
        return sum(s.event_pmt for s in sales if s.event_pmt)
    
    def get_tickets_sold_by_event(self, event_code: str) -> int:
        """Get total tickets sold for an event."""
        sales = self.get_by_event(event_code)
        return sum(s.order_qty for s in sales if s.order_qty)
    
    def aggregate_by_event(self) -> pd.DataFrame:
        """Aggregate ticket sales by event."""
        data = []
        events = set(s.event_code for s in self._sales.values())
        
        for event in events:
            sales = self.get_by_event(event)
            data.append({
                "event_code": event,
                "total_qty": sum(s.order_qty or 0 for s in sales),
                "total_revenue": sum(s.event_pmt or 0 for s in sales),
                "transaction_count": len(sales),
            })
        
        return pd.DataFrame(data)
    
    def aggregate_by_season(self) -> pd.DataFrame:
        """Aggregate ticket sales by season."""
        data = []
        seasons = set(s.season_code for s in self._sales.values())
        
        for season in seasons:
            sales = self.get_by_season(season)
            data.append({
                "season_code": season,
                "total_qty": sum(s.order_qty or 0 for s in sales),
                "total_revenue": sum(s.event_pmt or 0 for s in sales),
                "transaction_count": len(sales),
                "unique_events": len(set(s.event_code for s in sales)),
            })
        
        return pd.DataFrame(data)


class InMemoryPredictionRepository(PredictionRepositoryPort):
    """In-memory implementation of prediction repository."""
    
    def __init__(self):
        """Initialize repository."""
        self._predictions: Dict[str, Prediction] = {}
        self._actuals: Dict[str, int] = {}  # game_id -> actual attendance
    
    def save(self, prediction: Prediction) -> str:
        """Save a prediction and return its ID."""
        if prediction.prediction_id is None:
            prediction.prediction_id = str(uuid.uuid4())
        self._predictions[prediction.prediction_id] = prediction
        return prediction.prediction_id
    
    def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
        """Get a prediction by ID."""
        return self._predictions.get(prediction_id)
    
    def get_by_game(self, game_id: str) -> List[Prediction]:
        """Get all predictions for a game."""
        predictions = [
            p for p in self._predictions.values()
            if p.game_id == game_id
        ]
        return sorted(predictions, key=lambda x: x.prediction_timestamp, reverse=True)
    
    def get_latest_by_game(self, game_id: str) -> Optional[Prediction]:
        """Get the most recent prediction for a game."""
        predictions = self.get_by_game(game_id)
        return predictions[0] if predictions else None
    
    def get_by_horizon(self, game_id: str, horizon: str) -> List[Prediction]:
        """Get predictions for a game at a specific horizon."""
        return [
            p for p in self._predictions.values()
            if p.game_id == game_id and p.horizon.name == horizon
        ]
    
    def get_all_with_actuals(self) -> List[Dict[str, Any]]:
        """Get all predictions that have corresponding actuals."""
        results = []
        
        for pred in self._predictions.values():
            if pred.game_id in self._actuals:
                results.append({
                    "prediction_id": pred.prediction_id,
                    "game_id": pred.game_id,
                    "predicted_attendance": pred.predicted_attendance,
                    "actual_attendance": self._actuals[pred.game_id],
                    "confidence_lower": pred.confidence_interval.lower,
                    "confidence_upper": pred.confidence_interval.upper,
                    "horizon": pred.horizon.name,
                })
        
        return results
    
    def set_actual(self, game_id: str, actual: int) -> None:
        """Set actual attendance for a game."""
        self._actuals[game_id] = actual


class InMemorySeasonTicketHolderRepository(SeasonTicketHolderRepositoryPort):
    """In-memory implementation of season ticket holder repository."""
    
    def __init__(self):
        """Initialize repository."""
        self._holders: Dict[str, SeasonTicketHolder] = {}
    
    def save(self, holder: SeasonTicketHolder) -> str:
        """Save a holder and return its ID."""
        if holder.holder_id is None:
            holder.holder_id = str(uuid.uuid4())
        self._holders[holder.holder_id] = holder
        return holder.holder_id
    
    def get_by_id(self, holder_id: str) -> Optional[SeasonTicketHolder]:
        """Get a holder by ID."""
        return self._holders.get(holder_id)
    
    def get_by_account(self, account_id: str) -> Optional[SeasonTicketHolder]:
        """Get a holder by account ID."""
        for holder in self._holders.values():
            if holder.account_id == account_id:
                return holder
        return None
    
    def get_active_holders(self, season: int) -> List[SeasonTicketHolder]:
        """Get all active holders for a season."""
        from src.domain.entities.season_ticket_holder import HolderStatus
        return [
            h for h in self._holders.values()
            if h.current_season == season and h.status == HolderStatus.ACTIVE
        ]
    
    def get_at_risk_holders(self, threshold: float = 0.5) -> List[SeasonTicketHolder]:
        """Get holders with churn probability above threshold."""
        return [
            h for h in self._holders.values()
            if (h.churn_probability or 0) >= threshold
        ]
    
    def get_churned_holders(self, season: int) -> List[SeasonTicketHolder]:
        """Get holders who churned in a season."""
        from src.domain.entities.season_ticket_holder import HolderStatus
        return [
            h for h in self._holders.values()
            if h.current_season == season and h.status == HolderStatus.CHURNED
        ]
    
    def update(self, holder: SeasonTicketHolder) -> bool:
        """Update a holder."""
        if holder.holder_id in self._holders:
            self._holders[holder.holder_id] = holder
            return True
        return False


class InMemoryPricingTrajectoryRepository(PricingTrajectoryRepositoryPort):
    """In-memory implementation of pricing trajectory repository."""
    
    def __init__(self):
        """Initialize repository."""
        self._trajectories: Dict[str, PricingTrajectory] = {}
    
    def save(self, trajectory: PricingTrajectory) -> str:
        """Save a trajectory and return its ID."""
        if trajectory.trajectory_id is None:
            trajectory.trajectory_id = str(uuid.uuid4())
        self._trajectories[trajectory.trajectory_id] = trajectory
        return trajectory.trajectory_id
    
    def get_by_id(self, trajectory_id: str) -> Optional[PricingTrajectory]:
        """Get a trajectory by ID."""
        return self._trajectories.get(trajectory_id)
    
    def get_by_section(self, section: str) -> List[PricingTrajectory]:
        """Get all trajectories for a seating section."""
        return [
            t for t in self._trajectories.values()
            if t.section.name.lower() == section.lower()
        ]
    
    def get_active_trajectories(self) -> List[PricingTrajectory]:
        """Get all active pricing trajectories."""
        return list(self._trajectories.values())
    
    def update(self, trajectory: PricingTrajectory) -> bool:
        """Update a trajectory."""
        if trajectory.trajectory_id in self._trajectories:
            self._trajectories[trajectory.trajectory_id] = trajectory
            return True
        return False
