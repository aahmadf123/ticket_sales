"""Ticket sale entity representing individual ticket transactions."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from decimal import Decimal


class TicketClass(Enum):
    """Classification of ticket type."""
    SINGLE_GAME_STANDARD = "Single Game - Standard"
    SINGLE_GAME_DAY_OF = "Single Game - Day of Game"
    SINGLE_GAME_FS = "Single Game - F/S"
    SINGLE_GAME_PROMO = "Single Game - Promo/Comp Non Sales Leads"
    SINGLE_GAME_PROMO_SALES = "Single Game - Promo/Comp Sales Leads"
    SEASON_TICKETS_STANDARD = "Season Tickets - Standard"
    SEASON_TICKETS_COMP = "Season Tickets - Comp"
    SEASON_TICKETS_FS = "Season Tickets - F/S"
    SEASON_TICKETS_FLEX = "Season Tickets - Flex"
    NEW_SEASON_STANDARD = "New Season Tickets - Standard"
    NEW_SEASON_COMP = "New Season Tickets - Comp"
    NEW_SEASON_FS = "New Season Tickets - F/S"
    NEW_SEASON_FLEX = "New Season Tickets - Flex"
    TRANSFER = "Transfer"
    GROUP_TICKETS = "Group Tickets"
    MINI_PLAN = "Mini Plan - Standard"
    MINI_PLAN_FS = "Mini Plan - F/S"
    PARKING = "Parking"
    MISCELLANEOUS = "Miscellaneous"
    OTHER = "Other"


@dataclass
class TicketSale:
    """Represents a ticket sale transaction."""
    
    sale_id: Optional[int] = None
    season_code: str = ""
    event_code: str = ""
    item_code: str = ""
    pr_level: str = ""
    order_qty: int = 0
    event_amt: Decimal = Decimal("0.00")
    event_pmt: Decimal = Decimal("0.00")
    price_type: str = ""
    class_name: str = ""
    
    # Derived fields
    is_new_ticket: bool = False
    is_alumni_ticket: bool = False
    class_category: str = "Other"
    ticket_class: TicketClass = TicketClass.OTHER
    
    # Pricing analysis
    unit_price: Decimal = Decimal("0.00")
    seat_value_index: float = 1.0
    pricing_tier: int = 2
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self._calculate_derived_fields()
    
    def _calculate_derived_fields(self):
        """Calculate derived fields based on raw data."""
        # Determine if new ticket
        self.is_new_ticket = "new" in self.class_name.lower()
        
        # Determine if alumni ticket
        self.is_alumni_ticket = "alumni" in self.price_type.lower()
        
        # Extract class category
        if " - " in self.class_name:
            parts = self.class_name.split(" - ")
            self.class_category = parts[-1] if len(parts) > 1 else "Other"
        else:
            self.class_category = "Other"
        
        # Map to ticket class enum
        self.ticket_class = self._map_ticket_class()
        
        # Calculate unit price
        if self.order_qty > 0:
            self.unit_price = self.event_pmt / self.order_qty
        
        # Calculate seat value index
        self.seat_value_index = self._calculate_seat_value_index()
        
        # Determine pricing tier
        self.pricing_tier = self._determine_pricing_tier()
    
    def _map_ticket_class(self) -> TicketClass:
        """Map class name to TicketClass enum."""
        class_mapping = {
            "single game - standard": TicketClass.SINGLE_GAME_STANDARD,
            "single game - day of game": TicketClass.SINGLE_GAME_DAY_OF,
            "single game - f/s": TicketClass.SINGLE_GAME_FS,
            "single game - promo/comp non sales leads": TicketClass.SINGLE_GAME_PROMO,
            "single game - promo/comp sales leads": TicketClass.SINGLE_GAME_PROMO_SALES,
            "season tickets - standard": TicketClass.SEASON_TICKETS_STANDARD,
            "season tickets - comp": TicketClass.SEASON_TICKETS_COMP,
            "season tickets - f/s": TicketClass.SEASON_TICKETS_FS,
            "season tickets - flex": TicketClass.SEASON_TICKETS_FLEX,
            "new season tickets - standard": TicketClass.NEW_SEASON_STANDARD,
            "new season tickets - comp": TicketClass.NEW_SEASON_COMP,
            "new season tickets - f/s": TicketClass.NEW_SEASON_FS,
            "new season tickets - flex": TicketClass.NEW_SEASON_FLEX,
            "transfer": TicketClass.TRANSFER,
            "group tickets": TicketClass.GROUP_TICKETS,
            "mini plan - standard": TicketClass.MINI_PLAN,
            "mini plan - f/s": TicketClass.MINI_PLAN_FS,
            "parking": TicketClass.PARKING,
            "miscellaneous": TicketClass.MISCELLANEOUS,
        }
        
        return class_mapping.get(self.class_name.lower(), TicketClass.OTHER)
    
    def _calculate_seat_value_index(self) -> float:
        """Calculate seat value index based on price level."""
        # Premium levels get higher index
        pr_level_lower = self.pr_level.lower()
        
        if "club" in pr_level_lower or "courtside" in pr_level_lower:
            return 1.5
        elif "loge" in pr_level_lower or "zone a" in pr_level_lower:
            return 1.3
        elif "zone b" in pr_level_lower or "lower" in pr_level_lower:
            return 1.1
        elif "upper" in pr_level_lower or "bleacher" in pr_level_lower:
            return 0.8
        else:
            return 1.0
    
    def _determine_pricing_tier(self) -> int:
        """Determine pricing tier (1=premium, 2=standard, 3=value)."""
        if self.unit_price > 50:
            return 1
        elif self.unit_price > 20:
            return 2
        else:
            return 3
    
    def is_revenue_generating(self) -> bool:
        """Check if this sale generates revenue."""
        return self.event_pmt > 0
    
    def is_parking(self) -> bool:
        """Check if this is a parking ticket."""
        return self.ticket_class == TicketClass.PARKING
    
    def is_comp(self) -> bool:
        """Check if this is a complimentary ticket."""
        return "comp" in self.class_name.lower() or "comp" in self.price_type.lower()
    
    def is_season_ticket(self) -> bool:
        """Check if this is a season ticket."""
        return "season ticket" in self.class_name.lower()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for analysis."""
        return {
            "season_code": self.season_code,
            "event_code": self.event_code,
            "item_code": self.item_code,
            "pr_level": self.pr_level,
            "order_qty": self.order_qty,
            "event_amt": float(self.event_amt),
            "event_pmt": float(self.event_pmt),
            "price_type": self.price_type,
            "class_name": self.class_name,
            "is_new_ticket": self.is_new_ticket,
            "is_alumni_ticket": self.is_alumni_ticket,
            "class_category": self.class_category,
            "ticket_class": self.ticket_class.value,
            "unit_price": float(self.unit_price),
            "seat_value_index": self.seat_value_index,
            "pricing_tier": self.pricing_tier,
            "is_revenue_generating": self.is_revenue_generating(),
            "is_comp": self.is_comp(),
            "is_season_ticket": self.is_season_ticket(),
        }
