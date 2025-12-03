"""Data preprocessing adapter that replicates R code functionality."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import re
from decimal import Decimal

from ...domain.entities.ticket_sale import TicketSale, TicketClass
from ...domain.entities.game import Game, SportType


class DataPreprocessingAdapter:
    """Adapter for preprocessing ticket sales data.
    
    Replicates the R code preprocessing logic:
    - Column selection and renaming
    - Filtering (exclude Parking, Comp, zero payments)
    - Type conversion
    - Feature engineering (new_tickets, alumni_tickets, class_category)
    """
    
    # Column mapping from raw CSV to standardized names
    COLUMN_MAPPING = {
        "Season Code": "season_code",
        "Season.Code": "season_code",
        "Event Code": "event_code",
        "Event.Code": "event_code",
        "Item Code": "item_code",
        "Item.Code": "item_code",
        "Pr Level Full Name": "pr_level",
        "Pr.Level.Full.Name": "pr_level",
        "Order Qty (Total)": "order_qty",
        "Order.Qty..Total.": "order_qty",
        "Event Amt (Total)": "event_amt",
        "Event.Amt..Total.": "event_amt",
        "Event Pmt (Total)": "event_pmt",
        "Event.Pmt..Total.": "event_pmt",
        "Price Type Full Name": "price_type",
        "Price.Type.Full.Name": "price_type",
        "Class Full Name": "class_name",
        "Class.Full.Name": "class_name",
    }
    
    # Patterns to exclude
    EXCLUDE_PATTERNS = ["Parking", "Comp"]
    
    def __init__(self):
        self.processed_data = None
        self.raw_data = None
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            DataFrame with raw data
        """
        # Try different encodings
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                self.raw_data = df.copy()
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not read file with any standard encoding: {file_path}")
    
    def load_excel(self, file_path: str, sheet_name: int = 0) -> pd.DataFrame:
        """Load Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet index to load
        
        Returns:
            DataFrame with raw data
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.raw_data = df.copy()
        return df
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using mapping.
        
        Args:
            df: DataFrame with original column names
        
        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        
        # Apply column mapping
        rename_map = {}
        for old_name in df.columns:
            if old_name in self.COLUMN_MAPPING:
                rename_map[old_name] = self.COLUMN_MAPPING[old_name]
        
        df = df.rename(columns=rename_map)
        
        return df
    
    def filter_data(
        self,
        df: pd.DataFrame,
        exclude_patterns: Optional[List[str]] = None,
        exclude_zero_payment: bool = True
    ) -> pd.DataFrame:
        """Filter data based on R code logic.
        
        Replicates:
        filter(
            !str_detect(class_name, "Parking|Comp"),
            event_pmt != 0
        )
        
        Args:
            df: DataFrame to filter
            exclude_patterns: Patterns to exclude from class_name
            exclude_zero_payment: Whether to exclude zero payment rows
        
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        if exclude_patterns is None:
            exclude_patterns = self.EXCLUDE_PATTERNS
        
        # Build regex pattern
        pattern = "|".join(exclude_patterns)
        
        # Filter out matching patterns in class_name
        if "class_name" in df.columns:
            mask = ~df["class_name"].str.contains(
                pattern, case=False, na=False, regex=True
            )
            df = df[mask]
        
        # Filter out zero payment
        if exclude_zero_payment and "event_pmt" in df.columns:
            df = df[df["event_pmt"] != 0]
        
        return df.reset_index(drop=True)
    
    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column types.
        
        Replicates:
        mutate(
            order_qty = as.numeric(gsub(",", "", order_qty)),
            event_pmt = as.numeric(gsub(",", "", event_pmt))
        )
        
        Args:
            df: DataFrame with string columns
        
        Returns:
            DataFrame with converted types
        """
        df = df.copy()
        
        # Convert order_qty
        if "order_qty" in df.columns:
            df["order_qty"] = df["order_qty"].apply(self._parse_numeric)
        
        # Convert event_pmt
        if "event_pmt" in df.columns:
            df["event_pmt"] = df["event_pmt"].apply(self._parse_numeric)
        
        # Convert event_amt
        if "event_amt" in df.columns:
            df["event_amt"] = df["event_amt"].apply(self._parse_numeric)
        
        return df
    
    def _parse_numeric(self, value) -> float:
        """Parse numeric value, removing commas and handling special cases.
        
        Args:
            value: Value to parse
        
        Returns:
            Float value
        """
        if pd.isna(value):
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Remove commas and convert
        clean_value = str(value).replace(",", "").strip()
        
        try:
            return float(clean_value)
        except ValueError:
            return 0.0
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data.
        
        Replicates:
        mutate(
            new_tickets = ifelse(str_detect(class_name, "(?i)New"), 1, 0),
            alumni_tickets = ifelse(str_detect(price_type, "(?i)Alumni"), 1, 0),
            class_category = ifelse(str_detect(class_name, " - "),
                                   str_extract(class_name, "(?<= - ).*"),
                                   "Other")
        )
        
        Args:
            df: DataFrame with standardized columns
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # new_tickets: detect "New" in class_name
        if "class_name" in df.columns:
            df["new_tickets"] = df["class_name"].str.contains(
                "(?i)new", regex=True, na=False
            ).astype(int)
        
        # alumni_tickets: detect "Alumni" in price_type
        if "price_type" in df.columns:
            df["alumni_tickets"] = df["price_type"].str.contains(
                "(?i)alumni", regex=True, na=False
            ).astype(int)
        
        # class_category: extract text after " - " or "Other"
        if "class_name" in df.columns:
            df["class_category"] = df["class_name"].apply(self._extract_class_category)
        
        # Calculate unit price
        if "event_pmt" in df.columns and "order_qty" in df.columns:
            df["unit_price"] = df.apply(
                lambda row: row["event_pmt"] / row["order_qty"]
                if row["order_qty"] > 0 else 0,
                axis=1
            )
        
        # Determine pricing tier
        if "unit_price" in df.columns:
            df["pricing_tier"] = df["unit_price"].apply(self._determine_pricing_tier)
        
        # Calculate seat value index
        if "pr_level" in df.columns:
            df["seat_value_index"] = df["pr_level"].apply(self._calculate_seat_value_index)
        
        # Sport type from season code
        if "season_code" in df.columns:
            df["sport_type"] = df["season_code"].apply(self._extract_sport_type)
        
        # Season year
        if "season_code" in df.columns:
            df["season_year"] = df["season_code"].apply(self._extract_season_year)
        
        # Is revenue generating
        if "event_pmt" in df.columns:
            df["is_revenue_generating"] = (df["event_pmt"] > 0).astype(int)
        
        # Is season ticket
        if "class_name" in df.columns:
            df["is_season_ticket"] = df["class_name"].str.contains(
                "(?i)season ticket", regex=True, na=False
            ).astype(int)
        
        # Is comp
        if "class_name" in df.columns:
            df["is_comp"] = df["class_name"].str.contains(
                "(?i)comp", regex=True, na=False
            ).astype(int)
        
        # Is transfer
        if "class_name" in df.columns:
            df["is_transfer"] = df["class_name"].str.contains(
                "(?i)transfer", regex=True, na=False
            ).astype(int)
        
        return df
    
    def _extract_class_category(self, class_name: str) -> str:
        """Extract class category from class name.
        
        Args:
            class_name: Full class name
        
        Returns:
            Category string
        """
        if pd.isna(class_name):
            return "Other"
        
        if " - " in str(class_name):
            # Extract text after " - "
            match = re.search(r"(?<= - ).*", str(class_name))
            if match:
                return match.group()
        
        return "Other"
    
    def _determine_pricing_tier(self, unit_price: float) -> int:
        """Determine pricing tier from unit price.
        
        Returns:
            1 = Premium, 2 = Standard, 3 = Value
        """
        if unit_price > 50:
            return 1
        elif unit_price > 20:
            return 2
        else:
            return 3
    
    def _calculate_seat_value_index(self, pr_level: str) -> float:
        """Calculate seat value index from price level.
        
        Args:
            pr_level: Price level string
        
        Returns:
            Seat value index (0.7 - 1.5)
        """
        if pd.isna(pr_level):
            return 1.0
        
        pr_level_lower = str(pr_level).lower()
        
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
    
    def _extract_sport_type(self, season_code: str) -> str:
        """Extract sport type from season code.
        
        Args:
            season_code: e.g., "FB24", "BB25", "VB23"
        
        Returns:
            Sport type string
        """
        if pd.isna(season_code):
            return "UNKNOWN"
        
        code = str(season_code).upper()
        
        if code.startswith("FB"):
            return "FOOTBALL"
        elif code.startswith("BB"):
            return "BASKETBALL"
        elif code.startswith("VB"):
            return "VOLLEYBALL"
        else:
            return "OTHER"
    
    def _extract_season_year(self, season_code: str) -> int:
        """Extract season year from season code.
        
        Args:
            season_code: e.g., "FB24", "BB25"
        
        Returns:
            Year as integer (20XX format)
        """
        if pd.isna(season_code):
            return 0
        
        # Extract digits
        digits = re.findall(r"\d+", str(season_code))
        
        if digits:
            year = int(digits[0])
            # Convert 2-digit to 4-digit year
            if year < 100:
                year = 2000 + year
            return year
        
        return 0
    
    def preprocess(
        self,
        df: pd.DataFrame,
        exclude_patterns: Optional[List[str]] = None,
        exclude_zero_payment: bool = True
    ) -> pd.DataFrame:
        """Full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            exclude_patterns: Patterns to exclude
            exclude_zero_payment: Whether to exclude zero payments
        
        Returns:
            Fully preprocessed DataFrame
        """
        # Step 1: Standardize columns
        df = self.standardize_columns(df)
        
        # Step 2: Filter data
        df = self.filter_data(df, exclude_patterns, exclude_zero_payment)
        
        # Step 3: Convert types
        df = self.convert_types(df)
        
        # Step 4: Engineer features
        df = self.engineer_features(df)
        
        self.processed_data = df
        
        return df
    
    def merge_datasets(
        self,
        current_year_df: pd.DataFrame,
        past_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge current year and past data.
        
        Replicates: bind_rows(currentyear_copy, twoyear_copy)
        
        Args:
            current_year_df: Current year data
            past_data_df: Past years data
        
        Returns:
            Merged DataFrame
        """
        # Preprocess both datasets
        current_processed = self.preprocess(current_year_df)
        past_processed = self.preprocess(past_data_df)
        
        # Merge
        merged = pd.concat([current_processed, past_processed], ignore_index=True)
        
        # Additional filter after merge (as in R code)
        merged = merged[merged["event_pmt"] != 0].reset_index(drop=True)
        
        return merged
    
    def process_from_files(
        self,
        current_year_path: str,
        past_data_path: str
    ) -> pd.DataFrame:
        """Process data from file paths.
        
        Args:
            current_year_path: Path to current year CSV
            past_data_path: Path to past data CSV
        
        Returns:
            Fully processed and merged DataFrame
        """
        # Load files
        current_df = self.load_csv(current_year_path)
        past_df = self.load_csv(past_data_path)
        
        # Merge and process
        merged = self.merge_datasets(current_df, past_df)
        
        return merged
    
    def to_ticket_sales(self, df: pd.DataFrame) -> List[TicketSale]:
        """Convert DataFrame rows to TicketSale entities.
        
        Args:
            df: Preprocessed DataFrame
        
        Returns:
            List of TicketSale entities
        """
        tickets = []
        
        for _, row in df.iterrows():
            ticket = TicketSale(
                season_code=str(row.get("season_code", "")),
                event_code=str(row.get("event_code", "")),
                item_code=str(row.get("item_code", "")),
                pr_level=str(row.get("pr_level", "")),
                order_qty=int(row.get("order_qty", 0)),
                event_amt=Decimal(str(row.get("event_amt", 0))),
                event_pmt=Decimal(str(row.get("event_pmt", 0))),
                price_type=str(row.get("price_type", "")),
                class_name=str(row.get("class_name", "")),
            )
            tickets.append(ticket)
        
        return tickets
    
    def aggregate_by_event(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ticket data by event.
        
        Args:
            df: Preprocessed DataFrame
        
        Returns:
            DataFrame aggregated by event
        """
        agg_df = df.groupby(["season_code", "event_code"]).agg({
            "order_qty": "sum",
            "event_pmt": "sum",
            "event_amt": "sum",
            "new_tickets": "sum",
            "alumni_tickets": "sum",
            "is_season_ticket": "sum",
            "is_revenue_generating": "sum",
            "unit_price": "mean",
            "seat_value_index": "mean",
        }).reset_index()
        
        agg_df.columns = [
            "season_code", "event_code", "total_tickets", "total_revenue",
            "total_amount", "new_ticket_count", "alumni_ticket_count",
            "season_ticket_count", "revenue_generating_count",
            "avg_unit_price", "avg_seat_value_index"
        ]
        
        return agg_df
    
    def aggregate_by_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ticket data by season.
        
        Args:
            df: Preprocessed DataFrame
        
        Returns:
            DataFrame aggregated by season
        """
        agg_df = df.groupby("season_code").agg({
            "order_qty": "sum",
            "event_pmt": "sum",
            "event_amt": "sum",
            "new_tickets": "sum",
            "alumni_tickets": "sum",
            "is_season_ticket": "sum",
            "unit_price": "mean",
        }).reset_index()
        
        agg_df.columns = [
            "season_code", "total_tickets", "total_revenue", "total_amount",
            "new_ticket_count", "alumni_ticket_count", "season_ticket_count",
            "avg_unit_price"
        ]
        
        return agg_df
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for preprocessed data.
        
        Args:
            df: Preprocessed DataFrame
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "total_rows": len(df),
            "unique_seasons": df["season_code"].nunique() if "season_code" in df.columns else 0,
            "unique_events": df["event_code"].nunique() if "event_code" in df.columns else 0,
            "total_tickets": int(df["order_qty"].sum()) if "order_qty" in df.columns else 0,
            "total_revenue": float(df["event_pmt"].sum()) if "event_pmt" in df.columns else 0,
            "avg_unit_price": float(df["unit_price"].mean()) if "unit_price" in df.columns else 0,
            "new_ticket_pct": float(df["new_tickets"].mean() * 100) if "new_tickets" in df.columns else 0,
            "season_ticket_pct": float(df["is_season_ticket"].mean() * 100) if "is_season_ticket" in df.columns else 0,
            "sport_distribution": df["sport_type"].value_counts().to_dict() if "sport_type" in df.columns else {},
        }
