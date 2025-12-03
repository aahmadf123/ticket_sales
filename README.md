# Toledo Attendance Forecasting System

An AI-driven machine learning system for predicting game attendance and optimizing ticket pricing for University of Toledo Athletics.

## Overview

This system provides:
- **Attendance Forecasting**: Ensemble ML models (XGBoost, Random Forest, Bayesian Ridge, Prophet) to predict game attendance
- **Price Optimization**: Multi-year ticket pricing trajectory optimization with churn constraints
- **Churn Modeling**: Season ticket holder retention analysis and risk prediction
- **Feature Engineering**: Automated feature extraction from ticket sales, weather, and team performance data

## Architecture

The system follows **hexagonal architecture** principles with clear separation of concerns:

```
toledo_attendance_system/
├── src/
│   ├── domain/                    # Business logic (no external dependencies)
│   │   ├── entities/              # Core domain objects
│   │   │   ├── game.py            # Game entity with weather, opponent features
│   │   │   ├── ticket_sale.py     # Ticket sale with pricing tiers
│   │   │   ├── prediction.py      # Attendance prediction with intervals
│   │   │   ├── season_ticket_holder.py  # Holder with churn features
│   │   │   └── pricing_trajectory.py    # Multi-year pricing plans
│   │   └── services/              # Domain services
│   │       ├── attendance_forecasting_service.py  # Ensemble ML models
│   │       ├── churn_modeling_service.py          # Churn prediction
│   │       ├── price_optimization_service.py      # PuLP/scipy optimization
│   │       └── feature_engineering_service.py     # Feature calculations
│   │
│   ├── application/               # Use cases and ports
│   │   ├── use_cases/
│   │   │   ├── forecast_attendance_use_case.py
│   │   │   ├── optimize_pricing_use_case.py
│   │   │   └── analyze_churn_use_case.py
│   │   └── ports/
│   │       └── ports.py           # Repository interfaces
│   │
│   └── infrastructure/            # External adapters
│       ├── adapters/
│       │   ├── data_preprocessing_adapter.py  # R code replication
│       │   ├── prophet_adapter.py             # Facebook Prophet
│       │   └── weather_api_adapter.py         # OpenWeatherMap
│       └── repositories/
│           └── in_memory_repositories.py
│
├── config/
│   └── config.yaml                # System configuration
├── data/
│   ├── raw/                       # Original data files
│   ├── processed/                 # Preprocessed data
│   └── models/                    # Trained models
├── tests/
├── scripts/
├── main.py                        # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone or copy the repository
cd toledo_attendance_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Preprocess Data

```bash
# Process ticket sales data (replicates R preprocessing logic)
python main.py preprocess --input data/raw/TicketSales_PastData.csv data/raw/Ticket_Sales_by_Event__Henry_BB_26.csv
```

### 2. Train Models

```bash
# Train ensemble forecasting models
python main.py train --input data/processed/processed_ticket_sales.csv
```

### 3. Generate Predictions

```bash
# Generate attendance predictions
python main.py predict --model data/models/attendance_model.joblib
```

### 4. Optimize Pricing

```bash
# Optimize 5-year pricing trajectory
python main.py optimize --section "Lower Reserved" --price 40 --method scipy
```

### 5. Analyze Churn

```bash
# Analyze season ticket holder churn
python main.py churn
```

## Core Features

### Ensemble Attendance Forecasting

The system uses an optimized ensemble of four models:

| Model | Default Weight | Strengths |
|-------|----------------|-----------|
| XGBoost | 35% | Non-linear patterns, feature interactions |
| Random Forest | 35% | Robustness, feature importance |
| Bayesian Ridge | 20% | Uncertainty quantification |
| Prophet | 10% | Seasonality, trend decomposition |

Weights are optimized via Leave-One-Season-Out cross-validation.

### Multi-Year Price Optimization

Optimizes ticket prices over a 5-year horizon with constraints:
- **Inflation Floor**: Minimum 3% annual increase
- **Churn Ceiling**: Maximum 15% annual increase (to limit churn)
- **Churn Elasticity**: Models churn response to price increases

```python
from src.domain.services.price_optimization_service import PriceOptimizationService
from src.domain.entities.pricing_trajectory import SeatingSection, OptimizationConstraints

optimizer = PriceOptimizationService()

constraints = OptimizationConstraints(
    min_annual_increase=0.00,
    max_annual_increase=0.15,  # Churn constraint
    inflation_floor=0.03,
    max_churn_rate=0.10,
)

trajectory = optimizer.create_pricing_trajectory(
    section=SeatingSection.LOWER_RESERVED,
    current_price=40.0,
    current_season=2025,
    planning_years=5,
    constraints=constraints,
)
```

### Churn Modeling

Predicts season ticket holder churn using gradient boosting with:
- Tenure-based features
- Attendance patterns
- Revenue trends
- Seat changes (upgrades/downgrades)

```python
from src.domain.services.churn_modeling_service import ChurnModelingService

churn_service = ChurnModelingService()

# Estimate churn from price increase
churn_rate = churn_service.estimate_churn_from_price_increase(0.10)  # 10% increase
print(f"Expected churn: {churn_rate:.1%}")
```

### Feature Engineering

Automatically calculates:
- **Weather Comfort Index**: 0-100 scale based on temperature, wind, precipitation
- **Opponent Strength Score**: Power 5 = 1.0, MAC = 0.6, FCS = 0.3
- **Game Importance Factor**: Rivalry +0.4, Homecoming +0.3, Senior Day +0.2
- **Seat Value Index**: Club = 1.5, Loge = 1.3, Lower = 1.1, Upper = 0.9

## Data Preprocessing

The DataPreprocessingAdapter replicates the original R code logic:

```r
# Original R code (replicated in Python)
filter(!str_detect(class_name, "Parking|Comp"), event_pmt != 0)
mutate(
    order_qty = as.numeric(gsub(",", "", order_qty)),
    new_tickets = ifelse(str_detect(class_name, "(?i)New"), 1, 0),
    alumni_tickets = ifelse(str_detect(price_type, "(?i)Alumni"), 1, 0),
    class_category = ifelse(str_detect(class_name, " - "),
                            str_extract(class_name, "(?<= - ).*"), "Other")
)
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model parameters
models:
  xgboost:
    max_depth: 3
    learning_rate: 0.03
    n_estimators: 200

# Optimization constraints
optimization:
  max_annual_increase: 0.15
  inflation_floor: 0.03
  max_acceptable_churn: 0.10

# Churn thresholds
churn:
  high_risk_threshold: 0.7
  price_increase_threshold: 0.05
```

## API Integration

### Weather API (OpenWeatherMap)

```python
from src.infrastructure.adapters.weather_api_adapter import WeatherAPIAdapter

weather = WeatherAPIAdapter(api_key="your_key")
conditions = weather.get_game_day_conditions(game_date, kickoff_hour=15)
print(f"Comfort Index: {conditions['comfort_index']}")
print(f"Attendance Impact: {conditions['attendance_impact']:.1%}")
```

### Prophet for Seasonality

```python
from src.infrastructure.adapters.prophet_adapter import ProphetAdapter

prophet = ProphetAdapter()
prophet.fit(dates, attendance, regressors=['temperature', 'opponent_tier'])
forecast = prophet.predict(future_dates)
```

## Performance Expectations

Based on limited historical data (60-70 games):

| Metric | Initial Target | Mature Target |
|--------|----------------|---------------|
| MAPE | 15-20% | 12-15% |
| MAE | 1,500-2,000 | 1,200-1,500 |
| CI Coverage | 80% | 85% |

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Files from Knowledge Base

The system is designed to work with:
- `TicketSales_PastData.csv` - Historical ticket sales (16,070 rows)
- `Ticket_Sales_by_Event__Henry_BB_26.csv` - Current season (5,068 rows)
- `Scan_rate_FB25.xlsx` - Football scan rate data
- `code_for_Ahmad.R` - Original R preprocessing (replicated in Python)

## Key Constraints

1. **Small Dataset**: ~60-70 games limits model complexity
2. **Churn Sensitivity**: Price increases >5% trigger accelerated churn
3. **Weather Dependency**: Outdoor events highly weather-sensitive
4. **Regime Changes**: Coach Candle era (2016+) may differ from earlier patterns

## Contributing

1. Follow hexagonal architecture patterns
2. Add tests for new features
3. Update configuration as needed
4. Document any new dependencies

## License

University of Toledo Athletics Department - Internal Use Only
