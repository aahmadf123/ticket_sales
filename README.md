# Toledo Ticket Sales Forecasting System

An AI-driven machine learning system for predicting ticket revenue/demand and optimizing ticket pricing for University of Toledo Athletics.

## Overview

This system provides:
- **Revenue/Demand Forecasting**: Ensemble ML models (XGBoost, Random Forest, Bayesian Ridge, Prophet) to predict ticket revenue and demand
- **Price Optimization**: Multi-year ticket pricing trajectory optimization with churn constraints (using PuLP and scipy)
- **Churn Modeling**: Season ticket holder retention analysis and risk prediction
- **Feature Engineering**: Automated feature extraction from ticket sales data

## Architecture

The system follows **hexagonal architecture** principles with clear separation of concerns:

```
ticket_sales/
├── src/
│   ├── domain/                    # Business logic (no external dependencies)
│   │   ├── entities/              # Core domain objects
│   │   │   ├── game.py            # Game entity with features
│   │   │   ├── ticket_sale.py     # Ticket sale with pricing tiers
│   │   │   ├── prediction.py      # Revenue/demand prediction with intervals
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
cd ticket_sales

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

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
# Train ensemble forecasting models with hyperparameter tuning
python main.py train --input data/processed/processed_ticket_sales.csv
```

**Note:** Training includes automatic hyperparameter tuning using RandomizedSearchCV. No preset hyperparameters - the model finds optimal values for your data.

### 3. Generate Predictions

```bash
# Generate revenue/demand predictions
python main.py predict --model data/models/attendance_model.joblib
```

### 4. Optimize Pricing

```bash
# Optimize 5-year pricing trajectory (uses actual prices from data)
python main.py optimize --method scipy

# Or use PuLP for linear programming approach
python main.py optimize --method pulp
```

**Note:** Prices come from your ticket sales data, not hardcoded values. The optimization runs on ALL ticket types together.

### 5. Analyze Churn

```bash
# Analyze season ticket holder churn
python main.py churn
```

## Core Features

### Ensemble Revenue/Demand Forecasting

The system uses an optimized ensemble of four models:

| Model | Default Weight | Strengths |
|-------|----------------|-----------|
| XGBoost | 35% | Non-linear patterns, feature interactions |
| Random Forest | 35% | Robustness, feature importance |
| Bayesian Ridge | 20% | Uncertainty quantification |
| Prophet | 10% | Seasonality, trend decomposition |

Weights are optimized via Leave-One-Season-Out cross-validation.

### Multi-Year Price Optimization (PuLP & scipy)

Optimizes ticket prices over a 5-year horizon with constraints:
- **Inflation (3%)**: This is the FLOOR - at 3%, you're just keeping up with inflation (NO real gain)
- **Churn Ceiling**: Maximum 15% annual increase (to limit churn)
- **Churn Elasticity**: Models churn response to price increases above 5% threshold

**Key Understanding**: 3% inflation means if you only increase prices by 3% each year, you have ZERO real revenue growth. Real gains require beating inflation.

```python
import pandas as pd
from src.domain.services.price_optimization_service import PriceOptimizationService

# Load your actual ticket data
ticket_df = pd.read_csv("data/processed/processed_ticket_sales.csv")

# Optimizer extracts prices from data - no hardcoding needed
optimizer = PriceOptimizationService()

# Optimize ALL ticket types together using actual prices from data
results = optimizer.optimize_all_tickets(ticket_df)

print(f"Current avg price (from data): ${results['current_data_summary']['avg_price']:.2f}")
print(f"Optimal year 5 price: ${results['optimal_prices'][-1]:.2f}")
print(f"Beats inflation: {results['beats_inflation']}")
```

### Churn Modeling with Constraints

Predicts season ticket holder churn using gradient boosting with:
- Tenure-based features
- Revenue trends
- Seat changes (upgrades/downgrades)
- Price increase sensitivity

```python
from src.domain.services.churn_modeling_service import ChurnModelingService

churn_service = ChurnModelingService()

# Estimate churn from price increase
churn_rate = churn_service.estimate_churn_from_price_increase(0.05, 0.10)  # baseline 5%, increase 10%
print(f"Expected churn: {churn_rate:.1%}")
```

### Facebook Prophet for Seasonality

```python
from src.infrastructure.adapters.prophet_adapter import ProphetAdapter

prophet = ProphetAdapter()
prophet.fit(df, yearly_seasonality=True, regressors=['price_level'])
forecast = prophet.predict(periods=10)
```

### Feature Engineering

Automatically calculates:
- **Seat Value Index**: Club = 1.5, Loge = 1.3, Lower = 1.1, Upper = 0.9
- **Ticket Composition**: Season ticket %, new buyer %
- **Pricing Metrics**: Revenue per ticket, average unit price
- **Temporal Features**: Game sequence, historical averages

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
# Hyperparameter tuning (model finds optimal params)
models:
  tuning_iterations: 50  # RandomizedSearchCV iterations
  cv_folds: 5  # Cross-validation folds

# Optimization constraints
optimization:
  max_annual_increase: 0.15  # Churn constraint
  inflation_floor: 0.03  # 3% = NO REAL GAIN, just inflation
  max_acceptable_churn: 0.10
  churn_threshold: 0.05  # Above this, churn accelerates

# Churn thresholds
churn:
  high_risk_threshold: 0.7
  price_increase_threshold: 0.05
```

**Important**: The `inflation_floor` of 3% is NOT a target - it's just keeping up with inflation. Real revenue growth requires price increases ABOVE 3%.

## Project Data Files

The system is designed to work with:
- `TicketSales_PastData.csv` - Historical ticket sales (16,070 rows)
- `Ticket_Sales_by_Event__Henry_BB_26.csv` - Current season (5,068 rows)

## Key Constraints

1. **Small Dataset**: ~60-70 events limits model complexity
2. **Churn Sensitivity**: Price increases >5% trigger accelerated churn
3. **Regime Changes**: Coach Candle era (2016+) may differ from earlier patterns

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## ML Methods Used

1. **XGBoost**: Gradient boosting for non-linear patterns
2. **Random Forest**: Ensemble of decision trees for robustness
3. **Bayesian Ridge Regression**: Probabilistic predictions with uncertainty
4. **Facebook Prophet**: Time series with seasonality decomposition
5. **PuLP**: Linear/Mixed-Integer Programming for price optimization
6. **scipy.optimize**: Non-linear optimization for pricing trajectories

## Contributing

1. Follow hexagonal architecture patterns
2. Add tests for new features
3. Update configuration as needed
4. Document any new dependencies

## License

University of Toledo Athletics Department - Internal Use Only
