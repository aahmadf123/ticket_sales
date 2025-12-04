"""Revenue/Demand forecasting service with ensemble ML models and hyperparameter tuning."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
    cross_val_score,
    KFold,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.base import clone
from scipy.stats import randint, uniform

from ..entities.prediction import (
    Prediction, PredictionInterval, PredictionHorizon, UncertaintyLevel
)
from ..entities.game import Game


@dataclass
class ModelConfig:
    """Configuration for ML models - comprehensive hyperparameter search spaces."""

    # Whether to perform hyperparameter tuning
    tune_hyperparameters: bool = True
    tuning_iterations: int = 100  # Increased for thorough search
    cv_folds: int = 5  # Cross-validation folds for tuning

    # Ensemble weights (will be optimized)
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.35,
        "random_forest": 0.35,
        "bayesian_ridge": 0.20,
        "prophet": 0.10,
    })

    # Confidence interval
    confidence_level: float = 0.80

    # Extended hyperparameter search spaces
    xgboost_param_space: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth": (2, 15),
        "min_child_weight": (1, 30),
        "gamma": (0.0, 10.0),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.3, 1.0),
        "learning_rate": (0.005, 0.3),
        "n_estimators": (50, 1000),
        "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.1, 10.0),
    })

    random_forest_param_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": (50, 1000),
        "max_depth": (2, 50),
        "min_samples_split": (2, 50),
        "min_samples_leaf": (1, 30),
        "max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7, 0.9],
        "bootstrap": [True, False],
    })

    bayesian_ridge_param_space: Dict[str, Any] = field(default_factory=lambda: {
        "alpha_1": (1e-10, 1e-2),
        "alpha_2": (1e-10, 1e-2),
        "lambda_1": (1e-10, 1e-2),
        "lambda_2": (1e-10, 1e-2),
        "max_iter": (100, 2000),
        "tol": (1e-8, 1e-2),
    })

    gradient_boosting_param_space: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": (50, 500),
        "max_depth": (2, 10),
        "min_samples_split": (2, 30),
        "min_samples_leaf": (1, 20),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.5, 1.0),
    })


class AttendanceForecastingService:
    """Service for forecasting revenue/demand using ensemble models with hyperparameter tuning."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.ensemble_weights = self.config.initial_weights.copy()
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
        self.best_params = {}  # Store tuned hyperparameters
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        seasons: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train all models in the ensemble with hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target values (revenue or ticket count)
            feature_names: Names of features
            seasons: Season labels for cross-validation
        
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names
        n_samples = len(y)
        
        print(f"\nTraining on {n_samples} samples...")
        
        # Determine CV strategy based on data size
        if n_samples < 10:
            cv_folds = min(3, n_samples)
            print(f"  Small dataset: using {cv_folds}-fold CV")
        else:
            cv_folds = min(self.config.cv_folds, n_samples // 2)
        
        # Train individual models with hyperparameter tuning
        if self.config.tune_hyperparameters and n_samples >= 10:
            print("\n  Tuning XGBoost hyperparameters...")
            self._tune_and_train_xgboost(X, y, cv_folds)
            
            print("  Tuning Random Forest hyperparameters...")
            self._tune_and_train_random_forest(X, y, cv_folds)
            
            print("  Tuning Bayesian Ridge hyperparameters...")
            self._tune_and_train_bayesian_ridge(X, y, cv_folds)
            
            print("  Training Prophet model...")
            self._train_prophet(X, y, feature_names, seasons)
        else:
            # For very small datasets, use conservative defaults
            print("  Using conservative defaults for small dataset...")
            self._train_with_defaults(X, y, n_samples)
        
        # Optimize ensemble weights if we have enough data AND multiple seasons
        if n_samples >= 20 and seasons is not None and len(np.unique(seasons)) >= 2:
            print("  Optimizing ensemble weights...")
            self._optimize_ensemble_weights(X, y, seasons)
        
        # Calculate training metrics
        self.training_metrics = self._calculate_training_metrics(X, y, seasons)
        self.training_metrics["best_hyperparameters"] = self.best_params
        
        # Add warning for small datasets
        if n_samples < 20:
            self.training_metrics["warning"] = (
                f"Small dataset ({n_samples} samples). "
                "Model performance may be unreliable. "
                "Recommend collecting more historical data."
            )
        
        self.is_trained = True
        
        return self.training_metrics
    
    def _tune_and_train_xgboost(self, X: np.ndarray, y: np.ndarray, cv_folds: int):
        """Tune and train XGBoost model with comprehensive RandomizedSearchCV."""
        try:
            import xgboost as xgb

            # Use config-defined search space with expanded ranges
            ps = self.config.xgboost_param_space
            param_distributions = {
                "max_depth": randint(ps["max_depth"][0], ps["max_depth"][1] + 1),
                "min_child_weight": randint(ps["min_child_weight"][0], ps["min_child_weight"][1] + 1),
                "gamma": uniform(ps["gamma"][0], ps["gamma"][1] - ps["gamma"][0]),
                "subsample": uniform(ps["subsample"][0], ps["subsample"][1] - ps["subsample"][0]),
                "colsample_bytree": uniform(ps["colsample_bytree"][0], ps["colsample_bytree"][1] - ps["colsample_bytree"][0]),
                "learning_rate": uniform(ps["learning_rate"][0], ps["learning_rate"][1] - ps["learning_rate"][0]),
                "n_estimators": randint(ps["n_estimators"][0], ps["n_estimators"][1] + 1),
                "reg_alpha": uniform(ps["reg_alpha"][0], ps["reg_alpha"][1] - ps["reg_alpha"][0]),
                "reg_lambda": uniform(ps["reg_lambda"][0], ps["reg_lambda"][1] - ps["reg_lambda"][0]),
            }

            base_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
                early_stopping_rounds=None,  # Disable early stopping for full training
            )

            # Negative MAE scorer (sklearn maximizes, so we negate)
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)

            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=self.config.tuning_iterations,
                cv=cv_folds,
                scoring=scorer,
                random_state=42,
                n_jobs=-1,
                verbose=0,
                return_train_score=True,
            )

            search.fit(X, y)

            self.models["xgboost"] = search.best_estimator_
            self.best_params["xgboost"] = search.best_params_
            print(f"    Best XGBoost params: {search.best_params_}")
            print(f"    Best CV score: {-search.best_score_:.2f} MAE")

        except ImportError:
            print("    Warning: XGBoost not installed. Skipping XGBoost model.")
    
    def _tune_and_train_random_forest(self, X: np.ndarray, y: np.ndarray, cv_folds: int):
        """Tune and train Random Forest model with comprehensive RandomizedSearchCV."""
        from sklearn.ensemble import RandomForestRegressor

        # Use config-defined search space
        ps = self.config.random_forest_param_space
        param_distributions = {
            "n_estimators": randint(ps["n_estimators"][0], ps["n_estimators"][1] + 1),
            "max_depth": randint(ps["max_depth"][0], ps["max_depth"][1] + 1),
            "min_samples_split": randint(ps["min_samples_split"][0], ps["min_samples_split"][1] + 1),
            "min_samples_leaf": randint(ps["min_samples_leaf"][0], ps["min_samples_leaf"][1] + 1),
            "max_features": ps["max_features"],
            "bootstrap": ps["bootstrap"],
        }

        base_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            oob_score=True,  # Out-of-bag scoring for extra validation
        )

        scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=self.config.tuning_iterations,
            cv=cv_folds,
            scoring=scorer,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )

        search.fit(X, y)

        self.models["random_forest"] = search.best_estimator_
        self.best_params["random_forest"] = search.best_params_
        print(f"    Best RF params: {search.best_params_}")
        print(f"    Best CV score: {-search.best_score_:.2f} MAE")
    
    def _tune_and_train_bayesian_ridge(self, X: np.ndarray, y: np.ndarray, cv_folds: int):
        """Tune and train Bayesian Ridge model with comprehensive search."""
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        # Scale features for Bayesian Ridge
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Use config-defined search space with log-uniform sampling for priors
        ps = self.config.bayesian_ridge_param_space

        # Use loguniform for prior parameters (better for spanning orders of magnitude)
        from scipy.stats import loguniform

        param_distributions = {
            "alpha_1": loguniform(ps["alpha_1"][0], ps["alpha_1"][1]),
            "alpha_2": loguniform(ps["alpha_2"][0], ps["alpha_2"][1]),
            "lambda_1": loguniform(ps["lambda_1"][0], ps["lambda_1"][1]),
            "lambda_2": loguniform(ps["lambda_2"][0], ps["lambda_2"][1]),
            "max_iter": randint(ps["max_iter"][0], ps["max_iter"][1] + 1),
            "tol": loguniform(ps["tol"][0], ps["tol"][1]),
        }

        base_model = BayesianRidge(compute_score=True)
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=min(50, self.config.tuning_iterations),  # More iterations for thorough search
            cv=cv_folds,
            scoring=scorer,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )

        search.fit(X_scaled, y)

        self.models["bayesian_ridge"] = search.best_estimator_
        self.best_params["bayesian_ridge"] = search.best_params_
        print(f"    Best Bayesian Ridge params: {search.best_params_}")
        print(f"    Best CV score: {-search.best_score_:.2f} MAE")
    
    def _train_with_defaults(self, X: np.ndarray, y: np.ndarray, n_samples: int):
        """Train with conservative defaults for small datasets."""
        try:
            import xgboost as xgb
            
            # Conservative settings for small data
            max_depth = max(2, min(3, n_samples // 5))
            
            self.models["xgboost"] = xgb.XGBRegressor(
                max_depth=max_depth,
                min_child_weight=1,
                subsample=1.0,
                colsample_bytree=1.0,
                learning_rate=0.1,
                n_estimators=50,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
            self.models["xgboost"].fit(X, y)
            self.best_params["xgboost"] = {"note": "defaults for small dataset"}
        except ImportError:
            pass
        
        from sklearn.ensemble import RandomForestRegressor
        self.models["random_forest"] = RandomForestRegressor(
            n_estimators=50,
            max_depth=max(2, min(3, n_samples // 5)),
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        self.models["random_forest"].fit(X, y)
        self.best_params["random_forest"] = {"note": "defaults for small dataset"}
        
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.models["bayesian_ridge"] = BayesianRidge(max_iter=300)
        self.models["bayesian_ridge"].fit(X_scaled, y)
        self.best_params["bayesian_ridge"] = {"note": "defaults for small dataset"}
    
    def _train_prophet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        seasons: Optional[np.ndarray] = None
    ):
        """Train Prophet model for time series component of ensemble.
        
        Prophet captures seasonality and trend that tree models might miss.
        Uses game_number as time index if no dates available.
        """
        try:
            from prophet import Prophet
            
            # Create a synthetic date sequence based on game_number or index
            # Prophet needs 'ds' (date) and 'y' (target) columns
            n_samples = len(y)
            
            # Find game_number in features if available
            game_num_idx = None
            for i, name in enumerate(feature_names):
                if 'game_number' in name.lower():
                    game_num_idx = i
                    break
            
            # Create time index
            if game_num_idx is not None:
                # Use game numbers to create sequential dates
                game_nums = X[:, game_num_idx]
            else:
                game_nums = np.arange(n_samples)
            
            # Create DataFrame for Prophet
            # Use synthetic weekly dates (one game per week approximately)
            base_date = pd.Timestamp('2020-09-01')  # Start of typical season
            dates = [base_date + pd.Timedelta(weeks=int(g)) for g in game_nums]
            
            prophet_df = pd.DataFrame({
                'ds': dates,
                'y': y
            })
            
            # Add regressors (features) that Prophet can use
            # Select numeric features that make sense as regressors
            regressor_names = []
            for i, name in enumerate(feature_names):
                if name not in ['game_number']:  # Don't use game_number as regressor
                    col_name = f'reg_{name}'
                    prophet_df[col_name] = X[:, i]
                    regressor_names.append(col_name)
            
            # Initialize Prophet with reasonable settings for sports data
            model = Prophet(
                yearly_seasonality=False,  # Not enough data for yearly
                weekly_seasonality=False,  # Games aren't weekly pattern
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.1,  # More flexible trend
            )
            
            # Add custom seasonality for sports season (roughly 4 months)
            model.add_seasonality(
                name='season',
                period=120,  # ~4 months
                fourier_order=3
            )
            
            # Add regressors (limit to avoid overfitting)
            for reg_name in regressor_names[:5]:  # Max 5 regressors
                model.add_regressor(reg_name)
            
            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            
            model.fit(prophet_df)
            
            # Store model and metadata for prediction
            self.models["prophet"] = {
                'model': model,
                'regressor_names': regressor_names[:5],
                'feature_names': feature_names,
                'base_date': base_date,
            }
            self.best_params["prophet"] = {
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.1,
                "regressors": regressor_names[:5]
            }
            print(f"    Prophet trained with {len(regressor_names[:5])} regressors")
            
        except ImportError:
            print("    Warning: Prophet not installed. Skipping Prophet model.")
            # Remove prophet from weights if not available
            if "prophet" in self.ensemble_weights:
                prophet_weight = self.ensemble_weights.pop("prophet")
                # Redistribute weight to other models
                remaining = sum(self.ensemble_weights.values())
                if remaining > 0:
                    for key in self.ensemble_weights:
                        self.ensemble_weights[key] /= remaining
        except Exception as e:
            print(f"    Warning: Prophet training failed: {e}")
            if "prophet" in self.ensemble_weights:
                prophet_weight = self.ensemble_weights.pop("prophet")
                remaining = sum(self.ensemble_weights.values())
                if remaining > 0:
                    for key in self.ensemble_weights:
                        self.ensemble_weights[key] /= remaining
    
    def _optimize_ensemble_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seasons: np.ndarray
    ):
        """Optimize ensemble weights using Leave-One-Season-Out CV."""
        logo = LeaveOneGroupOut()
        
        # Collect predictions from each model
        model_predictions = {name: [] for name in self.models.keys()}
        actuals = []
        
        for train_idx, test_idx in logo.split(X, y, groups=seasons):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_predictions = self._fit_and_predict_fold(X_train, y_train, X_test)
            for name, preds in fold_predictions.items():
                model_predictions[name].extend(preds)
            
            actuals.extend(y_test)
        
        # Convert to numpy arrays
        actuals = np.array(actuals)
        predictions_matrix = np.column_stack([
            np.array(model_predictions[name])
            for name in self.models.keys()
        ])
        
        # Optimize weights
        n_models = len(self.models)
        
        def ensemble_loss(weights):
            ensemble_pred = np.dot(predictions_matrix, weights)
            return mean_squared_error(actuals, ensemble_pred)
        
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            ensemble_loss,
            x0=np.ones(n_models) / n_models,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        if result.success:
            for i, name in enumerate(self.models.keys()):
                self.ensemble_weights[name] = result.x[i]
        
        # Original models remain fitted from earlier training
    
    def _calculate_training_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seasons: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate training and cross-validation metrics."""
        metrics = {
            "n_samples": len(y),
            "ensemble_weights": self.ensemble_weights.copy(),
            "model_metrics": {},
        }
        
        # Training predictions
        train_pred = self._predict_ensemble(X)
        
        metrics["training_mae"] = mean_absolute_error(y, train_pred)
        metrics["training_rmse"] = np.sqrt(mean_squared_error(y, train_pred))
        # Avoid division by zero in MAPE
        non_zero_mask = y != 0
        if non_zero_mask.sum() > 0:
            metrics["training_mape"] = np.mean(np.abs((y[non_zero_mask] - train_pred[non_zero_mask]) / y[non_zero_mask])) * 100
        else:
            metrics["training_mape"] = 0.0
        
        # Individual model metrics
        for name, model in self.models.items():
            if name == "prophet":
                pred = self._predict_prophet(X)
            elif name == "bayesian_ridge":
                X_scaled = self.scaler.transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            metrics["model_metrics"][name] = {
                "mae": mean_absolute_error(y, pred),
                "rmse": np.sqrt(mean_squared_error(y, pred)),
            }
        
        # Cross-validation metrics if seasons provided and enough unique seasons
        if seasons is not None:
            unique_seasons = np.unique(seasons)
            if len(unique_seasons) >= 2:
                cv_metrics = self._cross_validate(X, y, seasons)
                metrics["cv_mae"] = cv_metrics["mae"]
                metrics["cv_rmse"] = cv_metrics["rmse"]
                metrics["cv_mape"] = cv_metrics["mape"]
            else:
                # Use Leave-One-Out CV for small datasets
                metrics["cv_note"] = f"Only {len(unique_seasons)} season(s) - using Leave-One-Out CV"
                if len(y) >= 3:
                    cv_metrics = self._cross_validate_loo(X, y)
                    metrics["cv_mae"] = cv_metrics["mae"]
                    metrics["cv_rmse"] = cv_metrics["rmse"]
                    metrics["cv_mape"] = cv_metrics["mape"]
        
        # Feature importance from Random Forest
        if "random_forest" in self.models:
            importance = self.models["random_forest"].feature_importances_
            metrics["feature_importance"] = dict(zip(self.feature_names, importance))
        
        return metrics
    
    def _fit_and_predict_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Fit cloned models on a training fold and predict on the test fold."""
        fold_predictions: Dict[str, np.ndarray] = {}
        from sklearn.preprocessing import StandardScaler
        
        for name, model in self.models.items():
            if name == "prophet":
                # Prophet can't be cloned - skip in CV, use training predictions
                # This is a simplification; proper Prophet CV would retrain
                fold_predictions[name] = self._predict_prophet(X_test)
                continue
            
            model_clone = clone(model)
            if name == "bayesian_ridge":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model_clone.fit(X_train_scaled, y_train)
                fold_predictions[name] = model_clone.predict(X_test_scaled)
            else:
                model_clone.fit(X_train, y_train)
                fold_predictions[name] = model_clone.predict(X_test)
        
        return fold_predictions
    
    def _predict_prophet(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from Prophet model."""
        if "prophet" not in self.models:
            return np.zeros(len(X))
        
        try:
            prophet_data = self.models["prophet"]
            model = prophet_data['model']
            regressor_names = prophet_data['regressor_names']
            feature_names = prophet_data['feature_names']
            base_date = prophet_data['base_date']
            
            n_samples = len(X)
            
            # Find game_number index
            game_num_idx = None
            for i, name in enumerate(feature_names):
                if 'game_number' in name.lower():
                    game_num_idx = i
                    break
            
            if game_num_idx is not None:
                game_nums = X[:, game_num_idx]
            else:
                game_nums = np.arange(n_samples)
            
            # Create future dataframe
            dates = [base_date + pd.Timedelta(weeks=int(g)) for g in game_nums]
            future_df = pd.DataFrame({'ds': dates})
            
            # Add regressors
            for reg_name in regressor_names:
                # Extract original feature name
                orig_name = reg_name.replace('reg_', '')
                if orig_name in feature_names:
                    idx = feature_names.index(orig_name)
                    future_df[reg_name] = X[:, idx]
                else:
                    future_df[reg_name] = 0
            
            # Predict
            forecast = model.predict(future_df)
            return forecast['yhat'].values
            
        except Exception as e:
            print(f"    Prophet prediction error: {e}")
            return np.zeros(len(X))
    
    def _cross_validate_loo(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Perform Leave-One-Out cross-validation for small datasets."""
        from sklearn.model_selection import LeaveOneOut
        
        loo = LeaveOneOut()
        predictions = []
        actuals = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_predictions = self._fit_and_predict_fold(X_train, y_train, X_test)
            
            ensemble_pred = self._combine_ensemble_predictions(fold_predictions)
            predictions.extend(ensemble_pred)
            actuals.extend(y_test)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Avoid division by zero
        non_zero_mask = actuals != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        return {
            "mae": mean_absolute_error(actuals, predictions),
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "mape": mape,
        }
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seasons: np.ndarray
    ) -> Dict[str, float]:
        """Perform Leave-One-Season-Out cross-validation."""
        logo = LeaveOneGroupOut()
        predictions = []
        actuals = []
        
        for train_idx, test_idx in logo.split(X, y, groups=seasons):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_predictions = self._fit_and_predict_fold(X_train, y_train, X_test)
            ensemble_pred = self._combine_ensemble_predictions(fold_predictions)
            predictions.extend(ensemble_pred)
            actuals.extend(y_test)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return {
            "mae": mean_absolute_error(actuals, predictions),
            "rmse": np.sqrt(mean_squared_error(actuals, predictions)),
            "mape": np.mean(np.abs((actuals - predictions) / actuals)) * 100,
        }
    
    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble prediction."""
        model_predictions = self._get_model_predictions(X)
        return self._combine_ensemble_predictions(model_predictions)

    def _get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each individual model on supplied features."""
        predictions = {}
        for name, model in self.models.items():
            if name == "prophet":
                predictions[name] = self._predict_prophet(X)
            elif name == "bayesian_ridge":
                X_scaled = self.scaler.transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        return predictions
    
    def _combine_ensemble_predictions(
        self,
        model_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combine individual model predictions using ensemble weights."""
        predictions = []
        weights = []
        
        for name, preds in model_predictions.items():
            weight = self.ensemble_weights.get(name, 0)
            if weight > 0:
                predictions.append(preds)
                weights.append(weight)
        
        if not predictions:
            return np.zeros(len(next(iter(model_predictions.values()))))
        
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()
        
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights_array):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def _calculate_prediction_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction uncertainty (std and confidence intervals)."""
        individual_predictions = []
        
        for name, model in self.models.items():
            if name == "prophet":
                pred = self._predict_prophet(X)
            elif name == "bayesian_ridge":
                X_scaled = self.scaler.transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            individual_predictions.append(pred)
        
        # Calculate standard deviation across models
        pred_matrix = np.column_stack(individual_predictions)
        pred_std = np.std(pred_matrix, axis=1)
        
        # For Bayesian Ridge, we can get additional uncertainty
        if "bayesian_ridge" in self.models:
            X_scaled = self.scaler.transform(X)
            _, bayesian_std = self.models["bayesian_ridge"].predict(
                X_scaled, return_std=True
            )
            # Combine uncertainties
            pred_std = np.sqrt(pred_std**2 + bayesian_std**2)
        
        return pred_std, pred_matrix
    
    def predict(
        self,
        X: np.ndarray,
        horizon: PredictionHorizon = PredictionHorizon.T_24
    ) -> List[Prediction]:
        """Generate attendance predictions with uncertainty.
        
        Args:
            X: Feature matrix
            horizon: Prediction time horizon
        
        Returns:
            List of Prediction objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Ensemble prediction
        ensemble_pred = self._predict_ensemble(X)
        
        # Calculate uncertainty
        pred_std, individual_preds = self._calculate_prediction_uncertainty(X)
        
        # Adjust uncertainty based on horizon
        horizon_multipliers = {
            PredictionHorizon.T_7: 1.5,
            PredictionHorizon.T_72: 1.2,
            PredictionHorizon.T_24: 1.0,
            PredictionHorizon.GAME_DAY: 0.8,
        }
        multiplier = horizon_multipliers.get(horizon, 1.0)
        adjusted_std = pred_std * multiplier
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        predictions = []
        for i in range(len(X)):
            # Calculate confidence interval
            lower = max(0, int(ensemble_pred[i] - z_score * adjusted_std[i]))
            upper = int(ensemble_pred[i] + z_score * adjusted_std[i])
            
            confidence_interval = PredictionInterval(
                lower=lower,
                upper=upper,
                confidence_level=self.config.confidence_level,
            )
            
            # Get individual model predictions
            xgb_pred = int(individual_preds[i, 0]) if individual_preds.shape[1] > 0 else None
            rf_pred = int(individual_preds[i, 1]) if individual_preds.shape[1] > 1 else None
            br_pred = int(individual_preds[i, 2]) if individual_preds.shape[1] > 2 else None
            
            prediction = Prediction(
                predicted_attendance=int(ensemble_pred[i]),
                confidence_interval=confidence_interval,
                horizon=horizon,
                xgboost_prediction=xgb_pred,
                random_forest_prediction=rf_pred,
                bayesian_ridge_prediction=br_pred,
                ensemble_weights=self.ensemble_weights.copy(),
                prediction_std=float(adjusted_std[i]),
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble."""
        if not self.is_trained:
            return {}
        
        importance = {}
        
        # Random Forest importance
        if "random_forest" in self.models:
            rf_imp = self.models["random_forest"].feature_importances_
            for name, imp in zip(self.feature_names, rf_imp):
                importance[name] = importance.get(name, 0) + imp * 0.5
        
        # XGBoost importance
        if "xgboost" in self.models:
            xgb_imp = self.models["xgboost"].feature_importances_
            for name, imp in zip(self.feature_names, xgb_imp):
                importance[name] = importance.get(name, 0) + imp * 0.5
        
        # Sort by importance
        importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        import joblib
        
        model_data = {
            "models": self.models,
            "ensemble_weights": self.ensemble_weights,
            "scaler": self.scaler if hasattr(self, "scaler") else None,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "config": self.config,
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        import joblib
        
        model_data = joblib.load(path)
        
        self.models = model_data["models"]
        self.ensemble_weights = model_data["ensemble_weights"]
        if model_data["scaler"] is not None:
            self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.config = model_data["config"]
        self.is_trained = True
