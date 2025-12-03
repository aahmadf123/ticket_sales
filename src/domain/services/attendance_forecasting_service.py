"""Attendance forecasting service with ensemble ML models."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.optimize import minimize
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..entities.prediction import (
    Prediction, PredictionInterval, PredictionHorizon, UncertaintyLevel
)
from ..entities.game import Game


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    # XGBoost parameters
    xgb_max_depth: int = 3
    xgb_min_child_weight: int = 7
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_learning_rate: float = 0.03
    xgb_n_estimators: int = 200
    xgb_reg_alpha: float = 2.0
    xgb_reg_lambda: float = 2.0
    
    # Random Forest parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 4
    rf_min_samples_split: int = 12
    rf_min_samples_leaf: int = 6
    rf_max_features: str = "sqrt"
    
    # Bayesian Ridge parameters
    bayesian_n_iter: int = 300
    bayesian_tol: float = 1e-3
    
    # Ensemble weights (initial)
    initial_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.35,
        "random_forest": 0.35,
        "bayesian_ridge": 0.20,
        "prophet": 0.10,
    })
    
    # Confidence interval
    confidence_level: float = 0.80


class AttendanceForecastingService:
    """Service for forecasting game attendance using ensemble models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.ensemble_weights = self.config.initial_weights.copy()
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        seasons: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target values (attendance)
            feature_names: Names of features
            seasons: Season labels for cross-validation
        
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names
        
        # Train individual models
        self._train_xgboost(X, y)
        self._train_random_forest(X, y)
        self._train_bayesian_ridge(X, y)
        
        # Optimize ensemble weights if we have enough data
        if len(y) >= 20 and seasons is not None:
            self._optimize_ensemble_weights(X, y, seasons)
        
        # Calculate training metrics
        self.training_metrics = self._calculate_training_metrics(X, y, seasons)
        
        self.is_trained = True
        
        return self.training_metrics
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            
            self.models["xgboost"] = xgb.XGBRegressor(
                max_depth=self.config.xgb_max_depth,
                min_child_weight=self.config.xgb_min_child_weight,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                learning_rate=self.config.xgb_learning_rate,
                n_estimators=self.config.xgb_n_estimators,
                reg_alpha=self.config.xgb_reg_alpha,
                reg_lambda=self.config.xgb_reg_lambda,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
            
            self.models["xgboost"].fit(X, y)
            
        except ImportError:
            print("Warning: XGBoost not installed. Skipping XGBoost model.")
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor
        
        self.models["random_forest"] = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            max_features=self.config.rf_max_features,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        
        self.models["random_forest"].fit(X, y)
    
    def _train_bayesian_ridge(self, X: np.ndarray, y: np.ndarray):
        """Train Bayesian Ridge model."""
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler
        
        # Scale features for Bayesian Ridge
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.models["bayesian_ridge"] = BayesianRidge(
            n_iter=self.config.bayesian_n_iter,
            tol=self.config.bayesian_tol,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
            compute_score=True,
        )
        
        self.models["bayesian_ridge"].fit(X_scaled, y)
    
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
            
            # Train each model on training fold
            for name, model in self.models.items():
                # Clone and retrain model
                if name == "bayesian_ridge":
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                
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
        
        # Retrain all models on full data
        self._train_xgboost(X, y)
        self._train_random_forest(X, y)
        self._train_bayesian_ridge(X, y)
    
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
        metrics["training_mape"] = np.mean(np.abs((y - train_pred) / y)) * 100
        
        # Individual model metrics
        for name, model in self.models.items():
            if name == "bayesian_ridge":
                X_scaled = self.scaler.transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            metrics["model_metrics"][name] = {
                "mae": mean_absolute_error(y, pred),
                "rmse": np.sqrt(mean_squared_error(y, pred)),
            }
        
        # Cross-validation metrics if seasons provided
        if seasons is not None:
            cv_metrics = self._cross_validate(X, y, seasons)
            metrics["cv_mae"] = cv_metrics["mae"]
            metrics["cv_rmse"] = cv_metrics["rmse"]
            metrics["cv_mape"] = cv_metrics["mape"]
        
        # Feature importance from Random Forest
        if "random_forest" in self.models:
            importance = self.models["random_forest"].feature_importances_
            metrics["feature_importance"] = dict(zip(self.feature_names, importance))
        
        return metrics
    
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
            
            # Train models
            self._train_xgboost(X_train, y_train)
            self._train_random_forest(X_train, y_train)
            self._train_bayesian_ridge(X_train, y_train)
            
            # Predict
            pred = self._predict_ensemble(X_test)
            predictions.extend(pred)
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
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 0)
            if weight > 0:
                if name == "bayesian_ridge":
                    X_scaled = self.scaler.transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions.append(pred)
                weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def _calculate_prediction_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction uncertainty (std and confidence intervals)."""
        individual_predictions = []
        
        for name, model in self.models.items():
            if name == "bayesian_ridge":
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
