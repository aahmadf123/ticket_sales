"""Churn modeling service for season ticket holder retention analysis."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from ..entities.season_ticket_holder import SeasonTicketHolder, HolderStatus


@dataclass
class ChurnModelConfig:
    """Configuration for churn prediction model."""
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 4
    min_samples_split: int = 10
    class_weight: str = "balanced"
    
    # Thresholds
    churn_probability_threshold: float = 0.5
    high_risk_threshold: float = 0.7
    
    # Cross-validation
    cv_folds: int = 5


class ChurnModelingService:
    """Service for predicting and analyzing season ticket holder churn."""
    
    def __init__(self, config: Optional[ChurnModelConfig] = None):
        self.config = config or ChurnModelConfig()
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train churn prediction model.
        
        Args:
            X: Feature matrix
            y: Target labels (1=churned, 0=retained)
            feature_names: Names of features
        
        Returns:
            Dictionary with training metrics
        """
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.feature_names = feature_names
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=42,
        )
        
        self.model.fit(X, y)
        
        # Cross-validation
        cv_metrics = self._cross_validate(X, y)
        
        # Training metrics
        train_pred = self.model.predict(X)
        train_proba = self.model.predict_proba(X)[:, 1]
        
        self.training_metrics = {
            "n_samples": len(y),
            "churn_rate": np.mean(y),
            "train_auc": roc_auc_score(y, train_proba),
            "train_precision": precision_score(y, train_pred),
            "train_recall": recall_score(y, train_pred),
            "train_f1": f1_score(y, train_pred),
            "cv_auc": cv_metrics["auc"],
            "cv_precision": cv_metrics["precision"],
            "cv_recall": cv_metrics["recall"],
            "cv_f1": cv_metrics["f1"],
            "feature_importance": dict(zip(
                feature_names,
                self.model.feature_importances_
            )),
        }
        
        self.is_trained = True
        
        return self.training_metrics
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Perform cross-validation."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=42
        )
        
        aucs, precisions, recalls, f1s = [], [], [], []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=42,
            )
            
            model.fit(X_train, y_train)
            
            pred = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]
            
            aucs.append(roc_auc_score(y_val, proba))
            precisions.append(precision_score(y_val, pred))
            recalls.append(recall_score(y_val, pred))
            f1s.append(f1_score(y_val, pred))
        
        return {
            "auc": np.mean(aucs),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s),
        }
    
    def predict_churn_probability(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict churn probability for holders.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of churn probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_churn(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict churn (binary) for holders.
        
        Args:
            X: Feature matrix
            threshold: Optional custom threshold
        
        Returns:
            Array of binary predictions
        """
        if threshold is None:
            threshold = self.config.churn_probability_threshold
        
        probabilities = self.predict_churn_probability(X)
        return (probabilities >= threshold).astype(int)
    
    def classify_risk_level(
        self,
        churn_probability: float
    ) -> str:
        """Classify holder into risk level.
        
        Args:
            churn_probability: Predicted churn probability
        
        Returns:
            Risk level string
        """
        if churn_probability >= self.config.high_risk_threshold:
            return "high_risk"
        elif churn_probability >= self.config.churn_probability_threshold:
            return "medium_risk"
        elif churn_probability >= 0.3:
            return "low_risk"
        else:
            return "safe"
    
    def analyze_holder(
        self,
        holder: SeasonTicketHolder
    ) -> Dict[str, Any]:
        """Analyze single holder's churn risk.
        
        Args:
            holder: SeasonTicketHolder entity
        
        Returns:
            Analysis dictionary
        """
        features = holder.calculate_churn_features()
        X = np.array([[features[name] for name in self.feature_names]])
        
        churn_prob = self.predict_churn_probability(X)[0]
        risk_level = self.classify_risk_level(churn_prob)
        
        # Calculate retention value
        holder.churn_probability = churn_prob
        holder.retention_probability = 1 - churn_prob
        retention_value = holder.calculate_retention_value()
        
        # Identify top risk factors
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Find which features are driving the risk
            risk_factors = []
            for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
                value = features.get(name, 0)
                risk_factors.append({
                    "feature": name,
                    "importance": imp,
                    "value": value,
                })
        else:
            risk_factors = []
        
        return {
            "holder_id": holder.holder_id,
            "churn_probability": churn_prob,
            "retention_probability": 1 - churn_prob,
            "risk_level": risk_level,
            "retention_value": retention_value,
            "risk_factors": risk_factors,
            "recommended_actions": self._get_retention_recommendations(
                holder, risk_level, risk_factors
            ),
        }
    
    def _get_retention_recommendations(
        self,
        holder: SeasonTicketHolder,
        risk_level: str,
        risk_factors: List[Dict]
    ) -> List[str]:
        """Generate retention recommendations based on analysis.
        
        Args:
            holder: SeasonTicketHolder entity
            risk_level: Classified risk level
            risk_factors: Top risk factors
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        if risk_level in ["high_risk", "medium_risk"]:
            recommendations.append("Initiate personal outreach call")
            recommendations.append("Offer renewal incentive package")
        
        # Check specific risk factors
        for factor in risk_factors:
            feature = factor["feature"]
            value = factor["value"]
            
            if feature == "attendance_rate" and value < 0.5:
                recommendations.append(
                    "Low attendance rate detected - offer seat relocation options"
                )
            
            if feature == "revenue_change_pct" and value < -20:
                recommendations.append(
                    "Significant revenue decline - review pricing and offer flexible payment"
                )
            
            if feature == "downgrades_count" and value > 0:
                recommendations.append(
                    "Recent downgrade detected - survey for satisfaction issues"
                )
            
            if feature == "tenure_category_new" and value == 1:
                recommendations.append(
                    "New holder at risk - assign dedicated account manager"
                )
        
        if holder.seats_removed_count > 0:
            recommendations.append(
                "Seats removed recently - offer incentives for seat recovery"
            )
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def calculate_portfolio_metrics(
        self,
        holders: List[SeasonTicketHolder]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level churn metrics.
        
        Args:
            holders: List of SeasonTicketHolder entities
        
        Returns:
            Portfolio metrics dictionary
        """
        if not holders:
            return {}
        
        # Get features for all holders
        feature_list = [h.calculate_churn_features() for h in holders]
        X = np.array([[f[name] for name in self.feature_names] for f in feature_list])
        
        # Predict churn probabilities
        churn_probs = self.predict_churn_probability(X)
        
        # Risk distribution
        risk_levels = [self.classify_risk_level(p) for p in churn_probs]
        risk_distribution = {
            "high_risk": risk_levels.count("high_risk"),
            "medium_risk": risk_levels.count("medium_risk"),
            "low_risk": risk_levels.count("low_risk"),
            "safe": risk_levels.count("safe"),
        }
        
        # Calculate total at-risk revenue
        at_risk_revenue = sum(
            float(h.current_season_revenue) * p
            for h, p in zip(holders, churn_probs)
        )
        
        # Calculate retention value
        total_retention_value = 0
        for h, p in zip(holders, churn_probs):
            h.churn_probability = p
            h.retention_probability = 1 - p
            total_retention_value += h.calculate_retention_value()
        
        return {
            "total_holders": len(holders),
            "expected_churn_count": sum(churn_probs),
            "expected_churn_rate": np.mean(churn_probs),
            "risk_distribution": risk_distribution,
            "at_risk_revenue": at_risk_revenue,
            "total_retention_value": total_retention_value,
            "avg_churn_probability": np.mean(churn_probs),
            "std_churn_probability": np.std(churn_probs),
        }
    
    def estimate_churn_from_price_increase(
        self,
        current_churn_rate: float,
        price_increase_pct: float,
        price_elasticity: float = 2.0
    ) -> float:
        """Estimate churn rate after price increase.
        
        Args:
            current_churn_rate: Current baseline churn rate
            price_increase_pct: Proposed price increase (e.g., 0.10 for 10%)
            price_elasticity: How sensitive churn is to price changes
        
        Returns:
            Estimated new churn rate
        """
        # Threshold before churn increases significantly
        threshold = 0.05  # 5%
        
        if price_increase_pct <= threshold:
            return current_churn_rate
        
        # Incremental churn from price increase
        excess_increase = price_increase_pct - threshold
        incremental_churn = excess_increase * price_elasticity
        
        new_churn_rate = current_churn_rate + incremental_churn
        
        return min(new_churn_rate, 0.50)  # Cap at 50%
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if not self.is_trained:
            return {}
        
        return dict(sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: -x[1]
        ))
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        import joblib
        
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "config": self.config,
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.config = model_data["config"]
        self.is_trained = True
