"""
Classical machine learning predictors for MixConfig.

Includes XGBoost, Random Forest, and Linear models.
"""

from typing import Optional, Dict, Any, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error


class XGBoostPredictor:
    """
    XGBoost predictor wrapper for MixConfig experiments.

    Supports both classification and regression with hyperparameter tuning.

    Args:
        is_classifier: Whether classification task.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        random_state: Random seed.
        **kwargs: Additional XGBoost parameters.
    """

    def __init__(
        self,
        is_classifier: bool = False,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        self.is_classifier = is_classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self, n_classes: Optional[int] = None):
        """Create XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_jobs": -1,
            **self.kwargs,
        }

        if self.is_classifier:
            if n_classes and n_classes > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> "XGBoostPredictor":
        """
        Fit the XGBoost model.

        Args:
            X: Training features.
            y: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            normalize: Whether to standardize features.

        Returns:
            self
        """
        if normalize:
            X = self.scaler.fit_transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        n_classes = len(np.unique(y)) if self.is_classifier else None
        self._create_model(n_classes)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )

        return self

    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Predict on new data."""
        if normalize and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba only available for classifiers")
        if normalize and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, normalize: bool = True) -> float:
        """Compute score on data."""
        preds = self.predict(X, normalize)

        if self.is_classifier:
            return accuracy_score(y, preds)
        else:
            return r2_score(y, preds)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.feature_importances_


class RandomForestPredictor:
    """
    Random Forest predictor wrapper for MixConfig experiments.

    Args:
        is_classifier: Whether classification task.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_state: Random seed.
        **kwargs: Additional sklearn RF parameters.
    """

    def __init__(
        self,
        is_classifier: bool = False,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs,
    ):
        self.is_classifier = is_classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self):
        """Create Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_jobs": -1,
            **self.kwargs,
        }

        if self.is_classifier:
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
    ) -> "RandomForestPredictor":
        """Fit the Random Forest model."""
        if normalize:
            X = self.scaler.fit_transform(X)

        self._create_model()
        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Predict on new data."""
        if normalize and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba only available for classifiers")
        if normalize and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, normalize: bool = True) -> float:
        """Compute score on data."""
        preds = self.predict(X, normalize)

        if self.is_classifier:
            return accuracy_score(y, preds)
        else:
            return r2_score(y, preds)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.feature_importances_


class LinearPredictor:
    """
    Linear predictor wrapper for MixConfig experiments.

    Uses Ridge/Logistic regression depending on task type.

    Args:
        is_classifier: Whether classification task.
        alpha: Regularization strength.
        random_state: Random seed.
    """

    def __init__(
        self,
        is_classifier: bool = False,
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.is_classifier = is_classifier
        self.alpha = alpha
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self):
        """Create linear model."""
        from sklearn.linear_model import Ridge, LogisticRegression

        if self.is_classifier:
            self.model = LogisticRegression(
                C=1.0 / self.alpha,
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1,
            )
        else:
            self.model = Ridge(
                alpha=self.alpha,
                random_state=self.random_state,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
    ) -> "LinearPredictor":
        """Fit the linear model."""
        if normalize:
            X = self.scaler.fit_transform(X)

        self._create_model()
        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Predict on new data."""
        if normalize and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba only available for classifiers")
        if normalize and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray, normalize: bool = True) -> float:
        """Compute score on data."""
        preds = self.predict(X, normalize)

        if self.is_classifier:
            return accuracy_score(y, preds)
        else:
            return r2_score(y, preds)

    @property
    def coef_(self) -> np.ndarray:
        """Return coefficients."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.coef_


def get_predictor(
    predictor_type: str,
    is_classifier: bool = False,
    **kwargs,
) -> Union[XGBoostPredictor, RandomForestPredictor, LinearPredictor]:
    """
    Factory function to create predictors.

    Args:
        predictor_type: One of 'xgboost', 'rf', 'linear'.
        is_classifier: Whether classification task.
        **kwargs: Additional predictor parameters.

    Returns:
        Predictor instance.
    """
    predictors = {
        "xgboost": XGBoostPredictor,
        "xgb": XGBoostPredictor,
        "rf": RandomForestPredictor,
        "random_forest": RandomForestPredictor,
        "linear": LinearPredictor,
        "ridge": LinearPredictor,
        "logistic": LinearPredictor,
    }

    if predictor_type.lower() not in predictors:
        raise ValueError(f"Unknown predictor: {predictor_type}")

    return predictors[predictor_type.lower()](is_classifier=is_classifier, **kwargs)
