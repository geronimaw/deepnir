"""
Model Training and Inference Module

Framework-agnostic base classes and utilities for model training.

Example usage:
    from deep_nir.models import BaseModel
    
    class MyModel(BaseModel):
        def train(self, X, y):
            # Your training logic
            pass
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    
    This provides a framework-agnostic interface that can be implemented
    for any ML framework (PyTorch, TensorFlow, scikit-learn, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.is_trained = False
        
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and values
        """
        raise NotImplementedError("Implement evaluation logic")


__all__ = [
    "BaseModel",
]
