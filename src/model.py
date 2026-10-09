"""Interpretable binary Logistic Regression model using PyTorch CPU for CareTrail Phase 2.

Provides:
- PyTorch nn.Module for binary logistic regression (linear layer + sigmoid).
- Interpretable feature weights.
- Helper methods for getting/setting weights (FedAvg compatibility) and evaluation.
"""

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    """Interpretable binary Logistic Regression implemented using PyTorch CPU."""

    def __init__(self, in_features: int = 13):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear layer and sigmoid activation returning probabilities."""
        return torch.sigmoid(self.linear(x))

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Extract a deep copy of model parameter tensors on CPU."""
        return {name: param.detach().cpu().clone() for name, param in self.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load parameter tensors into model state."""
        self.load_state_dict(weights)

    def evaluate(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[float, float]:
        """Evaluate model predictions against labels, returning (auroc, accuracy).

        Args:
            X: Feature matrix of shape (N, in_features).
            y: Ground truth binary labels of shape (N,).

        Returns:
            Tuple of (auroc, accuracy).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X.float()

            if isinstance(y, torch.Tensor):
                y_arr = y.cpu().numpy().flatten()
            else:
                y_arr = np.asarray(y).flatten()

            probs = self.forward(X_tensor).squeeze(-1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            accuracy = float(accuracy_score(y_arr, preds))
            if len(np.unique(y_arr)) > 1:
                auroc = float(roc_auc_score(y_arr, probs))
            else:
                auroc = 0.5

            return auroc, accuracy
