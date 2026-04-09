"""Baseline models for comparison: LSTM, Standalone GNN, Random Forest."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor

from .model import GraphLayer


# ─────────────────────────────────────────────────────────────────────────────
# 1. LSTM Baseline
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Pure LSTM baseline — captures temporal patterns only."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_seq: [B, L, N, F]
        Returns: [B, N, O]
        """
        b, l, n, f = x_seq.shape
        # Process each node independently
        x = x_seq.permute(0, 2, 1, 3).reshape(b * n, l, f)  # [B*N, L, F]
        _, (h_n, _) = self.lstm(x)                            # h_n: [layers, B*N, H]
        h = h_n[-1].reshape(b, n, -1)                         # [B, N, H]
        return self.fc(h)                                      # [B, N, O]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Standalone GNN Baseline
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneGNN(nn.Module):
    """Standalone GNN — captures spatial patterns only (no temporal modelling)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Use last time step features only
        self.in_proj = nn.Linear(input_dim, d_model)
        self.gnn_layers = nn.ModuleList(
            [GraphLayer(d_model=d_model, dropout=dropout) for _ in range(gnn_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, L, N, F]
        adj:   [N, N]
        Returns: [B, N, O]
        """
        # Take last time step
        x = x_seq[:, -1, :, :]     # [B, N, F]
        h = self.in_proj(x)         # [B, N, D]
        for layer in self.gnn_layers:
            h = layer(h, adj)
        return self.head(h)         # [B, N, O]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Random Forest Wrapper (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestWrapper:
    """
    Random Forest baseline.  Flattens spatio-temporal windows into feature
    vectors and trains a separate RF per target dimension.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int | None = 20, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models: list[RandomForestRegressor] = []
        self._output_dim = 0

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _flatten_batch(x_seq: np.ndarray) -> np.ndarray:
        """[B, L, N, F] → [B*N, L*F]"""
        b, l, n, f = x_seq.shape
        # For each node, the feature vector is the flattened temporal window
        return x_seq.transpose(0, 2, 1, 3).reshape(b * n, l * f)

    @staticmethod
    def _flatten_target(y: np.ndarray) -> np.ndarray:
        """[B, N, O] → [B*N, O]"""
        b, n, o = y.shape
        return y.reshape(b * n, o)

    # ── API ──────────────────────────────────────────────────────────────
    def fit(self, x_seq: np.ndarray, y: np.ndarray) -> "RandomForestWrapper":
        X = self._flatten_batch(x_seq)
        Y = self._flatten_target(y)
        self._output_dim = Y.shape[1]
        self.models = []
        for i in range(self._output_dim):
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, Y[:, i])
            self.models.append(rf)
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        X = self._flatten_batch(x_seq)
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds  # [B*N, O]

    def feature_importances(self) -> np.ndarray:
        """Average feature importances across targets. Returns [L*F]."""
        return np.mean([m.feature_importances_ for m in self.models], axis=0)
