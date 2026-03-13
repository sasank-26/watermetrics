from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D], adj: [N, N]
        neigh = torch.einsum("ij,bjd->bid", adj, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class HybridSTModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        transformer_layers: int = 2,
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.spatial_layers = nn.ModuleList([GraphLayer(d_model=d_model, dropout=dropout) for _ in range(gnn_layers)])

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x_seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, L, N, F]
        b, l, n, f = x_seq.shape

        # Temporal encoding per node.
        x = x_seq.permute(0, 2, 1, 3).reshape(b * n, l, f)  # [B*N, L, F]
        x = self.in_proj(x)
        x = self.temporal_encoder(x)
        h_temporal = x[:, -1, :].reshape(b, n, -1)  # [B, N, D]

        # Spatial propagation on final temporal states.
        h_spatial = h_temporal
        for layer in self.spatial_layers:
            h_spatial = layer(h_spatial, adj)

        h = torch.cat([h_temporal, h_spatial], dim=-1)
        h = self.fusion(h)
        y_hat = self.head(h)  # [B, N, O]
        return y_hat
