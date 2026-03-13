from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _pairwise_distance(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def build_knn_adjacency(coords_df: pd.DataFrame, k: int = 5, self_loop_weight: float = 1.0) -> torch.Tensor:
    coords = coords_df[["lat", "lon"]].to_numpy(dtype=np.float32)

    # Replace missing coords with global mean to keep tensor shape stable.
    if np.isnan(coords).any():
        col_means = np.nanmean(coords, axis=0)
        inds = np.where(np.isnan(coords))
        coords[inds] = np.take(col_means, inds[1])

    d = _pairwise_distance(coords)
    n = d.shape[0]

    # Scale for RBF similarity.
    sigma = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    sigma = max(float(sigma), 1e-6)

    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nn_idx = np.argsort(d[i])[1 : k + 1]  # skip self
        for j in nn_idx:
            w = np.exp(-(d[i, j] ** 2) / (2 * sigma**2))
            a[i, j] = max(a[i, j], w)
            a[j, i] = max(a[j, i], w)

    # Self loops.
    np.fill_diagonal(a, np.maximum(np.diag(a), self_loop_weight))

    # Symmetric normalization D^-1/2 A D^-1/2
    deg = a.sum(axis=1)
    deg[deg < 1e-6] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    a_norm = d_inv_sqrt @ a @ d_inv_sqrt

    return torch.from_numpy(a_norm.astype(np.float32))
