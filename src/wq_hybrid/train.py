from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .data import SequenceDataset, build_tensors, temporal_split_indices
from .graph import build_knn_adjacency
from .model import HybridSTModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Flatten [B, N, O] into [B*N, O]
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    try:
        r2 = float(r2_score(yt, yp))
    except ValueError:
        r2 = float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def _run_epoch(
    model: HybridSTModel,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    ys = []
    yhs = []

    for x_seq, y in loader:
        x_seq = x_seq.to(device)  # [B, L, N, F]
        y = y.to(device)  # [B, N, O]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        y_hat = model(x_seq, adj)
        loss = torch.nn.functional.mse_loss(y_hat, y)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        ys.append(y.detach().cpu().numpy())
        yhs.append(y_hat.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yhs, axis=0)

    return float(np.mean(losses)), _metrics(y_true, y_pred)


def train_pipeline(config: ProjectConfig) -> Dict[str, float]:
    set_seed(config.seed)

    bundle, coords = build_tensors(config)
    adj = build_knn_adjacency(coords, k=config.knn_k, self_loop_weight=config.self_loop_weight)
    bundle.adj = adj

    train_idx, val_idx, test_idx = temporal_split_indices(
        total_t=bundle.x.shape[0],
        seq_len=config.seq_len,
        horizon=config.horizon,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    ds_train = SequenceDataset(bundle.x, bundle.y, config.seq_len, config.horizon, train_idx)
    ds_val = SequenceDataset(bundle.x, bundle.y, config.seq_len, config.horizon, val_idx)
    ds_test = SequenceDataset(bundle.x, bundle.y, config.seq_len, config.horizon, test_idx)

    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridSTModel(
        input_dim=len(config.indicators),
        output_dim=len(config.target_indicators),
        d_model=config.d_model,
        nhead=config.nhead,
        transformer_layers=config.transformer_layers,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
    ).to(device)
    adj = adj.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_metrics = _run_epoch(model, dl_train, adj, device, optimizer)
        va_loss, va_metrics = _run_epoch(model, dl_val, adj, device, optimizer=None)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"| val_mae={va_metrics['mae']:.4f} val_rmse={va_metrics['rmse']:.4f} val_r2={va_metrics['r2']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = _run_epoch(model, dl_test, adj, device, optimizer=None)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
            "feature_names": bundle.feature_names,
            "target_names": bundle.target_names,
            "node_ids": bundle.node_ids,
        },
        ckpt_path,
    )

    metrics = {
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nFinal Test Metrics:", metrics)
    print(f"Saved checkpoint: {ckpt_path}")

    return metrics
