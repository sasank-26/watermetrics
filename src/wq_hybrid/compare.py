"""Comprehensive model comparison pipeline.

Trains and evaluates all four models:
  1. Random Forest
  2. LSTM
  3. Standalone GNN
  4. Hybrid Transformer-GNN  (the proposed model)

Generates a unified comparison table and persists results.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .baselines import LSTMModel, RandomForestWrapper, StandaloneGNN
from .config import ProjectConfig
from .data import DataBundle, SequenceDataset, build_tensors, temporal_split_indices
from .graph import build_knn_adjacency
from .model import HybridSTModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1, y_true.shape[-1]) if y_true.ndim > 2 else y_true
    yp = y_pred.reshape(-1, y_pred.shape[-1]) if y_pred.ndim > 2 else y_pred
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    try:
        r2 = float(r2_score(yt, yp))
    except ValueError:
        r2 = float("nan")
    # MAPE (with safety)
    mask = np.abs(yt) > 1e-6
    if mask.any():
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
    else:
        mape = float("nan")
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE(%)": mape}


# ─────────────────────────────────────────────────────────────────────────────
# Train/Eval helpers for PyTorch models
# ─────────────────────────────────────────────────────────────────────────────

def _train_nn_model(
    model: torch.nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    config: ProjectConfig,
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val = float("inf")
    best_state = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for x_seq, y in dl_train:
            x_seq, y = x_seq.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_seq, adj)
            loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for x_seq, y in dl_val:
                x_seq, y = x_seq.to(device), y.to(device)
                y_hat = model(x_seq, adj)
                val_epoch_losses.append(torch.nn.functional.mse_loss(y_hat, y).item())
        val_loss = float(np.mean(val_epoch_losses))
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch:03d} train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def _eval_nn_model(
    model: torch.nn.Module,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, yhs = [], []
    with torch.no_grad():
        for x_seq, y in loader:
            x_seq = x_seq.to(device)
            y_hat = model(x_seq, adj)
            ys.append(y.numpy())
            yhs.append(y_hat.cpu().numpy())
    return np.concatenate(ys), np.concatenate(yhs)


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(config: ProjectConfig) -> Dict[str, Any]:
    """Run all 4 models and return unified results dict."""
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
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
    adj_dev = adj.to(device)

    input_dim = len(config.indicators)
    output_dim = len(config.target_indicators)

    results: Dict[str, Any] = {}
    all_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ── 1. Random Forest ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 1: RANDOM FOREST")
    print("=" * 60)
    t0 = time.time()

    # Collect numpy arrays from loaders
    train_x_list, train_y_list = [], []
    for x_s, y_s in dl_train:
        train_x_list.append(x_s.numpy())
        train_y_list.append(y_s.numpy())
    train_x_np = np.concatenate(train_x_list)
    train_y_np = np.concatenate(train_y_list)

    test_x_list, test_y_list = [], []
    for x_s, y_s in dl_test:
        test_x_list.append(x_s.numpy())
        test_y_list.append(y_s.numpy())
    test_x_np = np.concatenate(test_x_list)
    test_y_np = np.concatenate(test_y_list)

    rf = RandomForestWrapper(n_estimators=200, max_depth=20, random_state=config.seed)
    rf.fit(train_x_np, train_y_np)
    rf_pred = rf.predict(test_x_np)
    rf_true = rf._flatten_target(test_y_np)
    rf_time = time.time() - t0

    rf_metrics = _metrics(rf_true, rf_pred)
    rf_metrics["Train Time (s)"] = round(rf_time, 2)
    results["Random Forest"] = rf_metrics
    all_predictions["Random Forest"] = (rf_true, rf_pred)
    print(f"  RF metrics: {rf_metrics}")

    # ── 2. LSTM ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: LSTM")
    print("=" * 60)
    t0 = time.time()
    lstm_model = LSTMModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.d_model,
        num_layers=2,
        dropout=config.dropout,
    ).to(device)

    lstm_model, lstm_train_loss, lstm_val_loss = _train_nn_model(
        lstm_model, dl_train, dl_val, adj_dev, device, config
    )
    lstm_true, lstm_pred = _eval_nn_model(lstm_model, dl_test, adj_dev, device)
    lstm_time = time.time() - t0

    lstm_metrics = _metrics(lstm_true, lstm_pred)
    lstm_metrics["Train Time (s)"] = round(lstm_time, 2)
    results["LSTM"] = lstm_metrics
    all_predictions["LSTM"] = (lstm_true, lstm_pred)
    print(f"  LSTM metrics: {lstm_metrics}")

    # ── 3. Standalone GNN ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 3: STANDALONE GNN")
    print("=" * 60)
    t0 = time.time()
    gnn_model = StandaloneGNN(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=config.d_model,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
    ).to(device)

    gnn_model, gnn_train_loss, gnn_val_loss = _train_nn_model(
        gnn_model, dl_train, dl_val, adj_dev, device, config
    )
    gnn_true, gnn_pred = _eval_nn_model(gnn_model, dl_test, adj_dev, device)
    gnn_time = time.time() - t0

    gnn_metrics = _metrics(gnn_true, gnn_pred)
    gnn_metrics["Train Time (s)"] = round(gnn_time, 2)
    results["Standalone GNN"] = gnn_metrics
    all_predictions["Standalone GNN"] = (gnn_true, gnn_pred)
    print(f"  GNN metrics: {gnn_metrics}")

    # ── 4. Hybrid Transformer-GNN (proposed) ────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 4: HYBRID TRANSFORMER-GNN (PROPOSED)")
    print("=" * 60)
    t0 = time.time()
    hybrid_model = HybridSTModel(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        transformer_layers=config.transformer_layers,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
    ).to(device)

    hybrid_model, hybrid_train_loss, hybrid_val_loss = _train_nn_model(
        hybrid_model, dl_train, dl_val, adj_dev, device, config
    )
    hybrid_true, hybrid_pred = _eval_nn_model(hybrid_model, dl_test, adj_dev, device)
    hybrid_time = time.time() - t0

    hybrid_metrics = _metrics(hybrid_true, hybrid_pred)
    hybrid_metrics["Train Time (s)"] = round(hybrid_time, 2)
    results["Hybrid Transformer-GNN"] = hybrid_metrics
    all_predictions["Hybrid Transformer-GNN"] = (hybrid_true, hybrid_pred)
    print(f"  Hybrid metrics: {hybrid_metrics}")

    # Save hybrid checkpoint
    torch.save(
        {
            "state_dict": hybrid_model.state_dict(),
            "config": config.__dict__,
            "feature_names": bundle.feature_names,
            "target_names": bundle.target_names,
            "node_ids": bundle.node_ids,
        },
        output_dir / "best_model.pt",
    )

    # ── Comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    table = pd.DataFrame(results).T
    table.index.name = "Model"
    print(table.to_string())
    table.to_csv(output_dir / "comparison_table.csv")

    # ── Save all results ────────────────────────────────────────────────
    full_results = {
        "comparison": results,
        "loss_curves": {
            "LSTM": {"train": lstm_train_loss, "val": lstm_val_loss},
            "Standalone GNN": {"train": gnn_train_loss, "val": gnn_val_loss},
            "Hybrid Transformer-GNN": {"train": hybrid_train_loss, "val": hybrid_val_loss},
        },
        "predictions": {},
    }

    # Save predictions for visualization
    for name, (yt, yp) in all_predictions.items():
        np.save(output_dir / f"pred_{name.replace(' ', '_').lower()}_true.npy", yt)
        np.save(output_dir / f"pred_{name.replace(' ', '_').lower()}_pred.npy", yp)

    # Save RF feature importances
    np.save(output_dir / "rf_feature_importances.npy", rf.feature_importances())

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(
            {"comparison": results, "loss_curves": full_results["loss_curves"]},
            f, indent=2, default=str,
        )

    print(f"\nAll results saved to {output_dir}/")
    return full_results
