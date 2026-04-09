"""Hyperparameter tuning using Optuna.

Optimizes the Hybrid Transformer-GNN model hyperparameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader

from wq_hybrid.config import ProjectConfig
from wq_hybrid.data import SequenceDataset, build_tensors, temporal_split_indices
from wq_hybrid.graph import build_knn_adjacency
from wq_hybrid.model import HybridSTModel


def _train_and_evaluate(
    config: ProjectConfig,
    dl_train: DataLoader,
    dl_val: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
) -> float:
    """Train for a few epochs and return validation loss."""
    model = HybridSTModel(
        input_dim=len(config.indicators),
        output_dim=len(config.target_indicators),
        d_model=config.d_model,
        nhead=config.nhead,
        transformer_layers=config.transformer_layers,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    best_val = float("inf")
    
    for epoch in range(config.epochs):
        model.train()
        for x_seq, y in dl_train:
            x_seq, y = x_seq.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_seq, adj)
            loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_seq, y in dl_val:
                x_seq, y = x_seq.to(device), y.to(device)
                y_hat = model(x_seq, adj)
                val_losses.append(torch.nn.functional.mse_loss(y_hat, y).item())
        
        val_loss = sum(val_losses) / len(val_losses)
        if val_loss < best_val:
            best_val = val_loss
            
    return best_val


def objective(trial: optuna.Trial, config: ProjectConfig, bundle: Any, coords: Any, dl_train: DataLoader, dl_val: DataLoader, device: torch.device) -> float:
    """Optuna objective function."""
    
    # Suggest hyperparameters
    cfg = ProjectConfig(
        data_dir=config.data_dir,
        ocean_file=config.ocean_file,
        land_file=config.land_file,
        indicators=config.indicators,
        target_indicators=config.target_indicators,
        epochs=5,  # Short training for tuning
        batch_size=config.batch_size,
    )
    
    # Structural params
    cfg.d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    cfg.nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    cfg.transformer_layers = trial.suggest_int("transformer_layers", 1, 3)
    cfg.gnn_layers = trial.suggest_int("gnn_layers", 1, 3)
    
    # Graph params
    cfg.knn_k = trial.suggest_int("knn_k", 3, 10)
    
    # Optimization params
    cfg.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Rebuild adjacency with new k
    adj = build_knn_adjacency(coords, k=cfg.knn_k, self_loop_weight=cfg.self_loop_weight).to(device)
    
    return _train_and_evaluate(cfg, dl_train, dl_val, adj, device)


def run_tuning(n_trials: int = 20) -> optuna.study.Study:
    cfg = ProjectConfig()
    
    # Load data once
    print("Loading data for tuning...")
    bundle, coords = build_tensors(cfg)
    
    train_idx, val_idx, _ = temporal_split_indices(
        total_t=bundle.x.shape[0],
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )
    
    ds_train = SequenceDataset(bundle.x, bundle.y, cfg.seq_len, cfg.horizon, train_idx)
    ds_val = SequenceDataset(bundle.x, bundle.y, cfg.seq_len, cfg.horizon, val_idx)
    
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    study = optuna.create_study(direction="minimize", study_name="wq-hybrid-tuning")
    
    from typing import Any
    study.optimize(
        lambda t: objective(t, cfg, bundle, coords, dl_train, dl_val, device), 
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\nBest Trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
        
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
        
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Number of optuna trials")
    args = parser.parse_args()
    
    print(f"Starting hyperparameter tuning with {args.trials} trials...")
    run_tuning(args.trials)
