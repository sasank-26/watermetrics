"""Visualization module — generates publication-quality plots.

Creates:
  1. Actual vs Predicted scatter for each model
  2. Training / Validation loss curves
  3. Feature importance (Random Forest)
  4. Time trend analysis
  5. Error distribution plots
  6. Model comparison bar chart
  7. WQI time series
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

COLORS = {
    "Random Forest": "#e74c3c",
    "LSTM": "#3498db",
    "Standalone GNN": "#2ecc71",
    "Hybrid Transformer-GNN": "#9b59b6",
}


def plot_comparison_bar(results: Dict[str, Dict], output_dir: Path) -> str:
    """Bar chart comparing MAE, RMSE, R² across all models."""
    models = list(results.keys())
    metrics_names = ["MAE", "RMSE", "R²"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    for ax, metric in zip(axes, metrics_names):
        vals = [results[m].get(metric, 0) for m in models]
        colors = [COLORS.get(m, "#95a5a6") for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "comparison_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_actual_vs_predicted(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    target_names: Optional[List[str]] = None,
) -> List[str]:
    """Scatter plots of actual vs predicted for each model."""
    paths = []
    for model_name, (y_true, y_pred) in predictions.items():
        yt = y_true.reshape(-1, y_true.shape[-1]) if y_true.ndim > 2 else y_true
        yp = y_pred.reshape(-1, y_pred.shape[-1]) if y_pred.ndim > 2 else y_pred
        n_targets = yt.shape[1]
        t_names = target_names or [f"Target {i}" for i in range(n_targets)]

        fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        fig.suptitle(f"Actual vs Predicted — {model_name}", fontsize=14, fontweight="bold")

        color = COLORS.get(model_name, "#3498db")

        for i, (ax, tname) in enumerate(zip(axes, t_names)):
            ax.scatter(yt[:, i], yp[:, i], alpha=0.3, s=10, color=color, edgecolors="none")
            lims = [min(yt[:, i].min(), yp[:, i].min()),
                    max(yt[:, i].max(), yp[:, i].max())]
            ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Perfect")
            ax.set_xlabel(f"Actual {tname}")
            ax.set_ylabel(f"Predicted {tname}")
            ax.set_title(tname)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"actual_vs_pred_{model_name.replace(' ', '_').lower()}.png"
        p = output_dir / fname
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))

    return paths


def plot_loss_curves(
    loss_curves: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
) -> str:
    """Training and validation loss curves for neural network models."""
    n_models = len(loss_curves)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    for ax, (model_name, curves) in zip(axes, loss_curves.items()):
        epochs = range(1, len(curves["train"]) + 1)
        color = COLORS.get(model_name, "#3498db")
        ax.plot(epochs, curves["train"], label="Train", color=color, linewidth=2)
        ax.plot(epochs, curves["val"], label="Val", color=color, linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(model_name)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "loss_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    seq_len: int,
    output_dir: Path,
) -> str:
    """Random Forest feature importance plot (aggregated over time steps)."""
    n_features = len(feature_names)
    # importances has length seq_len * n_features — aggregate over time
    imp_matrix = importances.reshape(seq_len, n_features)
    agg_imp = imp_matrix.sum(axis=0)
    agg_imp = agg_imp / agg_imp.sum()

    sorted_idx = np.argsort(agg_imp)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(n_features),
        agg_imp[sorted_idx],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, n_features)),
        edgecolor="white",
    )
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_title("Feature Importance (Random Forest)", fontweight="bold")
    ax.set_ylabel("Importance")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, agg_imp[sorted_idx]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_error_distribution(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> str:
    """Error distribution histograms for each model."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Prediction Error Distribution", fontsize=14, fontweight="bold")

    for ax, (name, (yt, yp)) in zip(axes, predictions.items()):
        yt_flat = yt.reshape(-1)
        yp_flat = yp.reshape(-1)
        errors = yt_flat - yp_flat
        color = COLORS.get(name, "#3498db")
        ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(name.replace(" ", "\n"), fontsize=10)
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "error_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_time_trend(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    model_name: str,
    output_dir: Path,
    max_points: int = 200,
) -> str:
    """Time-series trend of actual vs predicted values."""
    yt = y_true.reshape(-1, y_true.shape[-1]) if y_true.ndim > 2 else y_true
    yp = y_pred.reshape(-1, y_pred.shape[-1]) if y_pred.ndim > 2 else y_pred

    n_targets = yt.shape[1]
    n_show = min(max_points, yt.shape[0])

    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4 * n_targets))
    if n_targets == 1:
        axes = [axes]
    fig.suptitle(f"Time Trend — {model_name}", fontsize=14, fontweight="bold")

    for i, (ax, tname) in enumerate(zip(axes, target_names)):
        ax.plot(range(n_show), yt[:n_show, i], label="Actual", color="#2c3e50", linewidth=1.5)
        color = COLORS.get(model_name, "#e74c3c")
        ax.plot(range(n_show), yp[:n_show, i], label="Predicted", color=color,
                linewidth=1.5, linestyle="--")
        ax.fill_between(range(n_show),
                        yt[:n_show, i], yp[:n_show, i],
                        alpha=0.15, color=color)
        ax.set_ylabel(tname)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    path = output_dir / f"time_trend_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def generate_all_plots(output_dir: str | Path) -> List[str]:
    """Generate all plots from saved comparison results."""
    output_dir = Path(output_dir)
    paths: List[str] = []

    # Load comparison results
    results_path = output_dir / "comparison_results.json"
    if not results_path.exists():
        print(f"No comparison results found at {results_path}")
        return paths

    with open(results_path) as f:
        data = json.load(f)

    comparison = data["comparison"]
    loss_curves = data.get("loss_curves", {})

    # 1. Comparison bar chart
    paths.append(plot_comparison_bar(comparison, output_dir))

    # 2. Loss curves
    if loss_curves:
        paths.append(plot_loss_curves(loss_curves, output_dir))

    # 3. Actual vs Predicted + Error distribution + Time trend
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    model_names = ["random_forest", "lstm", "standalone_gnn", "hybrid_transformer-gnn"]
    display_names = ["Random Forest", "LSTM", "Standalone GNN", "Hybrid Transformer-GNN"]

    for mkey, dname in zip(model_names, display_names):
        true_path = output_dir / f"pred_{mkey}_true.npy"
        pred_path = output_dir / f"pred_{mkey}_pred.npy"
        if true_path.exists() and pred_path.exists():
            predictions[dname] = (np.load(true_path), np.load(pred_path))

    if predictions:
        paths.extend(plot_actual_vs_predicted(predictions, output_dir))
        paths.append(plot_error_distribution(predictions, output_dir))

        # Time trend for hybrid
        if "Hybrid Transformer-GNN" in predictions:
            yt, yp = predictions["Hybrid Transformer-GNN"]
            target_names = ["DO", "pH"]  # default
            paths.append(plot_time_trend(yt, yp, target_names, "Hybrid Transformer-GNN", output_dir))

    # 4. Feature importance
    fi_path = output_dir / "rf_feature_importances.npy"
    if fi_path.exists():
        fi = np.load(fi_path)
        feature_names = ["DO", "pH", "COD", "CODMn", "NH4N", "DIN", "DIP", "TPH"]
        seq_len = fi.shape[0] // len(feature_names) if fi.shape[0] % len(feature_names) == 0 else 8
        paths.append(plot_feature_importance(fi, feature_names, seq_len, output_dir))

    return paths
