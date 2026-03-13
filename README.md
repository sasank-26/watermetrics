# Hybrid Spatiotemporal Transformer-GNN for Water Quality Prediction

This project provides an end-to-end baseline pipeline to forecast water quality using:
- **Transformer** blocks for temporal dynamics
- **Graph Neural** propagation for spatial interactions across stations

It uses the two available datasets:
- [monthly_ocean.csv](monthly_ocean.csv)
- [weekly_land.csv](weekly_land.csv)

## What is implemented

- Unified preprocessing for land + ocean observations
- Indicator pivoting into node-time feature tensors
- kNN spatial graph construction from station coordinates
- Hybrid Transformer-GNN model in PyTorch
- Time-based train/val/test split
- Metrics: **MAE**, **RMSE**, **R²**
- Model checkpoint saving and prediction export

## Project structure

- `src/wq_hybrid/config.py` – configuration dataclass
- `src/wq_hybrid/data.py` – loading, cleaning, feature tensor creation
- `src/wq_hybrid/graph.py` – graph building utilities
- `src/wq_hybrid/model.py` – Hybrid Transformer-GNN model
- `src/wq_hybrid/train.py` – train/eval loop and metrics
- `run_training.py` – CLI entrypoint

## Quick start

1) Create and activate a Python environment.

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run training:

```bash
python run_training.py --data-dir . --epochs 20 --batch-size 32 --horizon 1 --seq-len 8
```

## Notes

- `horizon=1` means next-step forecast in the chosen weekly cadence.
- You can customize indicators and targets through CLI arguments.
- This is a strong baseline scaffold and can be extended with directed hydro-topology edges, uncertainty heads, and multi-horizon decoding.
