from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wq_hybrid.config import ProjectConfig
from wq_hybrid.train import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid Spatiotemporal Transformer-GNN model")

    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--ocean-file", type=str, default="monthly_ocean.csv")
    parser.add_argument("--land-file", type=str, default="weekly_land.csv")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")

    parser.add_argument(
        "--indicators",
        type=str,
        default="DO,pH,COD,CODMn,NH4N,DIN,DIP,TPH",
        help="Comma-separated indicator names",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="DO,pH",
        help="Comma-separated target indicator names",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ProjectConfig(
        data_dir=args.data_dir,
        ocean_file=args.ocean_file,
        land_file=args.land_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        horizon=args.horizon,
        knn_k=args.knn_k,
        output_dir=args.output_dir,
        indicators=[s.strip() for s in args.indicators.split(",") if s.strip()],
        target_indicators=[s.strip() for s in args.targets.split(",") if s.strip()],
    )

    train_pipeline(cfg)


if __name__ == "__main__":
    main()
