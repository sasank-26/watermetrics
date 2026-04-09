#!/usr/bin/env python3
"""Run full model comparison pipeline.

Usage:
    python run_comparison.py --epochs 30
    python run_comparison.py --epochs 10 --output-dir outputs_comparison
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wq_hybrid.config import ProjectConfig
from wq_hybrid.compare import run_comparison
from wq_hybrid.visualize import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full model comparison: RF, LSTM, GNN, Hybrid Transformer-GNN"
    )
    parser.add_argument("--data-dir", type=str, default=ROOT)
    parser.add_argument("--ocean-file", type=str, default="monthly_ocean.csv")
    parser.add_argument("--land-file", type=str, default="weekly_land.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--indicators", type=str,
        default="DO,pH,COD,CODMn,NH4N,DIN,DIP,TPH",
    )
    parser.add_argument("--targets", type=str, default="DO,pH")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots")
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
        indicators=[x.strip() for x in args.indicators.split(",") if x.strip()],
        target_indicators=[x.strip() for x in args.targets.split(",") if x.strip()],
    )

    print("=" * 60)
    print("HYBRID SPATIOTEMPORAL TRANSFORMER-GNN")
    print("FULL MODEL COMPARISON PIPELINE")
    print("=" * 60)
    print(f"Epochs: {cfg.epochs}")
    print(f"Indicators: {cfg.indicators}")
    print(f"Targets: {cfg.target_indicators}")
    print(f"Output: {cfg.output_dir}")
    print()

    # Run all models
    results = run_comparison(cfg)

    # Generate plots
    if not args.skip_plots:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        try:
            plots = generate_all_plots(cfg.output_dir)
            print(f"Generated {len(plots)} plots")
            for p in plots:
                print(f"  ✓ {p}")
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")
            print("You can still view results in the Streamlit dashboard.")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nTo launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
