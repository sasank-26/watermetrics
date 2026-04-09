from dataclasses import dataclass, field
from typing import List


@dataclass
class ProjectConfig:
    data_dir: str = "."
    ocean_file: str = "monthly_ocean.csv"
    land_file: str = "weekly_land.csv"

    indicators: List[str] = field(
        default_factory=lambda: ["DO", "pH", "COD", "CODMn", "NH4N", "DIN", "DIP", "TPH"]
    )
    target_indicators: List[str] = field(default_factory=lambda: ["DO", "pH"])

    # Temporal setup
    resample_freq: str = "W"  # Weekly
    seq_len: int = 8
    horizon: int = 1

    # Graph setup
    knn_k: int = 5
    self_loop_weight: float = 1.0

    # Model setup
    d_model: int = 64
    nhead: int = 4
    transformer_layers: int = 2
    gnn_layers: int = 2
    dropout: float = 0.1

    # Training setup
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # Output
    output_dir: str = "outputs"
