from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from .config import ProjectConfig
except ImportError:  # direct script execution fallback
    from config import ProjectConfig


@dataclass
class DataBundle:
    x: torch.Tensor  # [T, N, F]
    y: torch.Tensor  # [T, N, O]
    adj: torch.Tensor  # [N, N]
    node_ids: List[str]
    feature_names: List[str]
    target_names: List[str]
    time_index: pd.DatetimeIndex


def _parse_value(v: object) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if not s:
        return np.nan
    # Censored values like "< DL" are kept missing for later imputation.
    if s.startswith("<"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_datetime_col(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # Fallback parse for ambiguous cases.
    mask = dt.isna()
    if mask.any():
        dt2 = pd.to_datetime(series[mask], errors="coerce", dayfirst=False)
        dt.loc[mask] = dt2
    return dt


def _standardize_ocean(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "station_id": df["MonitoringLocationIdentifier"].astype(str),
            "lon": pd.to_numeric(df["LongitudeMeasure_WGS84"], errors="coerce"),
            "lat": pd.to_numeric(df["LatitudeMeasure_WGS84"], errors="coerce"),
            "date": _to_datetime_col(df["MonitoringDate"]),
            "indicator": df["IndicatorsName"].astype(str),
            "value": df["Value"].map(_parse_value),
            "unit": df.get("Unit", pd.Series([None] * len(df))).astype(str),
            "source": df.get("SourceProvider", pd.Series([None] * len(df))).astype(str),
            "domain": "ocean",
        }
    )
    return out


def _standardize_land(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "station_id": df["MonitoringLocationIdentifier"].astype(str),
            "lon": pd.to_numeric(df["LongitudeMeasure_WGS84"], errors="coerce"),
            "lat": pd.to_numeric(df["LatitudeMeasure_WGS84"], errors="coerce"),
            "date": _to_datetime_col(df["MonitoringDate"]),
            "indicator": df["IndicatorsName"].astype(str),
            "value": df["Value"].map(_parse_value),
            "unit": df.get("Unit", pd.Series([None] * len(df))).astype(str),
            "source": df.get("SourceProvider", pd.Series([None] * len(df))).astype(str),
            "domain": "land",
        }
    )
    return out


def load_and_unify_data(config: ProjectConfig) -> pd.DataFrame:
    data_dir = Path(config.data_dir)
    ocean_path = data_dir / config.ocean_file
    land_path = data_dir / config.land_file

    ocean = pd.read_csv(ocean_path)
    land = pd.read_csv(land_path)

    ocean_std = _standardize_ocean(ocean)
    land_std = _standardize_land(land)

    df = pd.concat([ocean_std, land_std], ignore_index=True)
    df = df.dropna(subset=["station_id", "date", "indicator"])

    # Keep configured indicators only.
    df = df[df["indicator"].isin(config.indicators)].copy()

    # Basic deduplication by averaging repeated same-day measurements.
    df = (
        df.groupby(["station_id", "lon", "lat", "date", "indicator", "domain"], dropna=False)["value"]
        .mean()
        .reset_index()
    )

    return df


def _pivot_station_time(
    df: pd.DataFrame,
    indicators: List[str],
    resample_freq: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prepare complete station-date grid via per-station resampling.
    frames = []
    coord_rows = []

    for station, sdf in df.groupby("station_id"):
        sdf = sdf.sort_values("date")
        lon = sdf["lon"].dropna().iloc[0] if not sdf["lon"].dropna().empty else np.nan
        lat = sdf["lat"].dropna().iloc[0] if not sdf["lat"].dropna().empty else np.nan
        domain = sdf["domain"].dropna().iloc[0] if not sdf["domain"].dropna().empty else "unknown"

        pivot = sdf.pivot_table(index="date", columns="indicator", values="value", aggfunc="mean")
        pivot = pivot.reindex(columns=indicators)
        pivot = pivot.resample(resample_freq).mean()

        # Time interpolation then forward/backward fill.
        pivot = pivot.interpolate(limit_direction="both")
        pivot = pivot.ffill().bfill()

        pivot["station_id"] = station
        frames.append(pivot.reset_index())

        coord_rows.append({"station_id": station, "lon": lon, "lat": lat, "domain": domain})

    full = pd.concat(frames, ignore_index=True)
    coords = pd.DataFrame(coord_rows)

    # Global fallback imputation (if still missing).
    for c in indicators:
        med = full[c].median(skipna=True)
        full[c] = full[c].fillna(med if not np.isnan(med) else 0.0)

    return full, coords


def build_tensors(config: ProjectConfig) -> Tuple[DataBundle, pd.DataFrame]:
    df = load_and_unify_data(config)
    full, coords = _pivot_station_time(df, config.indicators, config.resample_freq)

    # Align all stations on common date index.
    all_dates = pd.DatetimeIndex(sorted(full["date"].unique()))
    node_ids = sorted(full["station_id"].unique().tolist())

    # Map station -> fixed lat/lon.
    coord_map: Dict[str, Tuple[float, float]] = {}
    for _, r in coords.iterrows():
        coord_map[str(r["station_id"])] = (float(r["lat"]) if pd.notna(r["lat"]) else np.nan, float(r["lon"]) if pd.notna(r["lon"]) else np.nan)

    t_len = len(all_dates)
    n_nodes = len(node_ids)
    f_len = len(config.indicators)
    o_len = len(config.target_indicators)

    x_np = np.zeros((t_len, n_nodes, f_len), dtype=np.float32)
    y_np = np.zeros((t_len, n_nodes, o_len), dtype=np.float32)

    station_index = {s: i for i, s in enumerate(node_ids)}
    time_index = {d: i for i, d in enumerate(all_dates)}

    full_idx = full.set_index(["date", "station_id"])
    for d in all_dates:
        for s in node_ids:
            if (d, s) not in full_idx.index:
                continue
            row = full_idx.loc[(d, s)]
            # If duplicate rows exist, take the first aggregated row.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            ti = time_index[d]
            ni = station_index[s]
            x_np[ti, ni, :] = row[config.indicators].to_numpy(dtype=np.float32)
            y_np[ti, ni, :] = row[config.target_indicators].to_numpy(dtype=np.float32)

    # Simple feature scaling over time*nodes dimension.
    x_flat = x_np.reshape(-1, f_len)
    x_mean = x_flat.mean(axis=0, keepdims=True)
    x_std = x_flat.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_np = ((x_flat - x_mean) / x_std).reshape(t_len, n_nodes, f_len)

    y_flat = y_np.reshape(-1, o_len)
    y_mean = y_flat.mean(axis=0, keepdims=True)
    y_std = y_flat.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0
    y_np = ((y_flat - y_mean) / y_std).reshape(t_len, n_nodes, o_len)

    bundle = DataBundle(
        x=torch.from_numpy(x_np),
        y=torch.from_numpy(y_np),
        adj=torch.empty(0),
        node_ids=node_ids,
        feature_names=config.indicators,
        target_names=config.target_indicators,
        time_index=all_dates,
    )

    coords_out = pd.DataFrame(
        {
            "station_id": node_ids,
            "lat": [coord_map.get(s, (np.nan, np.nan))[0] for s in node_ids],
            "lon": [coord_map.get(s, (np.nan, np.nan))[1] for s in node_ids],
        }
    )

    return bundle, coords_out


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, seq_len: int, horizon: int, indices: List[int]):
        self.x = x
        self.y = y
        self.seq_len = seq_len
        self.horizon = horizon
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        x_seq = self.x[t - self.seq_len : t]  # [L, N, F]
        y_t = self.y[t + self.horizon - 1]  # [N, O]
        return x_seq, y_t


def temporal_split_indices(total_t: int, seq_len: int, horizon: int, val_ratio: float, test_ratio: float):
    valid_start = seq_len
    valid_end = total_t - horizon
    all_idx = list(range(valid_start, valid_end + 1))

    n = len(all_idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train : n_train + n_val]
    test_idx = all_idx[n_train + n_val :]

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    cfg = ProjectConfig()
    bundle, coords = build_tensors(cfg)
    print("Data pipeline check complete")
    print(f"x shape: {tuple(bundle.x.shape)}")
    print(f"y shape: {tuple(bundle.y.shape)}")
    print(f"nodes: {len(bundle.node_ids)}")
    print(f"time steps: {len(bundle.time_index)}")
    print(f"coords rows: {len(coords)}")
