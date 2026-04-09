"""Water Quality Index (WQI) calculation.

Uses CCME Water Quality Index methodology adapted for available parameters.
Also supports the weighted arithmetic sub-index method.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Standard limits (Class III — suitable for drinking after treatment)
# Source: China Environmental Quality Standards for Surface Water (GB 3838-2002)
# ─────────────────────────────────────────────────────────────────────────────

STANDARDS: Dict[str, Dict[str, float]] = {
    "DO":    {"ideal": 7.5, "standard": 5.0, "direction": "higher_better"},
    "pH":    {"ideal": 7.0, "standard_low": 6.0, "standard_high": 9.0, "direction": "range"},
    "COD":   {"ideal": 0.0, "standard": 20.0, "direction": "lower_better"},
    "CODMn": {"ideal": 0.0, "standard": 6.0,  "direction": "lower_better"},
    "NH4N":  {"ideal": 0.0, "standard": 1.0,  "direction": "lower_better"},
    "DIN":   {"ideal": 0.0, "standard": 0.5,  "direction": "lower_better"},
    "DIP":   {"ideal": 0.0, "standard": 0.025, "direction": "lower_better"},
    "TPH":   {"ideal": 0.0, "standard": 0.05, "direction": "lower_better"},
}

# Weights for weighted arithmetic method (should sum to 1)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "DO": 0.20, "pH": 0.15, "COD": 0.15, "CODMn": 0.10,
    "NH4N": 0.15, "DIN": 0.10, "DIP": 0.10, "TPH": 0.05,
}


def _sub_index(indicator: str, value: float) -> float:
    """Compute a 0-100 sub-index for a single parameter value.
    
    100 = ideal quality, 0 = worst quality.
    """
    spec = STANDARDS.get(indicator)
    if spec is None:
        return 50.0  # unknown indicator → neutral

    if spec["direction"] == "higher_better":
        ideal = spec["ideal"]
        std = spec["standard"]
        if value >= ideal:
            return 100.0
        elif value <= 0:
            return 0.0
        else:
            return max(0.0, min(100.0, (value / ideal) * 100.0))

    elif spec["direction"] == "lower_better":
        std = spec["standard"]
        if value <= 0:
            return 100.0
        elif value >= std * 2:
            return 0.0
        else:
            return max(0.0, min(100.0, (1.0 - value / (std * 2)) * 100.0))

    elif spec["direction"] == "range":
        lo = spec["standard_low"]
        hi = spec["standard_high"]
        ideal = spec["ideal"]
        if lo <= value <= hi:
            deviation = abs(value - ideal)
            max_dev = max(ideal - lo, hi - ideal)
            return max(0.0, 100.0 - (deviation / max_dev) * 30.0)
        else:
            return max(0.0, 100.0 - abs(value - ideal) * 20.0)

    return 50.0


def compute_wqi(
    values: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute the weighted-arithmetic Water Quality Index.
    
    Args:
        values: Mapping of indicator name → measured value.
        weights: Optional mapping of indicator name → weight.
                 If None, uses DEFAULT_WEIGHTS.
    
    Returns:
        WQI score in [0, 100].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for ind, val in values.items():
        w = weights.get(ind, 0.0)
        if w <= 0 or np.isnan(val):
            continue
        si = _sub_index(ind, val)
        weighted_sum += w * si
        total_weight += w

    if total_weight < 1e-9:
        return 0.0

    return weighted_sum / total_weight


def wqi_category(score: float) -> str:
    """Classify WQI score into quality category."""
    if score >= 90:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Medium"
    elif score >= 25:
        return "Bad"
    else:
        return "Very Bad"


def compute_wqi_series(
    df: pd.DataFrame,
    indicator_columns: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Compute WQI for each row in a DataFrame.
    
    Args:
        df: DataFrame where each row has indicator columns.
        indicator_columns: List of indicator column names present in df.
        weights: Optional weight mapping.
    
    Returns:
        Series of WQI scores.
    """
    results = []
    for _, row in df.iterrows():
        vals = {ind: float(row[ind]) for ind in indicator_columns if ind in row.index and pd.notna(row[ind])}
        results.append(compute_wqi(vals, weights))
    return pd.Series(results, index=df.index, name="WQI")
