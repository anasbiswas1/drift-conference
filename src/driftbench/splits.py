"""Split strategies for Demo A.

The whole leakage argument lives here: a random split mixes near-neighbour
traffic across train/test, whereas chronological and leave-one-day-out splits
respect the day/scenario structure the CIC capture imposes.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    n: int, labels: pd.Series, test_size: float = 0.2, val_size: float = 0.2, seed: int = 0
) -> Dict[str, np.ndarray]:
    """Stratified random split -- the naive baseline most papers report."""
    idx = np.arange(n)
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed,
                                       stratify=labels.values)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, random_state=seed,
                                  stratify=labels.values[train_val])
    return {"train": train, "val": val, "test": test}


def _time_order(metadata: pd.DataFrame, timestamps: Optional[pd.Series]) -> np.ndarray:
    """Return row indices in chronological order, using an explicit timestamp
    series if given, else the first time-like metadata column, else identity."""
    if timestamps is not None:
        ts = pd.to_datetime(timestamps, errors="coerce")
        return np.argsort(ts.values, kind="stable")
    for c in metadata.columns:
        if "timestamp" in c or "date" in c:
            ts = pd.to_datetime(metadata[c], errors="coerce")
            return np.argsort(ts.values, kind="stable")
    return np.arange(len(metadata))


def chronological_split(
    metadata: pd.DataFrame, timestamps: Optional[pd.Series] = None,
    train_frac: float = 0.6, val_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Time-ordered 60/20/20: earliest traffic trains, latest is held out."""
    order = _time_order(metadata, timestamps)
    n = len(order)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return {"train": order[:a], "val": order[a:b], "test": order[b:]}


def _day_key(metadata: pd.DataFrame, timestamps: Optional[pd.Series]) -> pd.Series:
    if timestamps is not None:
        return pd.to_datetime(timestamps, errors="coerce").dt.date.astype(str)
    for c in metadata.columns:
        if c == "day" or c.endswith("_day"):
            return metadata[c].astype(str)
        if "timestamp" in c or "date" in c:
            return pd.to_datetime(metadata[c], errors="coerce").dt.date.astype(str)
    raise KeyError("No day/timestamp metadata available for leave-one-day-out.")


def leave_one_day_out(
    metadata: pd.DataFrame, timestamps: Optional[pd.Series] = None
) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """Yield (held_out_day, {'train','test'}) folds. One fold per capture day."""
    days = _day_key(metadata, timestamps)
    idx = np.arange(len(days))
    folds = []
    for d in sorted(days.dropna().unique()):
        test = idx[(days == d).values]
        train = idx[(days != d).values]
        folds.append((d, {"train": train, "test": test}))
    return folds
