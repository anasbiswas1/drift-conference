"""Phase 0 -- data hygiene for CIC flow tables.

Handles the documented failure modes: infinities/NaNs (notably in
``Flow Bytes/s`` and ``Flow Packets/s``), constant/zero-variance columns,
duplicate flows, and leakage-prone identifier/timestamp columns. Crucially,
timestamps and day identifiers are *retained as metadata* (for chronological
and leave-one-day-out splitting) even though they are stripped from the
model's input features.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise column names so schema naming drift across CIC releases
    (extra spaces, case, underscores) does not break feature alignment."""
    def norm(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"[\s_]+", "_", c)
        c = re.sub(r"[^0-9a-z_/%.\-]", "", c)
        return c
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out


# Columns that leak or are non-predictive identifiers. Stripped from features.
ID_PATTERNS = (
    "flow_id", "src_ip", "source_ip", "dst_ip", "destination_ip",
    "src_port", "source_port", "dst_port", "destination_port", "protocol",
    "fwd_header_length.1",  # known duplicate column in some CIC exports
)
TIME_PATTERNS = ("timestamp", "date", "day")
LABEL_CANDIDATES = ("label", "attack", "type", "class")


@dataclass
class CleanResult:
    features: pd.DataFrame      # numeric model inputs only
    metadata: pd.DataFrame      # timestamps / day ids, retained for splitting
    labels: pd.Series           # native label column
    dropped_constant: List[str]
    n_dedup_removed: int


def _find_label_col(df: pd.DataFrame) -> str:
    for cand in LABEL_CANDIDATES:
        for c in df.columns:
            if c == cand or c.endswith("_" + cand) or c.startswith(cand):
                return c
    raise KeyError(f"No label column found among {LABEL_CANDIDATES}; columns={list(df.columns)[:10]}...")


def clean_flows(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    drop_constant: bool = True,
    dedup: bool = True,
) -> CleanResult:
    """Run the full Phase 0 pipeline on a single flow table.

    Returns a :class:`CleanResult` separating model features from retained
    metadata. All steps are deterministic and side-effect free.
    """
    df = normalise_columns(df)
    label_col = label_col or _find_label_col(df)

    labels = df[label_col].astype(str).str.strip()

    # Partition columns: metadata (time/day) retained, ids dropped, rest features.
    meta_cols = [c for c in df.columns if any(p in c for p in TIME_PATTERNS)]
    id_cols = [c for c in df.columns if any(c == p or c.endswith(p) for p in ID_PATTERNS)]
    feature_cols = [c for c in df.columns
                    if c not in meta_cols and c not in id_cols and c != label_col]

    metadata = df[meta_cols].copy()
    features = df[feature_cols].copy()

    # Coerce to numeric; non-numeric feature columns become NaN then are pruned.
    features = features.apply(pd.to_numeric, errors="coerce")

    # Infinities -> NaN (Flow Bytes/s, Flow Packets/s are the usual offenders).
    features = features.replace([np.inf, -np.inf], np.nan)

    # Median impute remaining NaNs (robust to the heavy tails in flow features).
    features = features.fillna(features.median(numeric_only=True))
    # Any column still all-NaN (no numeric content) is dropped.
    features = features.dropna(axis=1, how="all")

    dropped_constant: List[str] = []
    if drop_constant:
        nunique = features.nunique()
        dropped_constant = nunique[nunique <= 1].index.tolist()
        features = features.drop(columns=dropped_constant)

    n_dedup_removed = 0
    if dedup:
        before = len(features)
        combined = pd.concat([features, labels.rename("__label__")], axis=1)
        keep = ~combined.duplicated()
        features = features[keep].reset_index(drop=True)
        metadata = metadata[keep].reset_index(drop=True)
        labels = labels[keep].reset_index(drop=True)
        n_dedup_removed = before - len(features)

    return CleanResult(
        features=features,
        metadata=metadata,
        labels=labels,
        dropped_constant=dropped_constant,
        n_dedup_removed=n_dedup_removed,
    )
