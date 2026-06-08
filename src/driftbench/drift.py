"""Phase 4 -- offline drift statistics.

Conference scope: PSI and KS (static, window-to-window). ``kl_divergence`` is
implemented but intentionally NOT part of the default report -- the directional
KL story is reserved for the Q1 extension.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import PSI_SMALL, PSI_MODERATE


def _bin_edges(ref: np.ndarray, bins: int) -> np.ndarray:
    """Quantile edges from the reference distribution (robust to heavy tails)."""
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref, qs))
    if edges.size < 2:                       # degenerate / constant feature
        edges = np.array([np.nanmin(ref), np.nanmax(ref) + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf    # absorb out-of-range target values
    return edges


def psi(ref: Sequence[float], tgt: Sequence[float], bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between reference and target samples."""
    ref = np.asarray(ref, float); tgt = np.asarray(tgt, float)
    ref = ref[~np.isnan(ref)]; tgt = tgt[~np.isnan(tgt)]
    if ref.size == 0 or tgt.size == 0:
        return float("nan")
    edges = _bin_edges(ref, bins)
    p = np.histogram(ref, edges)[0] / ref.size
    q = np.histogram(tgt, edges)[0] / tgt.size
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def psi_band(value: float) -> str:
    """Label a PSI value: small / moderate / significant."""
    if np.isnan(value):
        return "undefined"
    if value < PSI_SMALL:
        return "small"
    if value < PSI_MODERATE:
        return "moderate"
    return "significant"


def ks_stat(ref: Sequence[float], tgt: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (sup distance of ECDFs)."""
    ref = np.asarray(ref, float); tgt = np.asarray(tgt, float)
    ref = ref[~np.isnan(ref)]; tgt = tgt[~np.isnan(tgt)]
    if ref.size == 0 or tgt.size == 0:
        return float("nan")
    return float(ks_2samp(ref, tgt).statistic)


def kl_divergence(ref: Sequence[float], tgt: Sequence[float], bins: int = 10, eps: float = 1e-6) -> float:
    """Directional KL(P_ref || Q_tgt) on binned distributions. (Q1 extension.)"""
    ref = np.asarray(ref, float); tgt = np.asarray(tgt, float)
    ref = ref[~np.isnan(ref)]; tgt = tgt[~np.isnan(tgt)]
    edges = _bin_edges(ref, bins)
    p = np.histogram(ref, edges)[0] / ref.size
    q = np.histogram(tgt, edges)[0] / tgt.size
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def feature_drift_report(
    ref_df: pd.DataFrame, tgt_df: pd.DataFrame, features: Sequence[str], bins: int = 10
) -> pd.DataFrame:
    """Per-feature PSI + KS, ranked by PSI descending. This is the table that
    turns the drift taxonomy into evidence by lining drift up against the
    performance drop."""
    rows = []
    for f in features:
        p = psi(ref_df[f].values, tgt_df[f].values, bins=bins)
        rows.append({"feature": f, "psi": p, "psi_band": psi_band(p),
                     "ks": ks_stat(ref_df[f].values, tgt_df[f].values)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    return out


def label_psi(ref_labels: pd.Series, tgt_labels: pd.Series, eps: float = 1e-6) -> float:
    """PSI over the (categorical) label distribution -- label-distribution drift."""
    cats = sorted(set(ref_labels.astype(str)) | set(tgt_labels.astype(str)))
    p = ref_labels.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).values
    q = tgt_labels.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).values
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))
