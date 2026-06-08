"""The metric panel.

Lead the headline table with macro-F1 + MCC + ECE -- the trio that exposes
accuracy holding while balanced performance and calibration degrade under drift.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

from .config import BENIGN_TOKENS


def _is_benign(label: str) -> bool:
    s = str(label).strip().lower()
    return any(t in s for t in BENIGN_TOKENS)


def expected_calibration_error(y_true, y_pred, confidences, n_bins: int = 15) -> float:
    """Top-label ECE: |accuracy - confidence| averaged over confidence bins."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    conf = np.asarray(confidences, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc = np.mean(y_pred[m] == y_true[m])
        ece += (m.sum() / n) * abs(acc - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true, proba, classes) -> float:
    """Multiclass Brier score: mean squared error between one-hot truth and proba."""
    Y = label_binarize(y_true, classes=list(classes))
    if Y.shape[1] == 1:                      # binary edge-case from label_binarize
        Y = np.hstack([1 - Y, Y])
    proba = np.asarray(proba, float)
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def fp_rate(y_true, y_pred, timestamps: Optional[pd.Series] = None) -> Dict[str, float]:
    """False-positive rate treating any attack as positive, benign as negative.
    Adds FP/hour if a timestamp series is supplied."""
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    benign_true = np.array([_is_benign(v) for v in y_true])
    pred_attack = np.array([not _is_benign(v) for v in y_pred])
    n_benign = int(benign_true.sum())
    fp = int(np.sum(benign_true & pred_attack))
    out = {"fp": fp, "fp_rate": (fp / n_benign) if n_benign else float("nan")}
    if timestamps is not None:
        ts = pd.to_datetime(pd.Series(timestamps), errors="coerce").dropna()
        if len(ts) > 1:
            hours = max((ts.max() - ts.min()).total_seconds() / 3600.0, 1e-9)
            out["fp_per_hour"] = fp / hours
    return out


def compute_metrics(
    y_true, y_pred, proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence] = None, timestamps: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Full panel. ``proba`` (n_samples x n_classes aligned to ``classes``)
    enables AUROC/AUPRC/Brier/ECE; omit it for label-only metrics."""
    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    m: Dict[str, float] = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    m.update(fp_rate(y_true, y_pred, timestamps))

    if proba is not None and classes is not None:
        classes = [str(c) for c in classes]
        Y = label_binarize(y_true, classes=classes)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])
        try:
            m["auroc"] = roc_auc_score(Y, proba, average="macro", multi_class="ovr")
        except ValueError:
            m["auroc"] = float("nan")
        try:
            m["auprc"] = average_precision_score(Y, proba, average="macro")
        except ValueError:
            m["auprc"] = float("nan")
        m["brier"] = multiclass_brier(y_true, proba, classes)
        conf = proba.max(axis=1)
        pred_idx = proba.argmax(axis=1)
        pred_lbl = np.array([classes[i] for i in pred_idx])
        m["ece"] = expected_calibration_error(y_true, pred_lbl, conf)
    return m


def metrics_frame(rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """rows = {condition_name: metric_dict} -> tidy DataFrame for results/."""
    return pd.DataFrame(rows).T
