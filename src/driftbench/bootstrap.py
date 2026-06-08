"""Phase 5 -- the rigour layer.

Bootstrap 95% CIs (B=1000, 2.5/97.5 percentile) on the metric GAPS, plus a
multi-seed runner. A gap reported as ``0.36 [0.31, 0.41]`` is what makes the
headline claim unassailable; bare point estimates invite "is this just variance?"
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple
import numpy as np

from .config import BOOTSTRAP_B, CI_LOW, CI_HIGH


def _resample(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, n, size=n)


def bootstrap_metric_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = BOOTSTRAP_B, seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile CI for a single condition's metric. Returns (point, lo, hi)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    samples = np.empty(B)
    for b in range(B):
        idx = _resample(rng, n)
        samples[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(samples, [CI_LOW, CI_HIGH])
    return float(point), float(lo), float(hi)


def bootstrap_gap_ci(
    eval_a: Dict[str, np.ndarray], eval_b: Dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = BOOTSTRAP_B, seed: int = 0,
) -> Dict[str, float]:
    """CI for (metric_a - metric_b) where a and b are evaluated on DIFFERENT
    test sets (e.g. random-split vs chronological-split). Each condition is
    resampled independently per bootstrap iteration.

    eval_a / eval_b: {'y_true': array, 'y_pred': array}.
    Returns {'gap', 'lo', 'hi'}.
    """
    ya, pa = np.asarray(eval_a["y_true"]), np.asarray(eval_a["y_pred"])
    yb, pb = np.asarray(eval_b["y_true"]), np.asarray(eval_b["y_pred"])
    rng = np.random.default_rng(seed)
    gap = metric_fn(ya, pa) - metric_fn(yb, pb)
    diffs = np.empty(B)
    for i in range(B):
        ia = _resample(rng, len(ya)); ib = _resample(rng, len(yb))
        diffs[i] = metric_fn(ya[ia], pa[ia]) - metric_fn(yb[ib], pb[ib])
    lo, hi = np.percentile(diffs, [CI_LOW, CI_HIGH])
    return {"gap": float(gap), "lo": float(lo), "hi": float(hi)}


def multi_seed(run_fn: Callable[[int], Dict[str, float]], seeds) -> Dict[str, Dict[str, float]]:
    """Run ``run_fn(seed)`` over seeds; return per-metric mean and std.

    run_fn must return a {metric: value} dict. Output:
    {metric: {'mean': .., 'std': .., 'values': [...]}}.
    """
    collected: Dict[str, List[float]] = {}
    for s in seeds:
        res = run_fn(s)
        for k, v in res.items():
            collected.setdefault(k, []).append(float(v))
    summary = {}
    for k, vals in collected.items():
        arr = np.asarray(vals, float)
        summary[k] = {"mean": float(np.nanmean(arr)),
                      "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
                      "values": vals}
    return summary


def fmt_ci(point: float, lo: float, hi: float, nd: int = 3) -> str:
    """Format as '0.360 [0.310, 0.410]' for tables and prose."""
    return f"{point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
