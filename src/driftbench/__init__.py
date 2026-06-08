"""driftbench -- offline drift-aware evaluation for the conference paper.

Scope (deliberately a subset of the Q1 study):
  * datasets: CSE-CIC-IDS2018 + CIC-DDoS2019 (flow / CICFlowMeter)
  * drift: PSI, KS (offline, static window-to-window)
  * adaptation: calibration buffer only (recalibration, not retraining)

Held back for the Q1 paper: TON_IoT + CICIoT2023, KL directional analysis,
online detectors (ADWIN/DDM/EDDM), and the full retrain-vs-recalibrate policy.
"""
from . import config, manifest, cleaning, harmonise, splits, drift, metrics, models, bootstrap

from .config import DATASETS, SEEDS, SEED_ANCHOR, ensure_dirs
from .cleaning import clean_flows, CleanResult
from .harmonise import common_core, to_common_core, harmonise_labels, shared_families
from .splits import random_split, chronological_split, leave_one_day_out
from .drift import psi, ks_stat, feature_drift_report, label_psi, psi_band
from .metrics import compute_metrics, metrics_frame, expected_calibration_error
from .models import make_model, recalibrate, time_ordered_buffer, MODEL_NAMES
from .bootstrap import bootstrap_metric_ci, bootstrap_gap_ci, multi_seed, fmt_ci

__version__ = "0.1.0"

__all__ = [
    "config", "manifest", "cleaning", "harmonise", "splits", "drift",
    "metrics", "models", "bootstrap",
    "DATASETS", "SEEDS", "SEED_ANCHOR", "ensure_dirs",
    "clean_flows", "CleanResult",
    "common_core", "to_common_core", "harmonise_labels", "shared_families",
    "random_split", "chronological_split", "leave_one_day_out",
    "psi", "ks_stat", "feature_drift_report", "label_psi", "psi_band",
    "compute_metrics", "metrics_frame", "expected_calibration_error",
    "make_model", "recalibrate", "time_ordered_buffer", "MODEL_NAMES",
    "bootstrap_metric_ci", "bootstrap_gap_ci", "multi_seed", "fmt_ci",
    "__version__",
]
