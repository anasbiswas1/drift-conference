"""Model family and calibration.

Models: Random Forest, LightGBM, and a 2-layer MLP (scaled). Consistent with
the IHCONCS NIDS paper so code and reviewer-facing methodology carry over.

Calibration: ``recalibrate`` fits a Platt (sigmoid) or isotonic calibrator on a
small time-ordered buffer of the *target* dataset -- recalibration only, no
weight retraining. This is the Demo B adaptation knob; full retrain-vs-
recalibrate policy is held back for the Q1 paper.
"""
from __future__ import annotations
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def make_model(name: str, seed: int = 0):
    """Return an unfitted estimator. name in {'rf','lgbm','mlp'}."""
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample",
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier  # imported lazily; optional dep
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
        )
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=300,
                random_state=seed)),
        ])
    raise ValueError(f"Unknown model '{name}'. Use 'rf', 'lgbm', or 'mlp'.")


MODEL_NAMES = ("rf", "lgbm", "mlp")


def recalibrate(fitted_model, X_buffer, y_buffer, method: str = "sigmoid"):
    """Wrap an already-fitted model with a calibrator trained on the buffer.

    method='sigmoid' = Platt scaling; method='isotonic' = isotonic regression.
    Uses cv='prefit' so the base model's weights are untouched.
    """
    try:
        # sklearn >= 1.6: cv="prefit" removed; freeze the fitted estimator instead.
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except ImportError:
        # sklearn < 1.6 fallback.
        calibrated = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    calibrated.fit(X_buffer, y_buffer)
    return calibrated


def time_ordered_buffer(order, frac: float):
    """First ``frac`` of a time-ordered index array -> (buffer_idx, rest_idx)."""
    n = len(order)
    k = max(int(n * frac), 1)
    return order[:k], order[k:]
