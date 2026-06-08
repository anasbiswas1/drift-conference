"""Central configuration: paths, seeds, and shared constants.

The phases referenced throughout (Phase 0..6) map to the experiment plan
and to the notebooks/ directory.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. On Colab, DATA_ROOT points at the mounted Drive folder; the raw CIC
# CSVs live there and are NEVER committed to the public repo.
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DRIFT_DATA_ROOT", "/content/drive/MyDrive/drift-conference/data"))
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"

# Results and manifests DO live in the repo (small, reproducibility artefacts).
REPO_ROOT = Path(os.environ.get("DRIFT_REPO_ROOT", "."))
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_DIR = REPO_ROOT / "manifests"

DATASETS = ("CSE-CIC-IDS2018", "CIC-DDoS2019")

# Pre-registered anchor seed plus a small fan-out for multi-seed reporting.
SEED_ANCHOR = 0
SEEDS = (0, 1, 2, 3, 4)

# PSI interpretation bands (monitoring convention; starting points, not laws).
PSI_SMALL = 0.10
PSI_MODERATE = 0.25

# Bootstrap settings (matches the established methodology: B=1000, 95% CI).
BOOTSTRAP_B = 1000
CI_LOW, CI_HIGH = 2.5, 97.5

# Label string that denotes benign / negative traffic (case-insensitive match).
BENIGN_TOKENS = ("benign", "normal")


def ensure_dirs() -> None:
    """Create the writable data/result directories if missing."""
    for d in (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)
