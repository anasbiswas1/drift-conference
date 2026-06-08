# Concept Drift in Cybersecurity Datasets — Conference Experiment

Offline, drift-aware evaluation of two flow-based intrusion-detection benchmarks
(**CSE-CIC-IDS2018** and **CIC-DDoS2019**). This repository is the **conference
(lightweight) version** — a deliberately scoped subset of a larger Q1 journal
study in preparation.

## What this repo does

Two contained experiments that turn a "drift matters" *assertion* into a
*measured* result:

- **Demo A — within-dataset temporal leakage (2018).** Random vs chronological
  vs leave-one-day-out splits. The macro-F1 gap is the headline finding.
- **Demo B — cross-regime transfer (2018 → 2019).** Zero-shot transfer on the
  common CICFlowMeter feature core, then recalibration on a small time-ordered
  buffer (1/5/10%) of the target.

Drift is quantified with **PSI** and **KS**; every headline gap is reported with
a **bootstrap 95% CI (B=1000)** and **multi-seed mean ± std**.

### Held back for the Q1 study (intentionally out of scope here)

TON_IoT + CICIoT2023, schema/modality harmonisation, KL directional analysis,
online detectors (ADWIN/DDM/EDDM), and the full retrain-vs-recalibrate
adaptation policy. Keeping these out makes the conference paper a clean subset,
not a duplicate, which keeps the journal submission clear of prior-publication
overlap.

## The data ↔ code contract

**The raw CIC CSVs are never committed here** — they are large and the CIC
licence restricts redistribution. They live on Google Drive; this repo ships
everything needed to reproduce against them:

```
Google Drive (private)                 GitHub (public, this repo)
  drift-conference/data/                 src/driftbench/   reusable code
    raw/                                  notebooks/        00 → 05 phases
      CSE-CIC-IDS2018/*.csv               results/          metric & drift tables (committed)
      CIC-DDoS2019/*.csv                  manifests/        checksums + row counts + source URLs
    interim/   (written by nb 00)         paper/            manuscript + figures
    processed/ (written by nb 01)
```

A third party downloads the datasets themselves from the official sources,
runs notebook `00`, and `verify_manifest` confirms byte-identical inputs.

Official sources:
- CSE-CIC-IDS2018 — https://www.unb.ca/cic/datasets/ids-2018.html
- CIC-DDoS2019 — https://www.unb.ca/cic/datasets/ddos-2019.html

## Install (Colab or local)

```bash
pip install "git+https://github.com/USERNAME/drift-conference.git@main"
# with the LightGBM baseline:
pip install "driftbench[lgbm] @ git+https://github.com/USERNAME/drift-conference.git@main"
```

> Replace `USERNAME` (and the `@main` ref — pin to a commit/tag for a frozen
> reproduction) in this README, in `pyproject.toml`, and in each notebook's
> first cell before pushing.

## Run order

| Notebook | Phase | Output |
|---|---|---|
| `00_hygiene.ipynb`   | manifest + cleaning            | `interim/<ds>/{features,metadata,labels}.parquet` |
| `01_harmonise.ipynb` | common-core + family labels    | `processed/<ds>/{features_core,labels_family}.parquet` |
| `02_demoA.ipynb`     | within-dataset leakage         | `results/demoA_*.csv` |
| `03_demoB.ipynb`     | cross-regime transfer + buffer | `results/demoB_*.csv` |
| `04_drift.ipynb`     | PSI / KS reports               | `results/drift_*.csv` |
| `05_rigour.ipynb`    | bootstrap CIs + multi-seed     | console / `results/` |

Models: `rf`, `lgbm`, `mlp` (the notebooks set `MODEL=` near the top — rerun per model).

## Reproducibility

- Anchor seed `0`; multi-seed fan-out `0–4` (see `driftbench.config.SEEDS`).
- Bootstrap `B=1000`, 2.5/97.5 percentile CIs.
- Pin the install to a commit/tag for a frozen reproduction.

## Licence

Code: **MIT** (see `LICENSE`). The CIC datasets are **not** covered by this
licence and remain governed by their own terms — use for lawful research,
education, and defensive security evaluation only.
