# Label-Efficient Adaptation of Flow-Based Intrusion Detectors Under Cross-Corpus Shift

Code and result tables for the study *Calibrate, Rethreshold, or Retrain? Label-Efficient
Adaptation of Flow-Based Intrusion Detectors Under Cross-Corpus Distribution Shift*.

A detector trained on one network and deployed on another usually fails. An operator who can
afford to label a small sample of local traffic must decide what to do with those labels:
recalibrate the deployed model, move its operating point, or discard it and retrain. This
repository contains everything needed to reproduce the answer.

The study covers **four corpora** from the standardised NetFlow v2 collection, converted by one
feature extractor to one 43-feature schema, so that transfer failure cannot be attributed to
tooling. It spans twelve directed corpus pairs, three model families, three seeds and five label
budgets from 0.01% to 10% of the target training partition.

## Headline results

- Zero-shot transfer collapses in 33 of 36 pair-model cells.
- Half of all transferred rankings are **inverted**: a decreasing transform of the source scores
  discriminates the target better than any increasing one. Ablation implicates features whose
  association with the label reverses between corpora.
- Because every standard post-hoc calibrator is monotone in the score, an exact ceiling bounds
  what any of them can achieve. That ceiling averages 0.273 MCC; Platt scaling reaches 0.011.
- Retraining on 144 to 217 labelled flows reaches 0.707 MCC and on 2,000 flows 0.876.
- Coverage-selected acquisition beats a label-blind uniform draw by 0.186 MCC at the smallest
  budget, and matches an oracle draw that needs the attack family of every candidate flow.
- Two boundaries qualify the recommendation: a small retrained model does not cover attack
  families absent from its sample, and the same buffer is too small to fix an operating point
  even when the threshold is chosen on held-out labels.

## Repository layout

```
notebooks/        numbered, run in order; each is self-contained and resumable
src/driftbench/   reusable code: cleaning, splits, metrics, drift, bootstrap
results/          every metric table the paper reports (committed)
manifests/        dataset manifest: source, row counts, label mapping
tools/            commit helper used from a scratch console
```

### Notebooks

| Notebook | What it produces |
|---|---|
| `09_prepare_corpora` | Streams and prepares NF-ToN-IoT-v2 and NF-BoT-IoT-v2; verifies the four-corpus schema |
| `10_four_corpus_study` | Transfer matrix, zero-shot baseline, buffer-only and augment strategies |
| `11_exact_sweep_rethreshold` | Exact threshold sweep, rethresholding, LightGBM in-domain diagnostic |
| `12_acquisition_selector` | Two-sided ceiling, acquisition rules, ceiling-gated selector |
| `13_source_use_baselines_selector` | Fine-tuning, importance weighting, INSOMNIA-style and iterative-AL baselines |
| `14_cost_operating_points_durability` | Timing, operating points, fine-tune update sweep, attack-family holdout |
| `15_inversion_mechanism_replication` | Sign-reversal ablation with a matched control; replication of the update sweep |
| `16_alignment_predictor` | Covariance decomposition for linear scores, and its limits |
| `17_deployable_operating_points` | Held-out thresholding, label-blind acquisition, three definitions of inversion |
| `18_full_pipeline` | The complete recommended procedure end to end, with per-cell label accounting |
| `19_reserved_threshold_sample` | Reserving a representative threshold sample: selection bias versus sample size |

Notebooks `00` to `08` are the earlier two-corpus work on CSE-CIC-IDS2018 and CIC-DDoS2019 that
preceded this study. They are retained for provenance and are not part of the four-corpus results.

### Result files

All tables are in `results/fourcorpus/`. The mapping from each reported item to the file and
notebook that produced it is in Appendix C of the paper's supplementary material.

| File | Contents |
|---|---|
| `fc_results.csv` | In-domain, zero-shot, buffer-only and augment across seeds |
| `fc_results_v2.csv` to `fc_results_v9.csv` | Successive strategy sets, one file per notebook |
| `fc_matrix_aggregate.csv`, `fc_matrix_bootstrap.csv` | Transfer matrix and bootstrap intervals |
| `fc_inversion_*.csv` | Sign-reversal features, causal drop test, failed predictor |
| `fc_orientation.csv` | Covariance sign, AUROC and best-MCC orientation per cell |
| `fc_pipeline.csv`, `fc_reserved_threshold.csv` | End-to-end pipeline and threshold-sample design |
| `fc_cost.csv`, `fc_operating_points.csv`, `fc_fpr.csv` | Timing, operating points, false-alarm burden |

## Data

**The corpora are not committed here.** The NetFlow v2 releases are distributed by the University
of Queensland; we compute on deduplicated mirrors that exclude the source and destination IP
columns and the removed IP modelling features, so flow counts are lower than the published ones.
Deduplication matters: duplicate flows in the ToN-IoT family are a documented defect, and
duplicates straddling a train-test split are a leakage channel of exactly the kind this study
measures. `manifests/dataset_manifest.json` records the source, row counts and label mapping for
each corpus.

## Reproducing

Notebooks assume Google Colab with the corpora in Drive under `drift-conference/data/nfv2/cache`
as prepared parquet. Each numbered notebook is resumable: results append to CSV and completed
cells are skipped on restart, so an interrupted session can be continued without repeating work.

```bash
pip install -r requirements.txt
```

Run the notebooks in order from `09`. Every experiment is seeded; where a result is reported at a
single seed, the paper says so.

## Citation

The manuscript is under review. Until it appears, please cite this repository.

## Licence

Code is released under the licence in `LICENSE`. The corpora carry their own terms from their
respective providers.
