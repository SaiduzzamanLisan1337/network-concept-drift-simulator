# Network Concept Drift Simulator — Run Report

## Run configuration

- Baseline training rows: **6,000**
- Time-ordered stream rows: **20,000**
- Evaluation window size: **1,000**
- Random seed: **42**
- Baseline classifier training time: **0.004 seconds**
- Data source: **synthetic only**; no real network traffic was collected or used.

## Drift phases

- **stable**: events 0–3,999 (4,000 rows)
- **covariate_shift**: events 4,000–7,999 (4,000 rows)
- **concept_drift**: events 8,000–11,999 (4,000 rows)
- **class_prior_shift**: events 12,000–15,999 (4,000 rows)
- **gradual_transition**: events 16,000–19,999 (4,000 rows)

## Overall window-level results

The values below are the unweighted mean across evaluation windows. This is useful for comparing trends, but it is not the same as calculating one metric across all stream rows.

| Model                                         |   Mean accuracy |   Mean balanced accuracy |   Mean attack recall |   Mean macro F1 |   Lowest window accuracy |
|:----------------------------------------------|----------------:|-------------------------:|---------------------:|----------------:|-------------------------:|
| Frozen SGD logistic classifier                |           0.618 |                    0.595 |                0.553 |           0.581 |                    0.287 |
| Incrementally updated SGD logistic classifier |           0.617 |                    0.600 |                0.565 |           0.582 |                    0.292 |

## Results by phase

| phase              | model                                         |   accuracy_mean |   attack_recall_mean |   macro_f1_mean |   feature_drift_psi_mean |
|:-------------------|:----------------------------------------------|----------------:|---------------------:|----------------:|-------------------------:|
| stable             | Frozen SGD logistic classifier                |           0.925 |                0.742 |           0.871 |                    0.010 |
| stable             | Incrementally updated SGD logistic classifier |           0.925 |                0.753 |           0.872 |                    0.010 |
| covariate_shift    | Frozen SGD logistic classifier                |           0.890 |                0.890 |           0.838 |                    0.394 |
| covariate_shift    | Incrementally updated SGD logistic classifier |           0.883 |                0.899 |           0.830 |                    0.394 |
| concept_drift      | Frozen SGD logistic classifier                |           0.316 |                0.338 |           0.288 |                    0.685 |
| concept_drift      | Incrementally updated SGD logistic classifier |           0.309 |                0.347 |           0.284 |                    0.685 |
| class_prior_shift  | Frozen SGD logistic classifier                |           0.304 |                0.311 |           0.303 |                    0.602 |
| class_prior_shift  | Incrementally updated SGD logistic classifier |           0.314 |                0.321 |           0.313 |                    0.602 |
| gradual_transition | Frozen SGD logistic classifier                |           0.652 |                0.485 |           0.607 |                    0.203 |
| gradual_transition | Incrementally updated SGD logistic classifier |           0.655 |                0.505 |           0.613 |                    0.203 |

## Figures

- `accuracy_over_time.png`
- `balanced_accuracy_over_time.png`
- `attack_recall_over_time.png`
- `macro_f1_over_time.png`
- `feature_drift_over_time.png`
- `attack_rate_over_time.png`
- `feature_distributions.png`
- `confusion_matrices.png`

## How to interpret this run

- `feature_drift_psi_mean` summarizes how each window's feature distributions differ from the baseline training features. PSI is a heuristic indicator, not a statistical proof that drift occurred.
- `attack_rate` tracks the observed share of attack-labelled rows, making the class-prior phase visible.
- The frozen model is trained only on baseline data and never updated.
- The incremental model begins with the same fitted classifier. It predicts a whole window first; only after that window is scored are its labels used for `partial_fit`. The next window therefore benefits from the previous window's labels.
- The adaptive comparison assumes labels become available at the end of each window. This is an experimental assumption, not a claim that production IDS labels are immediately available.
- Accuracy can hide missed attacks under imbalance. Inspect attack recall, balanced accuracy, macro F1, and PR-AUC alongside accuracy.

## Limitations

This project demonstrates controlled drift mechanisms; it is not a validated deployment-ready IDS. Synthetic feature distributions are simplified and do not capture all protocol, host, temporal, or adversarial behaviours found in real networks. The simulated timestamp is an event-ordering aid, not a real capture time. Results are seed- and configuration-dependent. A research study should later validate the evaluation design on one or more documented real-world flow datasets, check label quality, and include repeated seeds or statistical uncertainty estimates.

## Output files

- `window_metrics.csv` — per-window metrics for each model.
- `phase_summary.csv` — averages grouped by phase and model.
- `feature_drift.csv` — per-window PSI values per feature plus aggregate drift indicators.
- `synthetic_stream.csv` — generated stream, if stream saving is enabled.
- `predictions_*.csv` — per-event labels, predictions, and attack probabilities.
- `experiment_config.json` and `run_metadata.json` — reproducibility metadata.
- `report.md` — this report.
