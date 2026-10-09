"""Create a concise Markdown report for each simulation run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from concept_drift_simulator.config import ExperimentConfig
from concept_drift_simulator.data import PhaseBoundary


def write_report(
    output_dir: Path,
    config: ExperimentConfig,
    metrics: pd.DataFrame,
    phase_summary: pd.DataFrame,
    boundaries: list[PhaseBoundary],
    training_seconds: float,
    plot_files: list[str],
) -> None:
    """Write results, configuration, assumptions, and limitations to report.md."""
    by_model = metrics.groupby("model", sort=False)
    overall_rows = []
    for model, group in by_model:
        overall_rows.append(
            {
                "Model": model,
                "Mean accuracy": group["accuracy"].mean(),
                "Mean balanced accuracy": group["balanced_accuracy"].mean(),
                "Mean attack recall": group["attack_recall"].mean(),
                "Mean macro F1": group["macro_f1"].mean(),
                "Lowest window accuracy": group["accuracy"].min(),
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall_table = overall.to_markdown(index=False, floatfmt=".3f")
    phase_table = phase_summary[
        ["phase", "model", "accuracy_mean", "attack_recall_mean", "macro_f1_mean", "feature_drift_psi_mean"]
    ].to_markdown(index=False, floatfmt=".3f")

    plot_list = "\n".join(f"- `{name}`" for name in plot_files) if plot_files else "- Plot generation was disabled."
    phase_list = "\n".join(
        f"- **{boundary.phase}**: events {boundary.start:,}–{boundary.end - 1:,} ({boundary.end - boundary.start:,} rows)"
        for boundary in boundaries
    )
    report = f"""# Network Concept Drift Simulator — Run Report

## Run configuration

- Baseline training rows: **{config.n_train:,}**
- Time-ordered stream rows: **{config.n_stream:,}**
- Evaluation window size: **{config.window_size:,}**
- Random seed: **{config.seed}**
- Baseline classifier training time: **{training_seconds:.3f} seconds**
- Data source: **synthetic only**; no real network traffic was collected or used.

## Drift phases

{phase_list}

## Overall window-level results

The values below are the unweighted mean across evaluation windows. This is useful for comparing trends, but it is not the same as calculating one metric across all stream rows.

{overall_table}

## Results by phase

{phase_table}

## Figures

{plot_list}

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
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
