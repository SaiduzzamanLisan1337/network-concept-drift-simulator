"""Plot experiment results with phase changes marked on the time axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from concept_drift_simulator.config import FEATURES, PHASE_NAMES
from concept_drift_simulator.data import PhaseBoundary


def _decorate_phase_axis(ax, boundaries: list[PhaseBoundary], n_stream: int) -> None:
    phase_colors = ["#f2f2f2", "#e7f1fb", "#fbe9e7", "#fff4d6", "#e8f5e9"]
    for i, boundary in enumerate(boundaries):
        ax.axvspan(boundary.start, boundary.end, color=phase_colors[i % len(phase_colors)], alpha=0.65, zorder=0)
        if i:
            ax.axvline(boundary.start, color="#555555", linestyle="--", linewidth=0.8, alpha=0.8)
        midpoint = (boundary.start + boundary.end) / 2
        ax.text(
            midpoint,
            1.015,
            boundary.phase.replace("_", "\n"),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
    ax.set_xlim(0, n_stream)
    ax.set_xlabel("Stream event index (chronological order)")
    ax.grid(axis="y", alpha=0.25)


def _plot_metric(
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    destination: Path,
    boundaries: list[PhaseBoundary],
    n_stream: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for model, group in metrics.groupby("model", sort=False):
        group = group.sort_values("window_start")
        ax.plot(group["window_midpoint"], group[metric], marker="o", markersize=3.8, linewidth=1.8, label=model)
    _decorate_phase_axis(ax, boundaries, n_stream)
    ax.set_title(title, pad=27, fontsize=13, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.04 if metric not in {"feature_drift_psi_mean", "mean_standardized_feature_shift"} else None)
    ax.legend(loc="best", frameon=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    stream: pd.DataFrame,
    metrics: pd.DataFrame,
    boundaries: list[PhaseBoundary],
    output_dir: str | Path,
) -> list[str]:
    """Write plots and return the generated filenames."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    n_stream = len(stream)
    plot_specs = [
        ("accuracy", "Detection accuracy over time", "Accuracy", "accuracy_over_time.png"),
        ("balanced_accuracy", "Balanced accuracy over time", "Balanced accuracy", "balanced_accuracy_over_time.png"),
        ("attack_recall", "Attack recall over time", "Attack recall", "attack_recall_over_time.png"),
        ("macro_f1", "Macro F1 over time", "Macro F1", "macro_f1_over_time.png"),
    ]
    written: list[str] = []
    for metric, title, ylabel, filename in plot_specs:
        destination = output_path / filename
        _plot_metric(metrics, metric, title, ylabel, destination, boundaries, n_stream)
        written.append(filename)

    drift_rows = metrics.drop_duplicates("window_start").sort_values("window_start")
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax.plot(drift_rows["window_midpoint"], drift_rows["feature_drift_psi_mean"], color="#7b3294", marker="o", markersize=3.5)
    _decorate_phase_axis(ax, boundaries, n_stream)
    ax.set_title("Feature-distribution drift relative to the baseline", pad=27, fontsize=13, weight="bold")
    ax.set_ylabel("Mean Population Stability Index (PSI)")
    ax.set_ylim(bottom=0)
    fig.savefig(output_path / "feature_drift_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append("feature_drift_over_time.png")

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax.plot(drift_rows["window_midpoint"], drift_rows["attack_rate"], color="#d95f02", marker="o", markersize=3.5)
    _decorate_phase_axis(ax, boundaries, n_stream)
    ax.set_title("Observed attack prevalence over time", pad=27, fontsize=13, weight="bold")
    ax.set_ylabel("Attack rate")
    ax.set_ylim(0, 1)
    fig.savefig(output_path / "attack_rate_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append("attack_rate_over_time.png")

    selected = ["src_bytes", "syn_ratio", "failed_connection_ratio"]
    fig, axes = plt.subplots(1, len(selected), figsize=(14, 5.2), constrained_layout=True)
    for ax, feature in zip(axes, selected):
        groups = []
        labels = []
        baseline = stream.loc[stream["phase"] == PHASE_NAMES[0], feature]
        groups.append(baseline.to_numpy()[: min(1000, len(baseline))])
        labels.append("stable")
        for phase in PHASE_NAMES[1:]:
            values = stream.loc[stream["phase"] == phase, feature].to_numpy()
            if len(values) > 1000:
                # Deterministic thinning keeps plot creation fast and repeatable.
                values = values[np.linspace(0, len(values) - 1, 1000, dtype=int)]
            groups.append(values)
            labels.append(phase.replace("_", "\n"))
        ax.boxplot(groups, showfliers=False, patch_artist=False)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(feature.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Example feature distributions across phases", fontsize=13, weight="bold")
    fig.savefig(output_path / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append("feature_distributions.png")

    # Confusion matrices aggregated across the full stream.
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    fig, axes = plt.subplots(1, metrics["model"].nunique(), figsize=(6 * metrics["model"].nunique(), 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, metrics["model"].drop_duplicates()):
        # Predictions are stored only for the test stream through a sibling CSV if enabled.
        prediction_file = output_path / f"predictions_{_safe_filename(model)}.csv"
        if prediction_file.exists():
            pred = pd.read_csv(prediction_file)
            matrix = confusion_matrix(pred["label"], pred["prediction"], labels=[0, 1])
            ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Attack"]).plot(ax=ax, colorbar=False, values_format="d")
            ax.set_title(model)
        else:
            ax.axis("off")
    fig.suptitle("Confusion matrices across the full stream", fontsize=13, weight="bold")
    fig.savefig(output_path / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append("confusion_matrices.png")
    return written


def _safe_filename(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
