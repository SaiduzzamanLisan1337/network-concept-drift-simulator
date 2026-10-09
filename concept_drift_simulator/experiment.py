"""Run training, streaming evaluation, drift measurement, and report generation."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from concept_drift_simulator.config import ExperimentConfig, FEATURES
from concept_drift_simulator.data import PhaseBoundary, make_baseline_training_data, make_stream
from concept_drift_simulator.metrics import classification_metrics, feature_drift_metrics
from concept_drift_simulator.plots import create_plots
from concept_drift_simulator.report import write_report

STATIC_MODEL = "Frozen SGD logistic classifier"
ADAPTIVE_MODEL = "Incrementally updated SGD logistic classifier"


def _new_classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=0.0005,
        max_iter=1000,
        tol=1e-3,
        learning_rate="constant",
        eta0=0.004,
        random_state=seed,
        average=True,
    )


def run_experiment(config: ExperimentConfig) -> dict:
    """Execute the full simulation and save reproducible outputs to disk."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Network Concept Drift Simulator")
    print("=" * 72)
    print(f"Baseline training rows : {config.n_train:,}")
    print(f"Time-ordered stream    : {config.n_stream:,}")
    print(f"Evaluation window size : {config.window_size:,}")
    print(f"Random seed            : {config.seed}")

    x_train_df, y_train = make_baseline_training_data(config.n_train, seed=config.seed)
    stream, boundaries = make_stream(config.n_stream, seed=config.seed + 1)
    x_train = x_train_df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    classifier = _new_classifier(config.seed)
    fit_start = time.perf_counter()
    classifier.fit(x_train_scaled, y_train)
    training_seconds = time.perf_counter() - fit_start
    adaptive_classifier = copy.deepcopy(classifier)

    baseline_reference = x_train_df[FEATURES].copy()
    metric_rows: list[dict] = []
    feature_rows: list[dict] = []
    prediction_parts: dict[str, list[pd.DataFrame]] = {STATIC_MODEL: [], ADAPTIVE_MODEL: []}
    model_names = [STATIC_MODEL, ADAPTIVE_MODEL] if config.adaptation else [STATIC_MODEL]

    stream_feature_matrix = stream[FEATURES]
    for start in range(0, config.n_stream, config.window_size):
        end = min(start + config.window_size, config.n_stream)
        window = stream.iloc[start:end].copy()
        x_window = stream_feature_matrix.iloc[start:end].to_numpy(dtype=float)
        y_window = window["label"].to_numpy(dtype=int)
        x_scaled = scaler.transform(x_window)
        drift = feature_drift_metrics(baseline_reference, window)
        phase = str(window["phase"].mode().iloc[0])
        window_info = {
            "window_id": len(metric_rows) // len(model_names),
            "window_start": start,
            "window_end": end - 1,
            "window_midpoint": (start + end - 1) / 2,
            "n_samples": len(window),
            "phase": phase,
            **drift,
        }
        feature_rows.append(
            {
                "window_id": window_info["window_id"],
                "window_start": start,
                "window_end": end - 1,
                "window_midpoint": (start + end - 1) / 2,
                "phase": phase,
                **{f"psi_{feature}": drift[f"psi_{feature}"] for feature in FEATURES},
                "feature_drift_psi_mean": drift["feature_drift_psi_mean"],
                "mean_standardized_feature_shift": drift["mean_standardized_feature_shift"],
                "attack_rate": drift["attack_rate"],
            }
        )

        for model_name in model_names:
            active_model = classifier if model_name == STATIC_MODEL else adaptive_classifier
            scores = active_model.predict_proba(x_scaled)[:, 1]
            predictions = (scores >= 0.5).astype(int)
            metric_rows.append(
                {
                    **window_info,
                    "model": model_name,
                    **classification_metrics(y_window, predictions, scores),
                }
            )
            prediction_parts[model_name].append(
                pd.DataFrame(
                    {
                        "event_index": window["event_index"].to_numpy(),
                        "timestamp": window["timestamp"].astype(str).to_numpy(),
                        "phase": window["phase"].to_numpy(),
                        "label": y_window,
                        "prediction": predictions,
                        "attack_probability": scores,
                    }
                )
            )

        # A prequential/batch-online evaluation: predict the entire current window,
        # record metrics, and only then reveal its labels to the adaptive model.
        if config.adaptation:
            adaptive_classifier.partial_fit(x_scaled, y_window, classes=np.array([0, 1]))

    metrics = pd.DataFrame(metric_rows)
    feature_drift = pd.DataFrame(feature_rows)
    metrics.to_csv(output_dir / "window_metrics.csv", index=False)
    feature_drift.to_csv(output_dir / "feature_drift.csv", index=False)

    phase_summary = (
        metrics.groupby(["phase", "model"], sort=False)
        .agg(
            windows=("window_id", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            attack_recall_mean=("attack_recall", "mean"),
            attack_f1_mean=("attack_f1", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            feature_drift_psi_mean=("feature_drift_psi_mean", "mean"),
            attack_rate_mean=("attack_rate", "mean"),
        )
        .reset_index()
    )
    phase_summary.to_csv(output_dir / "phase_summary.csv", index=False)

    for model_name, parts in prediction_parts.items():
        if parts:
            filename = "predictions_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_") + ".csv"
            pd.concat(parts, ignore_index=True).to_csv(output_dir / filename, index=False)

    if config.save_stream:
        stream.to_csv(output_dir / "synthetic_stream.csv", index=False)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    run_metadata = {
        "baseline_training_seconds": training_seconds,
        "stream_rows": config.n_stream,
        "phase_boundaries": [boundary.__dict__ for boundary in boundaries],
        "models": model_names,
        "feature_names": FEATURES,
        "synthetic_data_only": True,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    plot_files = create_plots(stream, metrics, boundaries, output_dir) if config.make_plots else []
    write_report(output_dir, config, metrics, phase_summary, boundaries, training_seconds, plot_files)

    print("\nExperiment complete.")
    print(f"Baseline model training time: {training_seconds:.2f} seconds")
    print("Mean metrics by phase:")
    print(
        phase_summary[["phase", "model", "accuracy_mean", "attack_recall_mean", "macro_f1_mean"]]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nOutputs saved to: {output_dir.resolve()}")
    print("Read report.md for an interpretation guide.")
    return {
        "metrics": metrics,
        "phase_summary": phase_summary,
        "feature_drift": feature_drift,
        "stream": stream,
        "boundaries": boundaries,
        "output_dir": output_dir,
        "training_seconds": training_seconds,
    }
