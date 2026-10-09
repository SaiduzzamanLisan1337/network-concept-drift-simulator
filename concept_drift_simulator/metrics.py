"""Window-level performance and feature-distribution drift metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from concept_drift_simulator.config import FEATURES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Calculate useful binary IDS metrics, keeping attack as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using fixed quantile bins from the reference."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return float("nan")

    quantiles = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    inner_edges = np.unique(quantiles[1:-1])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
    ref_count = np.histogram(reference, bins=edges)[0].astype(float)
    cur_count = np.histogram(current, bins=edges)[0].astype(float)
    # Small smoothing avoids undefined logarithms for empty bins.
    ref_pct = (ref_count + 1e-6) / (ref_count.sum() + 1e-6 * len(ref_count))
    cur_pct = (cur_count + 1e-6) / (cur_count.sum() + 1e-6 * len(cur_count))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift_metrics(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    """Return per-feature PSI and aggregate drift indicators for one window."""
    values = {f"psi_{feature}": _psi(reference[feature].to_numpy(), current[feature].to_numpy()) for feature in FEATURES}
    values["feature_drift_psi_mean"] = float(np.nanmean(list(values.values())))

    ref_mean = reference[FEATURES].mean().to_numpy(dtype=float)
    ref_std = reference[FEATURES].std(ddof=0).replace(0, np.nan).to_numpy(dtype=float)
    cur_mean = current[FEATURES].mean().to_numpy(dtype=float)
    standardized_shift = np.abs((cur_mean - ref_mean) / ref_std)
    values["mean_standardized_feature_shift"] = float(np.nanmean(standardized_shift))
    values["attack_rate"] = float(current["label"].mean()) if "label" in current else float("nan")
    return values
