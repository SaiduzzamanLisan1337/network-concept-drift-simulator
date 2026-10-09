"""Synthetic, chronologically ordered network-flow data generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from concept_drift_simulator.config import FEATURES, PHASE_NAMES

# Standardized latent class separation in the initial concept. Each coordinate maps
# monotonically to one network-flow feature below.
_BASE_SEPARATION = np.array([0.95, 1.05, 0.75, 0.95, -0.95, 0.80, 1.05, -0.75])
# After concept drift, several previously informative signals reverse or weaken.
_DRIFTED_SEPARATION = np.array([-0.35, 0.35, -0.25, 0.25, 0.85, -0.30, -0.75, 0.90])
# A shared offset moves the observed feature distribution without changing the
# distance between benign and attack class centroids.
_COVARIATE_OFFSET = np.array([0.95, -0.75, 0.85, 0.30, -0.60, 0.55, -0.35, 0.80])
_GRADUAL_OFFSET = np.array([-0.35, 0.20, -0.50, 0.40, 0.35, -0.30, 0.45, -0.20])


@dataclass(frozen=True)
class PhaseBoundary:
    """Start/end positions for one phase in the stream."""

    phase: str
    start: int
    end: int


def _phase_boundaries(n_samples: int) -> list[PhaseBoundary]:
    points = np.linspace(0, n_samples, num=len(PHASE_NAMES) + 1, dtype=int)
    return [
        PhaseBoundary(phase=phase, start=int(points[i]), end=int(points[i + 1]))
        for i, phase in enumerate(PHASE_NAMES)
    ]


def _phase_parameters(phase: str, local_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prior, shared feature shift, and class-separation vectors."""
    n = len(local_fraction)
    zero = np.zeros((n, len(FEATURES)), dtype=float)
    base = np.repeat(_BASE_SEPARATION[None, :], n, axis=0)

    if phase == "stable":
        return np.full(n, 0.18), zero, base
    if phase == "covariate_shift":
        offset = np.repeat(_COVARIATE_OFFSET[None, :], n, axis=0)
        return np.full(n, 0.18), offset, base
    if phase == "concept_drift":
        offset = np.repeat(_COVARIATE_OFFSET[None, :], n, axis=0)
        separation = np.repeat(_DRIFTED_SEPARATION[None, :], n, axis=0)
        return np.full(n, 0.18), offset, separation
    if phase == "class_prior_shift":
        offset = np.repeat(_COVARIATE_OFFSET[None, :], n, axis=0)
        separation = np.repeat(_DRIFTED_SEPARATION[None, :], n, axis=0)
        return np.full(n, 0.42), offset, separation
    if phase == "gradual_transition":
        # Gradually move the data-generating process back toward the initial concept.
        alpha = np.clip(local_fraction, 0.0, 1.0)[:, None]
        offset_start = _COVARIATE_OFFSET
        offset_end = _GRADUAL_OFFSET
        offset = offset_start[None, :] * (1.0 - alpha) + offset_end[None, :] * alpha
        separation = _DRIFTED_SEPARATION[None, :] * (1.0 - alpha) + _BASE_SEPARATION[None, :] * alpha
        prior = 0.42 * (1.0 - local_fraction) + 0.18 * local_fraction
        return prior, offset, separation
    raise ValueError(f"Unknown phase: {phase}")


def _latent_to_features(latent: np.ndarray) -> pd.DataFrame:
    """Map latent values to positive, interpretable network-flow measurements."""
    # Clip latent values before exponentiation so very rare tails stay finite.
    z = np.clip(latent, -5.0, 5.0)

    def sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    return pd.DataFrame(
        {
            "flow_duration_ms": np.exp(5.1 + 0.55 * z[:, 0]),
            "src_bytes": np.exp(7.0 + 0.60 * z[:, 1]),
            "dst_bytes": np.exp(6.4 + 0.60 * z[:, 2]),
            "packet_count": np.exp(2.1 + 0.45 * z[:, 3]),
            "syn_ratio": sigmoid(1.15 * z[:, 4]),
            "dst_port_diversity": np.exp(1.1 + 0.50 * z[:, 5]),
            "failed_connection_ratio": sigmoid(1.20 * z[:, 6] - 0.25),
            "inter_arrival_ms": np.exp(2.4 + 0.65 * z[:, 7]),
        },
        columns=FEATURES,
    )


def _generate_from_parameters(
    n_samples: int,
    rng: np.random.Generator,
    prior: np.ndarray,
    offset: np.ndarray,
    separation: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    labels = (rng.random(n_samples) < prior).astype(np.int8)
    class_direction = (2 * labels - 1).astype(float)[:, None]
    # Class-conditional Gaussian feature distributions, with independent noise.
    latent = rng.normal(loc=0.0, scale=1.0, size=(n_samples, len(FEATURES)))
    latent += offset + class_direction * separation * 0.5
    return _latent_to_features(latent), labels


def make_baseline_training_data(n_samples: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate baseline training data before any drift occurs."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)
    prior = np.full(n_samples, 0.18)
    offset = np.zeros((n_samples, len(FEATURES)), dtype=float)
    separation = np.repeat(_BASE_SEPARATION[None, :], n_samples, axis=0)
    return _generate_from_parameters(n_samples, rng, prior, offset, separation)


def make_stream(n_samples: int, seed: int = 42) -> tuple[pd.DataFrame, list[PhaseBoundary]]:
    """Generate a time-ordered stream containing five distinct drift phases.

    The stream has an explicit event index, timestamp, phase name, and binary label
    (0 = benign, 1 = attack). It is synthetic and does not represent a real network.
    """
    if n_samples < len(PHASE_NAMES) * 2:
        raise ValueError(f"n_samples must be at least {len(PHASE_NAMES) * 2}.")

    rng = np.random.default_rng(seed)
    boundaries = _phase_boundaries(n_samples)
    feature_frames: list[pd.DataFrame] = []
    label_parts: list[np.ndarray] = []
    phase_parts: list[np.ndarray] = []

    for boundary in boundaries:
        count = boundary.end - boundary.start
        if count <= 0:
            continue
        local_fraction = np.linspace(0.0, 1.0, num=count, endpoint=True)
        prior, offset, separation = _phase_parameters(boundary.phase, local_fraction)
        frame, labels = _generate_from_parameters(count, rng, prior, offset, separation)
        feature_frames.append(frame)
        label_parts.append(labels)
        phase_parts.append(np.repeat(boundary.phase, count))

    features = pd.concat(feature_frames, ignore_index=True)
    labels = np.concatenate(label_parts)
    phases = np.concatenate(phase_parts)
    # The timestamp is a synthetic event clock, not the collection time of real traffic.
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=n_samples, freq="s")
    stream = features.copy()
    stream.insert(0, "timestamp", timestamps)
    stream.insert(0, "event_index", np.arange(n_samples, dtype=np.int64))
    stream["phase"] = phases
    stream["label"] = labels
    return stream, boundaries
