"""Experiment configuration and feature definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

FEATURES = [
    "flow_duration_ms",
    "src_bytes",
    "dst_bytes",
    "packet_count",
    "syn_ratio",
    "dst_port_diversity",
    "failed_connection_ratio",
    "inter_arrival_ms",
]

PHASE_NAMES = [
    "stable",
    "covariate_shift",
    "concept_drift",
    "class_prior_shift",
    "gradual_transition",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters controlling a reproducible experiment."""

    n_train: int = 6_000
    n_stream: int = 20_000
    window_size: int = 1_000
    seed: int = 42
    output_dir: str = "outputs"
    save_stream: bool = True
    make_plots: bool = True
    adaptation: bool = True

    def validate(self) -> None:
        if self.n_train < 100:
            raise ValueError("n_train must be at least 100 to train a useful baseline model.")
        if self.n_stream < len(PHASE_NAMES) * 10:
            raise ValueError(
                f"n_stream must be at least {len(PHASE_NAMES) * 10} so every drift phase has samples."
            )
        if self.window_size < 10:
            raise ValueError("window_size must be at least 10.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    def to_dict(self) -> dict:
        """Return a JSON-serializable view of this configuration."""
        return asdict(self)
