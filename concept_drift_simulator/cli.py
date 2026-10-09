"""Command-line interface for the simulator."""

from __future__ import annotations

import argparse

from concept_drift_simulator.config import ExperimentConfig
from concept_drift_simulator.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="network-drift",
        description="Simulate network traffic concept drift and compare frozen vs. incremental learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-train", type=int, default=6_000, help="Number of baseline training records.")
    parser.add_argument("--n-stream", type=int, default=20_000, help="Number of chronologically ordered stream records.")
    parser.add_argument("--window-size", type=int, default=1_000, help="Records scored together in one evaluation window.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV, JSON, Markdown, and figure outputs.")
    parser.add_argument("--no-stream-csv", action="store_true", help="Do not save the generated stream to CSV.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG figure generation.")
    parser.add_argument("--static-only", action="store_true", help="Disable the incrementally updated comparison model.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        n_train=args.n_train,
        n_stream=args.n_stream,
        window_size=args.window_size,
        seed=args.seed,
        output_dir=args.output_dir,
        save_stream=not args.no_stream_csv,
        make_plots=not args.no_plots,
        adaptation=not args.static_only,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
