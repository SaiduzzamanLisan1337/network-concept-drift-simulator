# Contributing

Thanks for helping improve the Network Concept Drift Simulator.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"
```

## Before opening a pull request

```bash
python -m pytest
python -m concept_drift_simulator --n-train 600 --n-stream 1000 --window-size 100 --output-dir outputs_smoke
```

Please keep generation deterministic under a fixed seed, preserve chronological ordering, avoid evaluating a model on a window after updating on that same window, and document any change to the drift definitions. If new dependencies are introduced, update both `pyproject.toml` and `requirements.txt` when they are runtime dependencies.

## Suggested contributions

- Additional controlled drift types with tests and documentation.
- Repeated-seed analysis and uncertainty estimates.
- Drift detectors with explicit detection-delay evaluation.
- Label-delay and label-noise simulations.
- Evaluation adapters for carefully documented public network-flow datasets.

Do not commit private network captures, credentials, personal data, or generated large data files.
