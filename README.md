# Network Concept Drift Simulator

[![Python CI](https://github.com/SaiduzzamanLisan1337/network-concept-drift-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/SaiduzzamanLisan1337/network-concept-drift-simulator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible cybersecurity research project that simulates **time-ordered network traffic whose statistical properties change over time**, then measures how those changes affect an intrusion-detection classifier. It compares a frozen baseline model with an incrementally updated model and produces CSV reports, drift indicators, and publication-friendly plots.

> **Research scope:** This is a synthetic-data simulator and experimental baseline, not a production network sensor or a validated intrusion-detection system. No real traffic is captured, and no external dataset download is required.

## Why this project matters

A machine-learning IDS is usually trained on historical traffic, but network behaviour, devices, services, and attacker tactics can change. A model that performs well on a static test set may degrade when the traffic distribution or the relationship between features and labels changes. This project makes those scenarios controllable and observable so that later continual-learning experiments have a reproducible starting point.

## What it includes

- **Five chronological phases:** stable baseline, covariate shift, concept drift, class-prior shift, and a gradual transition back toward the baseline concept.
- **Synthetic flow features:** flow duration, source/destination bytes, packet count, SYN ratio, destination-port diversity, failed-connection ratio, and inter-arrival time.
- **Frozen vs. incremental comparison:** two copies of the same baseline-fitted SGD logistic classifier; one stays fixed, while the other is updated after each evaluation window.
- **Prequential window evaluation:** a window is predicted and scored before its labels are used for the adaptive model's update.
- **IDS-appropriate metrics:** accuracy, balanced accuracy, attack precision/recall/F1, macro F1, and precision-recall AUC (average precision).
- **Drift measurements:** per-feature Population Stability Index (PSI), average PSI, standardized feature shift, and observed attack prevalence.
- **Reproducibility:** configurable seed and sizes, saved configuration and metadata, tests, and GitHub Actions CI.
- **Outputs:** per-window metrics, phase-level summary, per-feature drift metrics, per-event predictions, a generated stream CSV, charts, and a Markdown report.

## Drift phases

| Phase | What changes | What it helps investigate |
|---|---|---|
| `stable` | Baseline class-conditional distributions and 18% target attack prevalence | Initial reference behaviour |
| `covariate_shift` | A shared offset moves the flow-feature distributions | Sensitivity to changing feature distributions |
| `concept_drift` | Several benign/attack feature-separation directions change | Degradation when feature-label relationships change |
| `class_prior_shift` | Target attack prevalence rises from about 18% to about 42% | Effects of changing class balance on alert quality and metrics |
| `gradual_transition` | Features, class separation, and attack prevalence move gradually back toward baseline | Recovery and adaptation during gradual change |

The exact observed class rate varies stochastically around the configured target rate. Phase boundaries and event order are explicit in the stream output.

## Project structure

```text
network-concept-drift-simulator/
├── .github/workflows/ci.yml
├── concept_drift_simulator/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── plots.py
│   └── report.py
├── data/                         # optional local data; generated files ignored
├── outputs/                      # experiment output; generated files ignored
├── tests/
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- `pip`
- Internet access only for installing Python packages; the experiment itself runs locally without downloading a dataset

## Quick start

### 1. Clone your GitHub repository

Once you create the empty repository at `SaiduzzamanLisan1337/network-concept-drift-simulator`, run:

```bash
git clone https://github.com/SaiduzzamanLisan1337/network-concept-drift-simulator.git
cd network-concept-drift-simulator
```

If you are starting from the downloaded project folder instead, open a terminal in that folder and skip the clone step.

### 2. Create and activate a virtual environment

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Or install only runtime requirements:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the full experiment

```bash
python -m concept_drift_simulator
```

The default run generates 6,000 baseline training rows and evaluates 20,000 ordered stream rows in windows of 1,000. Output files are written to `outputs/`.

### 5. Inspect the results

Open `outputs/report.md` first. It summarises the run, reports mean metrics by phase, explains the label-availability assumption, and lists the generated files. Open the PNG files to compare the performance curves with the drift timeline.

## Useful commands

**Fast smoke run** (good before pushing code):

```bash
python -m concept_drift_simulator --n-train 600 --n-stream 1000 --window-size 100 --output-dir outputs_smoke
```

**Higher-resolution stream:**

```bash
python -m concept_drift_simulator --n-train 10000 --n-stream 50000 --window-size 500 --seed 123
```

**Change the random seed:**

```bash
python -m concept_drift_simulator --seed 7
```

**Disable generated stream CSV** (keeps metrics, predictions, report, and plots):

```bash
python -m concept_drift_simulator --no-stream-csv
```

**Skip figures for a faster run:**

```bash
python -m concept_drift_simulator --no-plots
```

**Evaluate only the frozen model:**

```bash
python -m concept_drift_simulator --static-only
```

**See all command-line options:**

```bash
python -m concept_drift_simulator --help
```

After installation, the equivalent console command is available:

```bash
network-drift --n-stream 30000 --window-size 500
```

## Push this project to GitHub

Create a **new empty repository** named `network-concept-drift-simulator` on GitHub. Do not add a README, licence, or `.gitignore` on GitHub, because this folder already includes them. Then, from a terminal opened in this project folder, run:

```bash
git init -b main
git add .
git status
git commit -m "Initial commit: network concept drift simulator"
git remote add origin https://github.com/SaiduzzamanLisan1337/network-concept-drift-simulator.git
git push -u origin main
```

If you cloned the empty repository first, do **not** run `git init` or add the remote again. Instead, copy these project files into the cloned folder (without replacing its `.git` directory), then run `git add .`, `git commit -m "Add concept drift simulator"`, and `git push`. GitHub may ask you to authenticate in your browser or through your configured Git credential manager; do not put a personal access token directly into the remote URL.

Before committing, run the smoke experiment and tests:

```bash
python -m concept_drift_simulator --n-train 600 --n-stream 1000 --window-size 100 --output-dir outputs_smoke
python -m pytest
git status
```

The `.gitignore` keeps generated datasets and experiment outputs out of the initial commit. Source code, tests, documentation, and the CI workflow are tracked.

## Output files

| File | Description |
|---|---|
| `report.md` | Run summary, phase-level tables, assumptions, and limitations |
| `window_metrics.csv` | One row per model per window, including performance and drift indicators |
| `phase_summary.csv` | Mean metrics grouped by phase and model |
| `feature_drift.csv` | Per-window PSI for each feature plus aggregate drift indicators |
| `synthetic_stream.csv` | Chronological synthetic records with `event_index`, timestamp, phase, features, and label |
| `predictions_frozen_sgd_logistic_classifier.csv` | Frozen model's event-level labels, predictions, and predicted attack probabilities |
| `predictions_incrementally_updated_sgd_logistic_classifier.csv` | Adaptive model's event-level predictions and scores |
| `experiment_config.json` | User-specified run configuration |
| `run_metadata.json` | Model names, phase boundaries, features, training time, and synthetic-data flag |
| `accuracy_over_time.png` | Accuracy over windows, with drift phases shaded |
| `balanced_accuracy_over_time.png` | Balanced accuracy over windows |
| `attack_recall_over_time.png` | Recall for the attack class over windows |
| `macro_f1_over_time.png` | Macro F1 over windows |
| `feature_drift_over_time.png` | Average feature PSI relative to the baseline |
| `attack_rate_over_time.png` | Observed attack prevalence over time |
| `feature_distributions.png` | Example feature distributions by phase |
| `confusion_matrices.png` | Whole-stream confusion matrices for available models |

Generated data, prediction files, reports, and figures are ignored by Git by default to prevent accidental commits of large or machine-specific run outputs. The `outputs/` folder is kept with `.gitkeep`. Commit selected results intentionally if your study or portfolio requires them.

## Methodology and interpretation

1. Generate baseline-labelled network-flow features and fit a `StandardScaler` and an SGD logistic classifier on baseline training data.
2. Generate the stream in chronological phase order, using a separate random seed from baseline training.
3. Transform each window using the scaler fitted on the baseline only. This deliberate choice keeps preprocessing fixed for both models so the comparison isolates classifier updating.
4. Score each model on a window and calculate detection metrics before revealing that window's labels to the adaptive model.
5. Update the adaptive model with `partial_fit` after scoring the current window. The frozen model is never updated.
6. Compare performance changes to distribution-shift and attack-rate indicators.

### Drift indicators

- **PSI:** each feature's current-window distribution is compared with baseline training data using fixed reference quantile bins. PSI is a monitoring heuristic; thresholds vary by context and it is not a significance test.
- **Mean standardized feature shift:** average absolute feature-mean displacement measured in baseline standard deviations.
- **Attack rate:** proportion of rows labelled as attacks in each window.

### Important evaluation assumption

The adaptive method receives the labels for a completed window immediately after that window is evaluated. This is a convenient simulation of delayed ground-truth availability. Real IDS labels can arrive late, be incomplete, or be noisy; a real deployment study should model that explicitly. Do not interpret this experiment as evidence that a model can automatically learn safely from unverified live traffic.

## Run the tests

```bash
python -m pytest
```

The GitHub Actions workflow runs the tests and a small end-to-end experiment on Python 3.10–3.13 for pushes and pull requests to `main` or `master`.

## Research extensions

This repository intentionally creates a small experimental foundation rather than claiming a complete continual-learning method. Sensible follow-up experiments include:

- Repeat runs across several seeds and report confidence intervals or paired statistical comparisons.
- Add drift detectors such as ADWIN or Page-Hinkley and specify exactly when adaptation is triggered.
- Compare supervised incremental updates with unsupervised/semi-supervised detection and delayed or noisy labels.
- Add replay buffers, forgetting controls, threshold calibration, and adaptation-cost measurements.
- Validate on a documented public flow dataset and report dataset provenance, label mapping, preprocessing, and leakage controls.
- Include attacks not represented during baseline training and distinguish novelty detection from ordinary concept drift.

## Security and ethics

Only synthetic examples are generated. The package does not sniff interfaces, connect to network devices, send packets, or include attack payloads. If adapting the workflow to real datasets, follow the dataset licence and institutional data-handling requirements, remove sensitive information, and document the limitations of each label.

## Citation

If you use this project in academic work, cite the repository release or commit you actually used and describe the generator, phase definitions, update protocol, and limitations in your methods section. The synthetic results should not be presented as measurements from an operational network.

## License

MIT. See [LICENSE](LICENSE).
