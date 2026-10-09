import numpy as np
import pandas as pd

from concept_drift_simulator.config import FEATURES
from concept_drift_simulator.metrics import classification_metrics, feature_drift_metrics


def test_classification_metrics_return_expected_range():
    y = np.array([0, 0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 1, 0])
    score = np.array([0.1, 0.2, 0.9, 0.4, 0.8, 0.3])
    results = classification_metrics(y, pred, score)
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["attack_recall"] <= 1
    assert 0 <= results["pr_auc"] <= 1


def test_feature_drift_metric_is_finite_for_shifted_data():
    rng = np.random.default_rng(4)
    reference = pd.DataFrame(rng.normal(size=(500, len(FEATURES))), columns=FEATURES)
    current = reference.copy() + 1.5
    results = feature_drift_metrics(reference, current)
    assert results["feature_drift_psi_mean"] > 0
    assert results["mean_standardized_feature_shift"] > 0
