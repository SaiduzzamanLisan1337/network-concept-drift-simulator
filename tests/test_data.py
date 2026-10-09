import numpy as np

from concept_drift_simulator.config import FEATURES, PHASE_NAMES
from concept_drift_simulator.data import make_baseline_training_data, make_stream


def test_stream_has_expected_schema_and_chronological_order():
    stream, boundaries = make_stream(1_000, seed=7)
    assert list(stream.columns) == ["event_index", "timestamp", *FEATURES, "phase", "label"]
    assert stream["event_index"].is_monotonic_increasing
    assert stream["timestamp"].is_monotonic_increasing
    assert set(stream["label"].unique()).issubset({0, 1})
    assert stream["phase"].tolist()[0] == PHASE_NAMES[0]
    assert stream["phase"].tolist()[-1] == PHASE_NAMES[-1]
    assert [item.phase for item in boundaries] == PHASE_NAMES


def test_stream_generator_is_reproducible():
    first, _ = make_stream(500, seed=10)
    second, _ = make_stream(500, seed=10)
    assert first.equals(second)


def test_baseline_training_data_is_reproducible_and_finite():
    x1, y1 = make_baseline_training_data(500, seed=21)
    x2, y2 = make_baseline_training_data(500, seed=21)
    assert x1.equals(x2)
    assert np.array_equal(y1, y2)
    assert np.isfinite(x1.to_numpy()).all()
    assert set(np.unique(y1)) == {0, 1}


def test_class_prior_shift_has_higher_attack_prevalence():
    stream, _ = make_stream(5_000, seed=15)
    stable_rate = stream.loc[stream["phase"] == "stable", "label"].mean()
    prior_shift_rate = stream.loc[stream["phase"] == "class_prior_shift", "label"].mean()
    assert stable_rate < 0.25
    assert prior_shift_rate > 0.35
    assert prior_shift_rate > stable_rate
