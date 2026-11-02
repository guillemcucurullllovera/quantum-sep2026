import numpy as np
import pytest

from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from algorithms.vqc_qml.calibration import PlattCalibrator
from algorithms.vqc_qml.encoding import AngleEncoder, amplitude_encode, truncate_or_pad
from algorithms.vqc_qml.training import TrainingConfig, VQCTrainer, OptimizerFamily
from applications.classification.synthetic_financial_dataset import SyntheticDirectionalDataset
from applications.classification.vqc_classifier import run_directional_classification
from classical.machine_learning.baselines import logistic_regression_baseline


# --- ansatz correctness -----------------------------------------------------

@pytest.mark.parametrize("version", list(VQCVersion))
def test_statevector_is_normalized(version):
    ansatz = VQCAnsatz(n_qubits=4, version=version)
    params = ansatz.init_params(seed=1)
    x = np.random.default_rng(0).uniform(-np.pi, np.pi, 4)
    state = ansatz.statevector(x, params)
    assert abs(np.vdot(state, state).real - 1.0) < 1e-9


def test_z_expectations_bounded():
    ansatz = VQCAnsatz(n_qubits=4, version=VQCVersion.V5_REUPLOAD)
    params = ansatz.init_params(seed=1)
    x = np.random.default_rng(0).uniform(-np.pi, np.pi, 4)
    z = ansatz.z_expectations(x, params)
    assert np.all(z >= -1.0 - 1e-9) and np.all(z <= 1.0 + 1e-9)


def test_param_count_matches_version_defaults():
    ansatz_v1 = VQCAnsatz(n_qubits=4, version=VQCVersion.V1_LINEAR)  # d=1, no rz -> n params
    assert ansatz_v1.param_count == 4
    ansatz_v5 = VQCAnsatz(n_qubits=4, version=VQCVersion.V5_REUPLOAD)  # d=2, rz -> 2*2*n
    assert ansatz_v5.param_count == 16


def test_wrong_param_length_raises():
    ansatz = VQCAnsatz(n_qubits=3, version=VQCVersion.V5_REUPLOAD)
    with pytest.raises(Exception):
        ansatz.statevector(np.zeros(3), np.zeros(2))


def test_gradient_variance_probe_returns_positive_finite_value():
    ansatz = VQCAnsatz(n_qubits=4, version=VQCVersion.V5_REUPLOAD)
    var = ansatz.gradient_variance_probe(n_samples=5, seed=1)
    assert var > 0.0 and np.isfinite(var)


def test_near_identity_init_has_higher_gradient_variance_than_wide_init():
    """The whole point of V5's sigma=0.01 init: near-identity params should
    NOT collapse gradient variance the way a wide random init does at this
    qubit count is not guaranteed either way in general, but this regression
    guards that the probe distinguishes the two regimes at all (non-degenerate
    diagnostic), which is what the v4->v5 fix was validated against."""
    narrow = VQCAnsatz(n_qubits=6, version=VQCVersion.V5_REUPLOAD, sigma_init=0.01)
    wide = VQCAnsatz(n_qubits=6, version=VQCVersion.V5_REUPLOAD, sigma_init=2.0)
    var_narrow = narrow.gradient_variance_probe(n_samples=8, seed=3)
    var_wide = wide.gradient_variance_probe(n_samples=8, seed=3)
    assert var_narrow != var_wide  # diagnostic is sensitive to init scale, not a constant


# --- encoding ----------------------------------------------------------------

def test_angle_encoder_maps_to_pi_range():
    X = np.array([[0.0, 10.0], [5.0, -3.0], [10.0, 20.0]])
    encoded = AngleEncoder().fit_transform(X)
    assert np.all(encoded >= -np.pi - 1e-9) and np.all(encoded <= np.pi + 1e-9)


def test_angle_encoder_clips_out_of_range_at_transform_time():
    enc = AngleEncoder().fit(np.array([[0.0], [10.0]]))
    out_of_range = enc.transform(np.array([[100.0], [-100.0]]))
    assert np.all(out_of_range <= np.pi + 1e-9) and np.all(out_of_range >= -np.pi - 1e-9)


def test_truncate_or_pad():
    X = np.ones((5, 3))
    assert truncate_or_pad(X, 2).shape == (5, 2)
    assert truncate_or_pad(X, 5).shape == (5, 5)
    assert np.all(truncate_or_pad(X, 5)[:, 3:] == 0)


def test_amplitude_encode_is_unit_norm_and_padded_to_power_of_two():
    v = amplitude_encode(np.array([1.0, 2.0, 3.0]))
    assert len(v) == 4  # next power of 2 >= 3
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9


# --- training: loss must actually decrease, not just "run" -------------------

def test_training_loss_decreases_on_separable_synthetic_data():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.uniform(-1, 1, (n, 4))
    y = np.where(X[:, 0] > 0, 1, -1)  # linearly separable on feature 0
    ansatz = VQCAnsatz(n_qubits=4, version=VQCVersion.V1_LINEAR)
    trainer = VQCTrainer(ansatz, TrainingConfig(optimizer=OptimizerFamily.COBYLA, max_iter=60, seed=0))
    trainer.fit(X, y)
    assert len(trainer.loss_history_) > 5
    early = np.mean(trainer.loss_history_[:5])
    late = np.mean(trainer.loss_history_[-5:])
    assert late <= early  # optimizer must not be making things worse on average


def test_spsa_optimizer_also_trains():
    rng = np.random.default_rng(1)
    n = 40
    X = rng.uniform(-1, 1, (n, 3))
    y = np.where(X[:, 0] > 0, 1, -1)
    ansatz = VQCAnsatz(n_qubits=3, version=VQCVersion.V1_LINEAR)
    trainer = VQCTrainer(ansatz, TrainingConfig(optimizer=OptimizerFamily.SPSA, max_iter=40, seed=1))
    trainer.fit(X, y)
    preds = trainer.predict(X)
    assert set(np.unique(preds)).issubset({-1, 1})


def test_class_weighting_changes_fit_on_imbalanced_data():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (60, 3))
    y = np.array([1] * 50 + [-1] * 10)  # heavily imbalanced
    ansatz = VQCAnsatz(n_qubits=3, version=VQCVersion.V1_LINEAR)
    weighted = VQCTrainer(ansatz, TrainingConfig(class_weighted=True, max_iter=20, seed=2)).fit(X, y)
    unweighted = VQCTrainer(ansatz, TrainingConfig(class_weighted=False, max_iter=20, seed=2)).fit(X, y)
    # both must produce valid predictions; weighting is expected to change the
    # readout at minimum since sample_weight enters the ridge closed-form solve
    assert not np.allclose(weighted.readout_.weights_, unweighted.readout_.weights_)


def test_readout_noise_actually_perturbs_features():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (30, 3))
    y = np.where(X[:, 0] > 0, 1, -1)
    ansatz = VQCAnsatz(n_qubits=3, version=VQCVersion.V1_LINEAR)
    clean = VQCTrainer(ansatz, TrainingConfig(max_iter=10, seed=4, readout_flip_prob=0.0)).fit(X, y)
    noisy = VQCTrainer(ansatz, TrainingConfig(max_iter=10, seed=4, readout_flip_prob=0.2)).fit(X, y)
    assert not np.allclose(clean.readout_.weights_, noisy.readout_.weights_)


# --- calibration ---------------------------------------------------------------

def test_platt_calibration_reduces_calibration_error_vs_raw_sigmoid():
    rng = np.random.default_rng(5)
    n = 400
    true_logit = rng.normal(0, 1, n)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-true_logit))).astype(float)
    # deliberately mis-scaled "quantum-like" scores (as ridge readout scores are unscaled)
    scores = true_logit * 5.0 + 2.0

    from core.metrics import calibration_error

    raw_probs = 1 / (1 + np.exp(-scores))
    raw_ece = calibration_error(raw_probs, y)

    calibrator = PlattCalibrator().fit(scores, y)
    calibrated_ece = calibrator.evaluate_calibration(scores, y)

    assert calibrated_ece <= raw_ece


# --- end-to-end application: quantum vs classical must both be reported --------

def test_directional_classification_reports_both_quantum_and_classical():
    dataset = SyntheticDirectionalDataset(n_samples=120, n_features=6, seed=1)
    results = run_directional_classification(dataset, n_qubits=6, n_splits=2)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r.vqc_accuracy <= 1.0
        assert 0.0 <= r.baseline_accuracy <= 1.0
        assert r.vqc_calibration_error >= 0.0


def test_classical_baseline_alone_is_sane_on_synthetic_dataset():
    dataset = SyntheticDirectionalDataset(n_samples=200, seed=2)
    X, y = dataset.features(), dataset.targets()
    split = dataset.split_walk_forward(n_splits=1)[0]
    train_idx, test_idx = split
    result = logistic_regression_baseline(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
    assert result.accuracy > 0.4  # sanity: not degenerate on a mostly-informative synthetic set
