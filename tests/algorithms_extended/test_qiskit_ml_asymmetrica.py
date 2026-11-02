"""Tests for the full-base qiskit-machine-learning adaptation.

Covers: real functional correctness of the imported upstream classes,
the kernel-parity cross-check against this platform's own hand-rolled
fidelity kernel, and the three material-extension comparisons.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp

from algorithms_extended.qiskit_ml_asymmetrica.neural_networks import EstimatorQNN, SamplerQNN
from algorithms_extended.qiskit_ml_asymmetrica.classifiers import PegasosQSVC
from algorithms_extended.qiskit_ml_asymmetrica.kernels import FidelityQuantumKernel, TrainableFidelityQuantumKernel
from algorithms_extended.qiskit_ml_asymmetrica.kernels.algorithms import QuantumKernelTrainer
from algorithms_extended.qiskit_ml_asymmetrica.optimizers import QNSPSA, SPSA
from algorithms_extended.qiskit_ml_asymmetrica.finance_extensions import (
    kernel_parity_check,
    classifier_comparison,
    regressor_comparison,
)


def test_estimator_qnn_forward_shape():
    fm = ZZFeatureMap(2, reps=1)
    ansatz = RealAmplitudes(2, reps=1)
    qc = fm.compose(ansatz)
    qnn = EstimatorQNN(
        circuit=qc, observables=SparsePauliOp("ZZ"),
        input_params=fm.parameters, weight_params=ansatz.parameters,
    )
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, (5, 2))
    w = rng.uniform(0, np.pi, qnn.num_weights)
    out = qnn.forward(x, w)
    assert out.shape == (5, 1)
    # observable Z⊗Z has eigenvalues in [-1, 1]
    assert np.all(out >= -1.0 - 1e-9) and np.all(out <= 1.0 + 1e-9)


def test_fidelity_kernel_is_valid_gram_matrix():
    fm = ZZFeatureMap(3, reps=1)
    kernel = FidelityQuantumKernel(feature_map=fm)
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, (5, 3))
    K = kernel.evaluate(X)
    assert K.shape == (5, 5)
    assert np.allclose(K, K.T, atol=1e-8)
    assert np.allclose(np.diag(K), 1.0, atol=1e-6)
    eigvals = np.linalg.eigvalsh((K + K.T) / 2)
    assert eigvals.min() > -1e-6, "Gram matrix must be PSD (up to numerical noise)"


def test_kernel_parity_against_platform_hand_rolled_kernel():
    """The load-bearing test: two independent code paths (upstream Qiskit
    fidelity primitive vs. this platform's own statevector-based
    fidelity_kernel_element) computing the identical physical quantity
    must agree to near machine precision."""
    result = kernel_parity_check(n_samples=6, n_features=2, seed=0)
    assert result["agrees_within_1e-6"], f"max_abs_diff={result['max_abs_diff']}"
    assert result["max_abs_diff"] < 1e-9


def test_pegasos_qsvc_trains_and_predicts_valid_labels():
    fm = ZZFeatureMap(2, reps=1)
    kernel = FidelityQuantumKernel(feature_map=fm)
    pegasos = PegasosQSVC(quantum_kernel=kernel, C=1.0, num_steps=50, seed=0)
    rng = np.random.default_rng(2)
    X_train = rng.uniform(0, np.pi, (20, 2))
    y_train = rng.choice([-1, 1], size=20)
    pegasos.fit(X_train, y_train)
    X_test = rng.uniform(0, np.pi, (5, 2))
    preds = pegasos.predict(X_test)
    assert set(np.unique(preds)).issubset({-1, 1})
    assert preds.shape == (5,)


def test_qnspsa_differs_from_vanilla_spsa_update():
    """QNSPSA (quantum natural SPSA, accounts for the quantum Fisher
    information metric) must produce a genuinely different parameter
    update than vanilla SPSA on the same loss landscape -- otherwise the
    "natural gradient" claim is not actually implemented."""
    fm = ZZFeatureMap(2, reps=1)
    ansatz = RealAmplitudes(2, reps=1)
    qc = fm.assign_parameters(np.array([0.3, 0.7])).compose(ansatz)

    def loss(params):
        return float(np.sum(np.sin(params) ** 2))

    from algorithms_extended.qiskit_ml_asymmetrica.support.sampler import QMLSampler

    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    spsa_opt = SPSA(maxiter=5)
    qnspsa_opt = QNSPSA(QNSPSA.get_fidelity(ansatz, sampler=QMLSampler()), maxiter=5)

    spsa_result = spsa_opt.minimize(loss, x0=x0.copy())
    qnspsa_result = qnspsa_opt.minimize(loss, x0=x0.copy())

    assert not np.allclose(spsa_result.x, qnspsa_result.x), (
        "QNSPSA and vanilla SPSA produced identical trajectories -- "
        "the natural-gradient metric preconditioning is not taking effect"
    )


@pytest.mark.slow
def test_classifier_comparison_reports_all_three_honestly():
    result = classifier_comparison(seed=0)
    for acc in (result.pegasos_qsvc_accuracy, result.platform_quantumsvc_accuracy, result.pqk_fidelity_accuracy):
        assert 0.0 <= acc <= 1.0
    # Explicitly does NOT assert any method wins -- honest reporting only.


@pytest.mark.slow
def test_regressor_comparison_reports_both_honestly():
    result = regressor_comparison(seed=0)
    assert result.upstream_vqr_mse >= 0.0
    assert result.platform_qnn_regressor_mse >= 0.0
