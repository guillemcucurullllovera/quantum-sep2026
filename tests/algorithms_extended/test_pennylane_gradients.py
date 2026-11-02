"""Tests for algorithms_extended/pennylane_asymmetrica/gradients/parameter_shift.py.

The core assertion is a genuine cross-implementation validation: two
completely independent statevector simulators --
``algorithms.vqc_qml.ansatz.VQCAnsatz`` (this platform's own hand-rolled
numpy tensor-contraction simulator) and Qiskit's ``Statevector`` (built from
a circuit assembled via this module's ``reupload_ansatz_circuit``, itself
built from the ported ``angle_embedding`` template) -- are both driven
through the *same* two-term parameter-shift-rule formula and checked to
agree to floating-point precision on the gradient of the same circuit
family.
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithms_extended.pennylane_asymmetrica.gradients.parameter_shift import (
    parameter_shift_gradient,
    reupload_ansatz_circuit,
    reupload_ansatz_z_expectations,
)
from algorithms.vqc_qml.ansatz import VQCAnsatz, VQCVersion
from core.errors import ValidationError


class TestReuploadAnsatzCrossValidation:
    @pytest.mark.parametrize(
        "version",
        [VQCVersion.V1_LINEAR, VQCVersion.V4_RING, VQCVersion.V5_REUPLOAD],
    )
    def test_z_expectations_match_vqcansatz(self, version):
        n_qubits = 3
        ansatz = VQCAnsatz(n_qubits=n_qubits, version=version)
        rng = np.random.default_rng(0)
        x = rng.uniform(-np.pi, np.pi, n_qubits)
        params = ansatz.init_params(seed=0)

        z_numpy = ansatz.z_expectations(x, params)
        z_qiskit = reupload_ansatz_z_expectations(
            x, params, n_qubits, d_layers=ansatz.d_layers,
            use_rz=ansatz.use_rz, entanglement=ansatz.entanglement,
        )
        np.testing.assert_allclose(z_numpy, z_qiskit, atol=1e-10)

    @pytest.mark.parametrize(
        "version",
        [VQCVersion.V1_LINEAR, VQCVersion.V4_RING, VQCVersion.V5_REUPLOAD],
    )
    def test_parameter_shift_gradient_matches_vqcansatz(self, version):
        n_qubits = 4
        ansatz = VQCAnsatz(n_qubits=n_qubits, version=version)
        rng = np.random.default_rng(42)
        x = rng.uniform(-np.pi, np.pi, n_qubits)
        params = ansatz.init_params(seed=42)

        grad_numpy = ansatz.parameter_shift_feature_grad(x, params)

        def expval_fn(p: np.ndarray) -> np.ndarray:
            return reupload_ansatz_z_expectations(
                x, p, n_qubits, d_layers=ansatz.d_layers,
                use_rz=ansatz.use_rz, entanglement=ansatz.entanglement,
            )

        grad_qiskit = parameter_shift_gradient(expval_fn, params)

        assert grad_qiskit.shape == grad_numpy.shape
        np.testing.assert_allclose(grad_numpy, grad_qiskit, atol=1e-9)

    def test_gradient_matches_finite_difference(self):
        # Independent sanity check not tied to either simulator's own math:
        # central finite differences on the Qiskit path should agree with
        # the exact parameter-shift result to O(h^2).
        n_qubits = 3
        d_layers, use_rz, entanglement = 2, True, "ring"
        rng = np.random.default_rng(1)
        x = rng.uniform(-np.pi, np.pi, n_qubits)
        params = rng.normal(0, 0.5, n_qubits * (2 if use_rz else 1) * d_layers)

        def expval_fn(p: np.ndarray) -> np.ndarray:
            return reupload_ansatz_z_expectations(
                x, p, n_qubits, d_layers=d_layers, use_rz=use_rz, entanglement=entanglement
            )

        grad_shift = parameter_shift_gradient(expval_fn, params)

        h = 1e-6
        grad_fd = np.zeros_like(grad_shift)
        for k in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[k] += h
            p_minus[k] -= h
            grad_fd[k] = (expval_fn(p_plus) - expval_fn(p_minus)) / (2 * h)

        np.testing.assert_allclose(grad_shift, grad_fd, atol=1e-5)

    def test_rejects_wrong_param_count(self):
        with pytest.raises(ValidationError):
            reupload_ansatz_circuit(
                np.zeros(3), np.zeros(2), n_qubits=3, d_layers=1, use_rz=True, entanglement="ring"
            )
