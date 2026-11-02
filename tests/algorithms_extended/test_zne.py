import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from algorithms_extended.error_mitigation.zne import fold_circuit_globally, zero_noise_extrapolate
from core.errors import ValidationError


def test_fold_circuit_preserves_unitary_up_to_global_phase():
    qc = QuantumCircuit(1)
    qc.rx(0.7, 0)
    folded = fold_circuit_globally(qc, scale_factor=3)  # U (U^dag U)^1 == U

    u_original = Operator(qc).data
    u_folded = Operator(folded).data
    # folding by (2k+1) with an exactly-invertible circuit reproduces the
    # same unitary (up to global phase) on a noiseless simulator — folding
    # only matters once real gate noise scales with circuit length
    ratio = u_folded @ np.linalg.inv(u_original)
    phase = ratio[0, 0]
    assert np.allclose(ratio, phase * np.eye(2), atol=1e-8)


def test_fold_rejects_even_scale_factor():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValidationError):
        fold_circuit_globally(qc, scale_factor=2)


def test_fold_depth_grows_with_scale_factor():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    depths = [fold_circuit_globally(qc, k).size() for k in (1, 3, 5, 7)]
    assert depths == sorted(depths)
    assert depths[0] < depths[-1]


def test_zne_linear_extrapolation_recovers_zero_noise_value():
    # synthetic linear noise model: measured = true_value - 0.02 * scale_factor
    true_value = 0.83
    scale_factors = [1, 3, 5]
    noisy_values = [true_value - 0.02 * s for s in scale_factors]

    extrapolated = zero_noise_extrapolate(scale_factors, noisy_values, method="linear")
    assert abs(extrapolated - true_value) < 1e-9

    # the *worst* noisy data point (highest scale factor) should be further
    # from the true value than the extrapolated estimate
    assert abs(noisy_values[-1] - true_value) > abs(extrapolated - true_value)


def test_zne_richardson_matches_polynomial_noise_exactly():
    true_value = 1.0
    scale_factors = [1.0, 3.0, 5.0]
    # quadratic noise model in scale factor
    noisy_values = [true_value + 0.01 * s + 0.002 * s**2 for s in scale_factors]
    extrapolated = zero_noise_extrapolate(scale_factors, noisy_values, method="richardson")
    assert abs(extrapolated - true_value) < 1e-6


def test_zne_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        zero_noise_extrapolate([1, 3], [0.5], method="linear")
