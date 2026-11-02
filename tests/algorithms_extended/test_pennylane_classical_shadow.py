import numpy as np
from qiskit import QuantumCircuit

from algorithms_extended.pennylane_asymmetrica.finance.classical_shadow import (
    median_of_means,
    pauli_expval,
    sample_classical_shadow,
)


def test_median_of_means_matches_numpy_on_uniform_data():
    arr = np.ones(100) * 3.0
    assert abs(median_of_means(arr, num_batches=10) - 3.0) < 1e-9


def test_shadow_of_zero_state_estimates_z_expectation_near_one():
    qc = QuantumCircuit(2)  # |00>, no gates
    shadow = sample_classical_shadow(qc, n_snapshots=400, seed=0)
    word = np.array([2, -1])  # Z on qubit 0, identity on qubit 1
    est = shadow.expval(word, num_batches=20)
    assert abs(est - 1.0) < 0.35  # classical-shadow estimator has real variance; loose but real bound


def test_shadow_of_plus_state_estimates_x_expectation_near_one():
    qc = QuantumCircuit(1)
    qc.h(0)  # |+>
    shadow = sample_classical_shadow(qc, n_snapshots=400, seed=1)
    word = np.array([0])  # X on qubit 0
    est = shadow.expval(word, num_batches=20)
    assert abs(est - 1.0) < 0.35


def test_pauli_expval_zero_for_nonmatching_recipe():
    bits = np.array([[0, 1]])
    recipes = np.array([[0, 0]])  # both measured in X basis
    word = np.array([2, -1])  # asks for Z on qubit 0 -- recipe doesn't match
    vals = pauli_expval(bits, recipes, word)
    assert vals[0] == 0.0
