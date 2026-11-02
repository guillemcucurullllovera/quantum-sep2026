import numpy as np
from qiskit.quantum_info import Statevector

from algorithms_extended.pennylane_asymmetrica.finance.mixer_hamiltonians import ring_graph_edges
from algorithms_extended.pennylane_asymmetrica.finance.xy_mixer_circuit import (
    build_cardinality_preserving_qaoa_circuit,
    dicke_state_circuit,
)


def test_dicke_state_has_correct_hamming_weight_support_only():
    n, k = 4, 2
    qc = dicke_state_circuit(n, k)
    sv = Statevector(qc)
    probs = sv.probabilities()
    for idx, p in enumerate(probs):
        weight = bin(idx).count("1")
        if p > 1e-9:
            assert weight == k, f"state {idx:04b} has weight {weight}, expected {k}"


def test_dicke_state_is_uniform_over_support():
    n, k = 5, 2
    qc = dicke_state_circuit(n, k)
    sv = Statevector(qc)
    probs = sv.probabilities()
    nonzero = probs[probs > 1e-9]
    assert len(nonzero) == 10  # C(5,2)
    assert np.allclose(nonzero, nonzero[0], atol=1e-9)


def test_full_qaoa_circuit_preserves_cardinality_through_all_layers():
    n, k = 4, 2
    edges = ring_graph_edges(n)
    rng = np.random.default_rng(0)
    cost_diag = rng.normal(size=2**n)
    gamma = np.array([0.3, 0.5])
    beta = np.array([0.2, 0.4])

    qc = build_cardinality_preserving_qaoa_circuit(n, k, edges, cost_diag, gamma, beta)
    sv = Statevector(qc)
    probs = sv.probabilities()

    total_wrong_weight_prob = sum(p for idx, p in enumerate(probs) if bin(idx).count("1") != k)
    assert total_wrong_weight_prob < 1e-6


def test_different_gamma_beta_produce_different_distributions():
    n, k = 4, 2
    edges = ring_graph_edges(n)
    cost_diag = np.arange(2**n, dtype=float)

    qc1 = build_cardinality_preserving_qaoa_circuit(n, k, edges, cost_diag, np.array([0.1]), np.array([0.1]))
    qc2 = build_cardinality_preserving_qaoa_circuit(n, k, edges, cost_diag, np.array([0.9]), np.array([1.3]))
    p1 = Statevector(qc1).probabilities()
    p2 = Statevector(qc2).probabilities()
    assert not np.allclose(p1, p2, atol=1e-6)
