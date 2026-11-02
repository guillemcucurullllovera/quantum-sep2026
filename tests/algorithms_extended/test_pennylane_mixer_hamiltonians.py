import numpy as np

from algorithms_extended.pennylane_asymmetrica.finance.mixer_hamiltonians import (
    bit_flip_mixer,
    complete_graph_edges,
    ring_graph_edges,
    x_mixer,
    xy_mixer,
)


def test_x_mixer_is_sum_of_single_qubit_x():
    H = x_mixer(3)
    assert H.num_qubits == 3
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T)  # Hermitian


def test_xy_mixer_hermitian_and_correct_term_count():
    edges = ring_graph_edges(4)
    H = xy_mixer(4, edges)
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T)
    # ring of 4 qubits -> 4 edges -> up to 8 Pauli terms (XX+YY per edge) before simplification
    assert len(H.paulis) <= 8


def test_xy_mixer_preserves_hamming_weight_subspace():
    # xy_mixer should only connect basis states of EQUAL Hamming weight;
    # verify by checking the matrix has zero amplitude between states of
    # different weight (e.g. |0011> and |0111>: weights 2 and 3)
    edges = complete_graph_edges(4)
    H = xy_mixer(4, edges)
    mat = H.to_matrix()
    # qiskit little-endian: bit order reversed vs string index
    def weight(idx, n=4):
        return bin(idx).count("1")
    n_states = 2**4
    cross_weight_amplitude = 0.0
    same_weight_amplitude = 0.0
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            if weight(i) != weight(j):
                cross_weight_amplitude += abs(mat[i, j])
            else:
                same_weight_amplitude += abs(mat[i, j])
    assert cross_weight_amplitude < 1e-9
    assert same_weight_amplitude > 0.0  # sanity: it does connect same-weight states


def test_bit_flip_mixer_hermitian():
    edges = ring_graph_edges(3)
    H = bit_flip_mixer(3, edges, b=0)
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T, atol=1e-9)


def test_complete_vs_ring_edge_counts():
    assert len(complete_graph_edges(5)) == 10  # C(5,2)
    assert len(ring_graph_edges(5)) == 5
