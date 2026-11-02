import numpy as np

from hardware.mitigation import TREXMitigator
from hardware.mitigation.trex import apply_bitflip_twirl, correct_bitflip_twirl


def test_twirl_and_correct_roundtrip_recovers_original_state():
    state = np.array([0.6, 0.8], dtype=complex)  # normalized single-qubit state
    flip = np.array([1])
    twirled = apply_bitflip_twirl(state, flip)
    # applying X twice returns to the original state
    restored = apply_bitflip_twirl(twirled, flip)
    assert np.allclose(restored, state)


def test_correct_bitflip_twirl_inverts_counts_correctly():
    counts = np.array([100, 0, 0, 0])  # all outcomes at basis state |00>
    flip_string = np.array([1, 0])  # flip qubit 0
    corrected = correct_bitflip_twirl(counts, flip_string)
    # after un-flipping qubit 0, mass should move from index 0 to index 1
    assert corrected[1] == 100
    assert corrected[0] == 0


def test_trex_calibration_matrix_is_valid_stochastic_matrix():
    mit = TREXMitigator(n_qubits=2, readout_error_0=0.01, readout_error_1=0.02)
    matrix = mit._calibration_matrix
    # columns must each sum to 1 (it's a column-stochastic transition matrix)
    col_sums = matrix.sum(axis=0)
    assert np.allclose(col_sums, 1.0)


def test_perfect_readout_correction_is_identity():
    mit = TREXMitigator(n_qubits=1, readout_error_0=0.0, readout_error_1=0.0)
    counts = np.array([700, 300], dtype=float)
    corrected = mit.correct(counts)
    assert np.allclose(corrected, counts / counts.sum())
