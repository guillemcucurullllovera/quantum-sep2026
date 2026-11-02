import numpy as np
import pytest

from algorithms_extended.pennylane_asymmetrica.finance.kernel_alignment import (
    closest_psd_matrix,
    displace_matrix,
    flip_matrix,
    mitigate_depolarizing_noise,
    polarity,
    target_alignment,
    threshold_matrix,
)


def _linear_kernel(x, y):
    return float(np.dot(x, y))


def test_polarity_matches_hand_computed_value():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    Y = np.array([1, 1, -1, -1])
    p = polarity(X, Y, _linear_kernel, rescale_class_labels=False)
    K = np.array([[float(np.dot(a, b)) for b in X] for a in X])
    T = np.outer(Y, Y)
    expected = float(np.sum(K * T))
    assert abs(p - expected) < 1e-9


def test_target_alignment_is_normalized_polarity():
    X = np.array([[1.0, 0.2], [0.3, 1.0], [0.9, 0.9], [-0.8, -0.7]])
    Y = np.array([1, -1, 1, -1])
    ta = target_alignment(X, Y, _linear_kernel)
    normalized_polarity = polarity(X, Y, _linear_kernel, normalize=True)
    assert abs(ta - normalized_polarity) < 1e-12
    assert -1.0 - 1e-6 <= ta <= 1.0 + 1e-6


@pytest.mark.parametrize("repair", [threshold_matrix, displace_matrix, flip_matrix, closest_psd_matrix])
def test_all_repair_strategies_produce_psd_output(repair):
    broken = np.array([[1.0, 0.99, 0.98], [0.99, 1.0, 0.99], [0.98, 0.99, 1.0]])
    broken -= 0.05 * np.eye(3)  # push toward indefinite
    fixed = repair(broken)
    eigvals = np.linalg.eigvalsh((fixed + fixed.T) / 2)
    assert eigvals.min() >= -1e-8


def test_repair_strategies_actually_differ():
    broken = np.array([[1.0, 0.99, 0.5], [0.99, 1.0, 0.5], [0.5, 0.5, 1.0]]) - 0.08 * np.eye(3)
    results = [f(broken) for f in (threshold_matrix, displace_matrix, flip_matrix)]
    # not all three strategies should produce identical output on a case with a real negative eigenvalue
    assert not (np.allclose(results[0], results[1]) and np.allclose(results[1], results[2]))


def test_mitigate_depolarizing_noise_recovers_clean_kernel_approximately():
    # Precondition for this technique: the DIAGONAL must also carry the noise
    # signature (measured on noisy hardware, not fixed at the ideal value 1) --
    # that's what makes lambda estimable at all.
    rng = np.random.default_rng(0)
    n = 6
    off = rng.uniform(0.3, 0.9, size=(n, n))
    clean = (off + off.T) / 2
    np.fill_diagonal(clean, 1.0)

    num_wires = 3
    dim = 2**num_wires
    lam_true = 0.7
    noisy = lam_true * clean + (1 - lam_true) / dim  # diagonal included -- not reset to 1

    recovered = mitigate_depolarizing_noise(noisy, num_wires=num_wires, method="average")
    off_diag_mask = ~np.eye(n, dtype=bool)
    err_before = np.abs(noisy[off_diag_mask] - clean[off_diag_mask]).mean()
    err_after = np.abs(recovered[off_diag_mask] - clean[off_diag_mask]).mean()
    assert err_after < err_before
    assert err_after < 1e-9  # this synthetic case has a single exact global lambda -- recovery should be near-exact


def test_mitigate_depolarizing_noise_is_noop_when_diagonal_already_ideal():
    # diagonal == 1 implies the estimated noise_rate is exactly 0 -- a
    # perfectly-measured diagonal correctly implies "nothing to mitigate",
    # not an error.
    K = np.array([[1.0, 0.5], [0.5, 1.0]])
    out = mitigate_depolarizing_noise(K, num_wires=1, method="average")
    assert np.allclose(out, K)


def test_mitigate_depolarizing_noise_rejects_too_small_diagonal():
    dim = 2  # num_wires=1
    K = np.array([[0.1, 0.05], [0.05, 0.1]])  # diagonal below 1/dim=0.5
    with pytest.raises(ValueError):
        mitigate_depolarizing_noise(K, num_wires=1, method="average")
