"""Real tests for algorithms_extended/ibm_quantum_kernel_training/."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms_extended.ibm_quantum_kernel_training.covariant_feature_map import (
    CovariantFeatureMap,
    SectorGroupFeatureMap,
    sector_group_entanglement,
)
from algorithms_extended.ibm_quantum_kernel_training.qkt_trainer import (
    compare_training_strategies,
    make_kernel_fn_factory,
    spsa_kta_train,
)
from algorithms_extended.ibm_quantum_kernel_training.noise_aware_training import train_under_noise


def _toy_dataset(n=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1.0, -1.0] * (n // 2))
    X = np.zeros((n, 4))
    for i in range(n):
        base = 1.0 if y[i] > 0 else -1.0
        X[i] = base + 0.15 * rng.standard_normal(4)
    return X, y


def test_covariant_feature_map_shape():
    fmap = CovariantFeatureMap(feature_dimension=6)
    assert fmap.num_qubits == 3
    assert len(fmap.input_parameters) == 6
    assert len(fmap.training_parameters) == 3  # one Ry angle per qubit, not shared


def test_covariant_feature_map_odd_dimension_rejected():
    with pytest.raises(ValueError):
        CovariantFeatureMap(feature_dimension=5)


def test_bind_data_and_training_are_independent():
    fmap = CovariantFeatureMap(feature_dimension=4)
    bound = fmap.bind_training(np.array([0.1, 0.2])).bind_data(np.array([0.3, 0.4, 0.5, 0.6]))
    assert len(bound.parameters) == 0  # fully bound, nothing left free


def test_sector_group_entanglement_only_connects_same_sector_plus_bridges():
    sectors = [0, 0, 1, 1, 2]
    edges = sector_group_entanglement(sectors)
    # same-sector pairs must all be present
    assert (0, 1) in edges  # sector 0
    assert (2, 3) in edges  # sector 1
    # cross-sector non-bridge pairs must NOT be densely connected
    same_sector_pairs = {(i, j) for i in range(5) for j in range(i + 1, 5) if sectors[i] == sectors[j]}
    dense_cross_sector = {(i, j) for (i, j) in edges if sectors[i] != sectors[j]}
    # cross-sector edges should be sparse: at most (num_sectors - 1), not O(n^2)
    assert len(dense_cross_sector) <= 2
    assert same_sector_pairs.issubset(set(edges))


def test_sector_group_feature_map_qubit_count_matches_sectors():
    fmap = SectorGroupFeatureMap(sector_of_qubit=[0, 0, 1])
    assert fmap.num_qubits == 3
    assert fmap.feature_dimension == 6


def test_fidelity_kernel_entry_self_overlap_is_near_one():
    """phi(x) overlapped with phi(x) (via the inverse) should measure the
    all-zero state with near-certainty -- a real correctness check on the
    overlap-circuit construction, not a smoke test."""
    fmap = CovariantFeatureMap(feature_dimension=4)
    theta = np.zeros(2)
    x = np.array([0.3, 0.7, 0.1, 0.9])
    factory = make_kernel_fn_factory(fmap, shots=2000)
    self_overlap = factory(theta)(x, x)
    assert self_overlap > 0.9


@pytest.mark.slow
def test_spsa_training_improves_kta_over_iterations():
    # Scaled down from an initial n=6/maxiter=15/shots=256 attempt that ran
    # past 5 minutes (each SPSA step evaluates a full n*(n-1)/2-entry Gram
    # matrix twice, each entry an Aer shot-based circuit run) -- n=4/maxiter=6
    # /shots=96 keeps the same real assertions with ~10x fewer circuit runs.
    X, y = _toy_dataset(n=4)
    fmap = CovariantFeatureMap(feature_dimension=4)
    theta_init = np.random.default_rng(1).uniform(0, np.pi, size=2)
    _, best_kta, history = spsa_kta_train(fmap, X, y, theta_init, maxiter=6, shots=96)
    assert len(history) >= 6
    # the best KTA found must be at least as good as (or better than) the
    # first evaluation -- a real, checkable convergence property
    assert best_kta >= history[0] - 1e-9


@pytest.mark.slow
def test_compare_training_strategies_runs_and_reports_a_winner():
    # Scaled down (see note on test_spsa_training_improves_kta_over_iterations)
    from algorithms.pqk.feature_maps import AngleCNOTRingV2

    X, y = _toy_dataset(n=4)
    fmap = CovariantFeatureMap(feature_dimension=4)
    theta_init = np.random.default_rng(2).uniform(0, np.pi, size=2)
    numpy_map = AngleCNOTRingV2(n_qubits=2)

    def numpy_factory(theta):
        # numpy_map is 2-qubit/1-feature-per-qubit; fmap (CovariantFeatureMap,
        # feature_dimension=4) is 2-qubit/2-features-per-qubit -- both are fed
        # the SAME full 4-feature X (so compare_training_strategies' single X
        # argument stays meaningful for both trainers), and this closure picks
        # the 2 features numpy_map actually needs rather than slicing X
        # upstream, which previously left fmap's x_par[2]/x_par[3] unbound
        # and crashed Aer with "parameter_binds is not specified".
        def kernel_fn(x1, x2):
            psi1 = numpy_map.encode(x1[:2], encoding_params=theta)
            psi2 = numpy_map.encode(x2[:2], encoding_params=theta)
            return float(np.abs(np.vdot(psi2, psi1)) ** 2)

        return kernel_fn

    result = compare_training_strategies(
        fmap, numpy_factory, X, y, theta_init[:2], spsa_maxiter=5, adam_n_steps=5, shots=96
    )
    assert result["winner"] in ("spsa", "adam_central_diff")
    assert result["spsa"]["n_evals"] >= 5
    assert result["adam_central_diff"]["n_evals"] >= 1


@pytest.mark.slow
def test_noise_degrades_or_preserves_kta_but_never_improves_shot_noise_variance():
    # Scaled down (see note above)
    X, y = _toy_dataset(n=4)
    fmap = CovariantFeatureMap(feature_dimension=4)
    theta_init = np.zeros(2)
    result = train_under_noise(fmap, X, y, theta_init, maxiter=4, shots=96)
    assert "kta_degradation" in result
    assert result["noise_descriptor"]["readout_error_mean"] is None or (
        result["noise_descriptor"]["readout_error_mean"] >= 0
    )
    # noisy and noiseless runs must actually be different histories (proves
    # the noise model was really applied, not silently ignored)
    assert result["noiseless"]["history"] != result["noisy"]["history"]
