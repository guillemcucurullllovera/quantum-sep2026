"""Real integration tests for the qiskit-optimization full-base
adaptation: the QuadraticProgram-vs-direct-QUBO parity proof and the
warm-start strategy comparison.
"""
import numpy as np
import pytest

from algorithms_extended.qiskit_optimization_asymmetrica.integration import (
    compare_encodings,
    direct_qubo_matrix,
    modeled_qubo_matrix,
)
from algorithms_extended.qiskit_optimization_asymmetrica.warmstart_comparison import (
    compare_warmstart_strategies,
    lp_relax_cardinality_portfolio,
)


@pytest.fixture()
def small_returns():
    rng = np.random.default_rng(7)
    return rng.uniform(0.01, 0.15, size=6)


def test_direct_and_modeled_qubo_agree_on_optimal_selection(small_returns):
    result = compare_encodings(small_returns, cardinality_k=3)
    assert result["direct_cardinality_respected"]
    assert result["modeled_cardinality_respected"]
    assert result["selections_match"], (
        f"direct picked {result['direct_selected_assets']}, "
        f"modeled (real QuadraticProgramToQubo converter) picked "
        f"{result['modeled_selected_assets']} -- these should select the "
        f"same top-3-expected-return assets since both encode the identical "
        f"unconstrained objective plus an equivalent cardinality constraint"
    )


def test_modeled_qubo_selects_the_true_top_k_by_expected_return(small_returns):
    # Ground truth: with no risk term, "pick k to maximize expected return"
    # is trivially solved by taking the k highest-return assets.
    true_top_k = set(np.argsort(-small_returns)[:3].tolist())
    result = compare_encodings(small_returns, cardinality_k=3)
    assert set(result["modeled_selected_assets"]) == true_top_k
    assert set(result["direct_selected_assets"]) == true_top_k


def test_qubo_matrices_are_not_byte_identical_but_both_valid():
    # Sanity: the two independently-written encoders should NOT produce
    # identical matrices (different penalty schemes) -- if they did, that
    # would suggest one is just calling the other, not an independent proof.
    returns = np.array([0.05, 0.09, 0.02, 0.07])
    direct_Q = direct_qubo_matrix(returns, cardinality_k=2)
    modeled_Q, _ = modeled_qubo_matrix(returns, cardinality_k=2)
    assert direct_Q.shape == modeled_Q.shape
    assert not np.allclose(direct_Q, modeled_Q)


def test_lp_relaxation_respects_cardinality_and_bounds():
    returns = np.array([0.1, 0.2, 0.3, 0.05, 0.15])
    sol = lp_relax_cardinality_portfolio(returns, cardinality_k=2)
    assert np.all(sol >= -1e-9) and np.all(sol <= 1 + 1e-9)
    assert abs(sol.sum() - 2) < 1e-6


def test_warmstart_factory_angles_respect_epsilon_bounds_and_correlate_with_lp():
    returns = np.array([0.1, 0.2, 0.3, 0.05, 0.15, 0.25])
    result = compare_warmstart_strategies(returns, cardinality_k=3, epsilon=0.2)
    assert result["bounds_respected"]
    # A correctly-implemented regularizer should preserve the LP relaxation's
    # relative ranking closely (allowing for epsilon-clamping ties).
    assert result["rank_correlation_lp_vs_warmstart"] > 0.5
    assert len(result["gamma_schedule"]) == 3
    assert len(result["beta_schedule"]) == 3


def test_warmstart_factory_rejects_invalid_epsilon():
    from algorithms_extended.qiskit_optimization_asymmetrica.algorithms.warm_start_qaoa_optimizer import (
        WarmStartQAOAFactory,
    )
    from algorithms_extended.qiskit_optimization_asymmetrica import QiskitOptimizationError

    with pytest.raises(QiskitOptimizationError):
        WarmStartQAOAFactory(epsilon=0.6)
