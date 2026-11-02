import numpy as np
import pytest

from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder, SectorLimit
from algorithms.qaoa_qubo.solvers import NumpyQAOASolver, QiskitQAOASolver
from algorithms_extended.qiskit_optimization_finance import (
    AdaptivePenaltyStrategy,
    FinancePortfolioModel,
    FixedPenaltyStrategy,
    ModelToQUBOConverter,
    is_feasible,
    repair_infeasible,
    solve_model,
)


def _toy_model(long_short=False):
    rng = np.random.default_rng(0)
    n = 6
    mu = rng.normal(0.05, 0.02, n)
    A = rng.normal(0, 0.1, (n, n))
    sigma = A @ A.T + np.eye(n) * 0.05
    names = [f"A{i}" for i in range(n)]
    m = FinancePortfolioModel(names, mu, sigma, risk_aversion=0.5, long_short=long_short)
    return m, mu, sigma, names


def test_constraint_summary_reflects_added_constraints():
    m, mu, sigma, names = _toy_model()
    m.add_cardinality(3).add_sector_range("techish", names[:3], max_count=2)
    kinds = {c["kind"] for c in m.constraint_summary()}
    assert kinds == {"cardinality", "sector_range"}


def test_model_to_qubo_matches_direct_build():
    """The converter must produce EXACTLY the same QUBO as building the
    equivalent constraints directly with PortfolioQUBOBuilder -- proving the
    object-model packaging step doesn't silently lose or misroute anything."""
    m, mu, sigma, names = _toy_model()
    m.add_cardinality(3).add_sector_range("techish", names[:3], max_count=2)

    Q_via_model, _ = ModelToQUBOConverter().convert(m, FixedPenaltyStrategy())

    direct_builder = PortfolioQUBOBuilder(mu, sigma, risk_aversion=0.5)
    direct_constraints = PortfolioConstraints(
        cardinality_k=3, cardinality_penalty=2.0,
        sector_limits=[SectorLimit("techish", [0, 1, 2], max_count=2)], sector_penalty=1.5,
    )
    Q_direct = direct_builder.build(direct_constraints)

    np.testing.assert_allclose(Q_via_model, Q_direct, atol=1e-12)


def test_adaptive_penalty_enforces_constraint_where_fixed_penalty_fails():
    """A deliberately badly-scaled objective (mu, sigma multiplied way up)
    should make FixedPenaltyStrategy's constant penalty negligible relative
    to the objective, so the unconstrained-minimum-adjacent solution
    violates cardinality; AdaptivePenaltyStrategy should not."""
    rng = np.random.default_rng(1)
    n = 8
    mu = rng.normal(0, 1.0, n) * 1e4  # objective coefficients ~1e4
    A = rng.normal(0, 1.0, (n, n)) * 1e2
    sigma = A @ A.T
    names = [f"A{i}" for i in range(n)]
    m = FinancePortfolioModel(names, mu, sigma, risk_aversion=0.0)  # pure-return objective, easiest to violate
    m.add_cardinality(3)

    Q_fixed, c_fixed = ModelToQUBOConverter().convert(m, FixedPenaltyStrategy())
    Q_adaptive, c_adaptive = ModelToQUBOConverter().convert(m, AdaptivePenaltyStrategy(k=3.0))

    from algorithms.qaoa_qubo.classical_baseline import brute_force_qubo

    x_fixed_free, _ = brute_force_qubo(Q_fixed, max_n=8)
    x_adaptive_free, _ = brute_force_qubo(Q_adaptive, max_n=8)

    fixed_card_ok = is_feasible(x_fixed_free, c_fixed).get("cardinality")
    adaptive_card_ok = is_feasible(x_adaptive_free, c_adaptive).get("cardinality")

    assert fixed_card_ok is False, "expected fixed penalty to be overwhelmed by the inflated objective"
    assert adaptive_card_ok is True, "adaptive penalty should scale with the objective and stay enforced"


def test_feasibility_repair_fixes_cardinality_violation():
    m, mu, sigma, names = _toy_model()
    m.add_cardinality(3)
    Q, constraints = ModelToQUBOConverter().convert(m, FixedPenaltyStrategy())

    x_bad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # 5 selected, want 3
    assert is_feasible(x_bad, constraints)["cardinality"] is False

    x_fixed, report = repair_infeasible(Q, x_bad, constraints, seed=0)
    assert report["cardinality"] is True
    assert int(x_fixed.sum()) == 3


def test_end_to_end_qaoa_integration_numpy_solver():
    m, mu, sigma, names = _toy_model()
    m.add_cardinality(3)
    solver = NumpyQAOASolver(p=2, n_iter=80, seed=0)
    result = solve_model(m, FixedPenaltyStrategy(), solver, seed=0)
    assert result.feasibility.get("cardinality") is True
    assert isinstance(result.cost, float)


def test_end_to_end_qaoa_integration_qiskit_solver_small_n():
    """Small n to keep this fast on Aer statevector-shot execution."""
    rng = np.random.default_rng(2)
    n = 4
    mu = rng.normal(0.05, 0.02, n)
    A = rng.normal(0, 0.1, (n, n))
    sigma = A @ A.T + np.eye(n) * 0.05
    names = [f"A{i}" for i in range(n)]
    m = FinancePortfolioModel(names, mu, sigma, risk_aversion=0.5).add_cardinality(2)

    solver = QiskitQAOASolver(p=1, shots=512, maxiter=15, seed=0)
    result = solve_model(m, FixedPenaltyStrategy(), solver, seed=0)
    assert result.engine == "qiskit_aer"
    assert result.feasibility.get("cardinality") is True


def test_long_short_model_produces_valid_qubo_shape():
    m, mu, sigma, names = _toy_model(long_short=True)
    m.add_leverage_limit(3).add_net_exposure_target(1.0)
    Q, constraints = ModelToQUBOConverter().convert(m, FixedPenaltyStrategy())
    assert Q.shape == (2 * len(names), 2 * len(names))


def test_unknown_asset_name_raises():
    m, mu, sigma, names = _toy_model()
    with pytest.raises(Exception):
        m.add_sector_range("bad", ["not_a_real_asset"])
