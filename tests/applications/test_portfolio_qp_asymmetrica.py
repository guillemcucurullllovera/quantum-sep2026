"""Real tests for the QP_ASYMMETRICA generation: builds an actual
`QuadraticProgram` (from the vendored, real qiskit-optimization full-base
package) with hard linear constraints, and solves it via each of the four
real meta-optimizer strategies that package provides.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec
from applications.portfolio.exceptions import SolverFailureError
from applications.portfolio.qp_backend import QPStrategy, build_quadratic_program, solve_qp_asymmetrica
from applications.portfolio.validation import validate_bitstring
from core.errors import ValidationError


def _toy_spec(n=5, k=2, seed=0, extra_constraints=None) -> PortfolioSpec:
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(n)]
    mu = rng.uniform(0.02, 0.15, size=n)
    L = rng.normal(scale=0.05, size=(n, n))
    cov = (L @ L.T + np.eye(n) * 0.01).tolist()
    constraints = [ConstraintSpec(kind="cardinality", params={"k": k})]
    constraints.extend(extra_constraints or [])
    return PortfolioSpec(
        assets=[AssetSpec(name=nm, expected_return=float(m)) for nm, m in zip(names, mu)],
        cov_matrix=cov, constraints=constraints,
    )


def test_build_quadratic_program_has_named_variables_and_hard_cardinality_constraint():
    spec = _toy_spec(n=4, k=2)
    qp = build_quadratic_program(spec)
    assert [v.name for v in qp.variables] == ["A0", "A1", "A2", "A3"]
    assert len(qp.linear_constraints) == 1
    c = qp.linear_constraints[0]
    assert c.name == "cardinality"
    assert c.rhs == 2


def test_exact_strategy_finds_the_true_optimum_matching_brute_force():
    spec = _toy_spec(n=6, k=3, seed=1)
    x, cost, dt, extra = solve_qp_asymmetrica(spec, strategy=QPStrategy.EXACT)
    assert extra["status"] == "OptimizationResultStatus.SUCCESS"
    v = validate_bitstring(spec, x)
    assert v.ok

    # brute-force ground truth over all C(6,3) = 20 cardinality-3 subsets,
    # using the SAME QP objective (q*cov - (1-q)*mu, no penalty terms)
    from itertools import combinations
    mu, cov, q = spec.expected_returns(), spec.cov(), spec.risk_aversion
    best = None
    for combo in combinations(range(6), 3):
        xb = np.zeros(6)
        xb[list(combo)] = 1.0
        c = q * xb @ cov @ xb - (1 - q) * mu @ xb
        if best is None or c < best:
            best = c
    assert abs(cost - best) < 1e-6


@pytest.mark.parametrize("strategy", [QPStrategy.RECURSIVE_QAOA, QPStrategy.WARM_START_QAOA, QPStrategy.ADMM])
def test_meta_optimizer_strategies_run_end_to_end(strategy):
    """Real (not mocked) runs of RecursiveMinimumEigenOptimizer (wrapping a
    real Aer-Sampler QAOA), WarmStartQAOAOptimizer (LP-relaxation-informed),
    and ADMMOptimizer -- each a genuinely different meta-algorithm from the
    vendored qiskit-optimization-asymmetrica package, not three names for
    the same code path."""
    spec = _toy_spec(n=4, k=2, seed=2)
    x, cost, dt, extra = solve_qp_asymmetrica(spec, strategy=strategy, p=1, maxiter=15)
    assert x.shape == (4,)
    assert np.isfinite(cost)
    assert extra["strategy"] == strategy.value


def test_turnover_is_encoded_as_a_real_linear_inequality_constraint():
    prev = [1.0, 0.0, 1.0, 0.0, 0.0]
    spec = _toy_spec(n=5, k=2, extra_constraints=[
        ConstraintSpec(kind="turnover", params={"previous_holdings": prev, "limit": 1}),
    ])
    qp = build_quadratic_program(spec)
    names = {c.name for c in qp.linear_constraints}
    assert "turnover" in names
    x, cost, dt, extra = solve_qp_asymmetrica(spec, strategy=QPStrategy.EXACT)
    v = validate_bitstring(spec, x)
    assert v.ok, f"hard turnover constraint violated: {v.violations}"


def test_sector_range_produces_two_bound_constraints():
    spec = _toy_spec(n=5, k=3, extra_constraints=[
        ConstraintSpec(kind="sector_range", params={"name": "techish", "asset_names": ["A0", "A1"], "min_count": 1, "max_count": 1}),
    ])
    qp = build_quadratic_program(spec)
    names = {c.name for c in qp.linear_constraints}
    assert "sector_min_techish" in names and "sector_max_techish" in names
    x, cost, dt, extra = solve_qp_asymmetrica(spec, strategy=QPStrategy.EXACT)
    v = validate_bitstring(spec, x)
    assert v.ok


def test_liquidity_floor_excludes_illiquid_assets_via_hard_constraint():
    names = [f"A{i}" for i in range(5)]
    assets = [AssetSpec(name=nm, expected_return=0.1, liquidity_score=1.0 if i != 0 else 0.0) for i, nm in enumerate(names)]
    spec = PortfolioSpec(
        assets=assets, cov_matrix=(np.eye(5) * 0.01).tolist(),
        constraints=[
            ConstraintSpec(kind="cardinality", params={"k": 2}),
            ConstraintSpec(kind="liquidity_floor", params={"floor": 0.5}),
        ],
    )
    qp = build_quadratic_program(spec)
    assert any(c.name == "liquidity_excl_A0" for c in qp.linear_constraints)
    x, cost, dt, extra = solve_qp_asymmetrica(spec, strategy=QPStrategy.EXACT)
    assert x[0] == 0.0  # the illiquid asset can never be selected
    v = validate_bitstring(spec, x)
    assert v.ok


def test_long_short_spec_is_rejected_with_a_clear_validation_error():
    spec = _toy_spec(n=4, k=2)
    spec = spec.model_copy(update={"long_short": True})
    with pytest.raises(ValidationError, match="long_short"):
        build_quadratic_program(spec)


def test_unknown_constraint_kind_raises_validation_error():
    spec = _toy_spec(n=4, k=2, extra_constraints=[ConstraintSpec(kind="not_a_real_kind", params={})])
    with pytest.raises(ValidationError, match="not_a_real_kind"):
        build_quadratic_program(spec)
