"""Coverage for the applications-layer extended constraint vocabulary
(leverage, net_exposure, concentration_limit, risk_budget, tracking_error,
factor_exposure) and long/short bitstring validation added recently.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import solve_portfolio
from applications.portfolio.validation import validate_bitstring


def _toy_spec(n=6, k=3, seed=0) -> PortfolioSpec:
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(n)]
    mu = rng.uniform(0.02, 0.15, size=n)
    L = rng.normal(scale=0.05, size=(n, n))
    cov = (L @ L.T + np.eye(n) * 0.01).tolist()
    return PortfolioSpec(
        assets=[AssetSpec(name=nm, expected_return=float(m)) for nm, m in zip(names, mu)],
        cov_matrix=cov,
        constraints=[ConstraintSpec(kind="cardinality", params={"k": k})],
    )


def test_leverage_validation_flags_excess_gross_positions():
    spec = _toy_spec(n=6, k=3)
    spec.constraints.append(ConstraintSpec(kind="leverage", params={"max_gross_positions": 2}))
    x = np.array([1, 1, 1, 0, 0, 0], dtype=float)  # 3 holdings > limit 2
    v = validate_bitstring(spec, x)
    assert not v.ok
    assert any("leverage" in msg for msg in v.violations)


def test_concentration_limit_validation():
    spec = _toy_spec(n=6, k=3)
    # holding 3 assets -> equal weight 1/3 each; max_weight below that must fail
    spec.constraints.append(ConstraintSpec(kind="concentration_limit", params={"max_weight": 0.2}))
    x = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    v = validate_bitstring(spec, x)
    assert not v.ok
    assert any("concentration_limit" in msg for msg in v.violations)

    spec2 = _toy_spec(n=6, k=3)
    spec2.constraints.append(ConstraintSpec(kind="concentration_limit", params={"max_weight": 0.5}))
    v2 = validate_bitstring(spec2, x)
    assert v2.ok


def test_risk_budget_validation_against_realized_variance():
    spec = _toy_spec(n=6, k=3, seed=1)
    x = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    w = x / 3
    realized_var = float(w @ spec.cov() @ w)

    spec_tight = _toy_spec(n=6, k=3, seed=1)
    spec_tight.constraints.append(ConstraintSpec(kind="risk_budget", params={"max_variance": realized_var / 2}))
    assert not validate_bitstring(spec_tight, x).ok

    spec_loose = _toy_spec(n=6, k=3, seed=1)
    spec_loose.constraints.append(ConstraintSpec(kind="risk_budget", params={"max_variance": realized_var * 2}))
    assert validate_bitstring(spec_loose, x).ok


def test_tracking_error_validation():
    spec = _toy_spec(n=6, k=3, seed=2)
    x = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    # benchmark identical to the held equal-weight portfolio -> zero tracking error
    bench = (x / 3).tolist()
    spec.constraints.append(ConstraintSpec(kind="tracking_error", params={"benchmark_weights": bench, "max_te": 1e-9}))
    assert validate_bitstring(spec, x).ok

    spec2 = _toy_spec(n=6, k=3, seed=2)
    far_bench = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]
    spec2.constraints.append(ConstraintSpec(kind="tracking_error", params={"benchmark_weights": far_bench, "max_te": 1e-6}))
    assert not validate_bitstring(spec2, x).ok


def test_factor_exposure_validation_bounds():
    spec = _toy_spec(n=6, k=3, seed=3)
    loadings = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    x = np.array([1, 1, 1, 0, 0, 0], dtype=float)  # normalized exposure = 3/3 = 1.0
    spec.constraints.append(ConstraintSpec(kind="factor_exposure", params={"loadings": loadings, "max_exposure": 0.5}))
    assert not validate_bitstring(spec, x).ok

    spec2 = _toy_spec(n=6, k=3, seed=3)
    spec2.constraints.append(ConstraintSpec(kind="factor_exposure", params={"loadings": loadings, "min_exposure": 0.5}))
    assert validate_bitstring(spec2, x).ok


def test_long_short_bitstring_validation_flags_dual_flagged_asset_and_leverage():
    spec = PortfolioSpec(
        assets=[AssetSpec(name=f"A{i}", expected_return=0.05) for i in range(3)],
        cov_matrix=(np.eye(3) * 0.02).tolist(),
        long_short=True,
        constraints=[ConstraintSpec(kind="leverage", params={"max_gross_positions": 2})],
    )
    # long_0, long_1, short_0 (index 0 flagged both long and short) -> 3 gross positions > 2
    x = np.array([1, 1, 0, 1, 0, 0], dtype=float)
    v = validate_bitstring(spec, x)
    assert not v.ok
    assert any("both long and short" in msg for msg in v.violations)
    assert any("leverage" in msg for msg in v.violations)


def test_modeled_repair_supports_long_short_leverage_and_net_exposure():
    spec = PortfolioSpec(
        assets=[AssetSpec(name=f"A{i}", expected_return=float(r)) for i, r in enumerate([0.05, -0.03, 0.02, -0.01])],
        cov_matrix=(np.eye(4) * 0.02).tolist(),
        long_short=True,
        constraints=[
            ConstraintSpec(kind="leverage", params={"max_gross_positions": 2}),
            ConstraintSpec(kind="net_exposure", params={"target": 0.0}),
        ],
    )
    result = solve_portfolio(spec, SolverChoice.MODELED_REPAIR, seed=5)
    assert result.bitstring.shape == (8,)  # 2n encoding


def test_with_turnover_helper_replaces_prior_turnover_constraint():
    spec = _toy_spec(n=4, k=2, seed=9)
    prev = np.array([1, 0, 1, 0], dtype=float)
    updated = spec.with_turnover(prev, limit=1)
    turnover_constraints = updated.constraints_of("turnover")
    assert len(turnover_constraints) == 1
    assert turnover_constraints[0].params["limit"] == 1
    assert list(turnover_constraints[0].params["previous_holdings"]) == list(prev)
    # original spec is untouched (returns a copy)
    assert spec.constraints_of("turnover") == []
