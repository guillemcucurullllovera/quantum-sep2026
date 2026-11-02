"""Coverage for the five solver generations added recently
(gen1_qaoa, qp_asymmetrica's four meta-optimizer strategies,
xy_mixer_qaoa, training_pipeline_tqa, classical_baseline) on top of the
pre-existing three-way comparison already covered by
test_portfolio_multi_solver.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.qaoa_qubo.classical_baseline import brute_force_k_hot
from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints, PortfolioQUBOBuilder
from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import ALL_SOLVER_CHOICES, compare_solvers, solve_portfolio
from applications.portfolio.qp_backend import QPStrategy, solve_qp_asymmetrica
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


def test_all_solver_choices_covers_eight_generations():
    names = {c.value for c in ALL_SOLVER_CHOICES}
    assert names == {
        "qaoa_direct", "gen1_qaoa", "qokit_fast", "modeled_repair",
        "qp_asymmetrica", "xy_mixer_qaoa", "training_pipeline_tqa", "classical_baseline",
    }


@pytest.mark.parametrize(
    "choice",
    [SolverChoice.GEN1_QAOA, SolverChoice.XY_MIXER_QAOA, SolverChoice.QP_ASYMMETRICA,
     SolverChoice.TRAINING_PIPELINE_TQA, SolverChoice.CLASSICAL_BASELINE],
)
def test_each_new_generation_runs_and_returns_well_shaped_result(choice):
    spec = _toy_spec(n=6, k=3)
    result = solve_portfolio(spec, choice, seed=1)
    assert result.bitstring.shape == (6,)
    assert np.isfinite(result.cost)
    assert result.wall_time_s >= 0
    assert result.solver == choice.value


def test_xy_mixer_qaoa_enforces_cardinality_by_construction():
    """The Dicke-state initial state + XY-mixer only ever visit equal-
    Hamming-weight basis states, so cardinality (not necessarily every
    other constraint) must hold exactly for every run, unlike the
    penalty-based generations."""
    spec = _toy_spec(n=6, k=3, seed=4)
    result = solve_portfolio(spec, SolverChoice.XY_MIXER_QAOA, seed=2)
    assert int(result.bitstring.sum()) == 3


def test_qp_asymmetrica_exact_strategy_matches_brute_force_optimum():
    spec = _toy_spec(n=6, k=3, seed=5)
    result = solve_qp_asymmetrica(spec, seed=0, strategy=QPStrategy.EXACT)
    x, native_cost, _dt, _extra = result

    constraints = PortfolioConstraints(cardinality_k=3)
    Q = PortfolioQUBOBuilder(spec.expected_returns(), spec.cov(), risk_aversion=spec.risk_aversion).build(constraints)
    _bf_x, bf_cost = brute_force_k_hot(Q, 3)
    qp_cost_on_shared_q = float(x @ Q @ x)
    # EXACT uses NumPyMinimumEigensolver -- true ground state of its own QP
    # encoding -- so on the shared reference QUBO it must be at least as
    # good as brute force allows for numerical slack, and cardinality must
    # hold exactly since that's a hard linear_constraint in the QP.
    assert int(x.sum()) == 3
    assert qp_cost_on_shared_q <= bf_cost + 1e-6


@pytest.mark.parametrize("strategy", [QPStrategy.WARM_START_QAOA, QPStrategy.RECURSIVE_QAOA, QPStrategy.ADMM])
def test_qp_asymmetrica_meta_optimizer_strategies_run(strategy):
    spec = _toy_spec(n=5, k=2, seed=6)
    x, native_cost, dt, extra = solve_qp_asymmetrica(spec, seed=1, strategy=strategy, p=1, maxiter=20)
    assert x.shape == (5,)
    assert np.isfinite(native_cost)
    assert extra["strategy"] == strategy.value


def test_qp_asymmetrica_two_sided_sector_range_is_a_real_hard_constraint():
    """Unlike the raw-QUBO path (whose SectorLimit only enforces ONE side
    of a [min, max] range via an equality-pull penalty -- see
    qubo_builder.py's own documented limitation), QP_ASYMMETRICA models
    both bounds as independent linear_constraint()s, so a genuinely
    two-sided range should be respected exactly under the EXACT strategy."""
    spec = _toy_spec(n=6, k=4, seed=7)
    spec.constraints.append(
        ConstraintSpec(kind="sector_range", params={"name": "grp", "asset_names": ["A0", "A1", "A2"], "min_count": 1, "max_count": 2})
    )
    x, _cost, _dt, _extra = solve_qp_asymmetrica(spec, seed=2, strategy=QPStrategy.EXACT)
    held_in_group = int(x[:3].sum())
    assert 1 <= held_in_group <= 2


def test_classical_baseline_uses_markowitz_projection_and_reports_weights():
    spec = _toy_spec(n=6, k=3, seed=8)
    result = solve_portfolio(spec, SolverChoice.CLASSICAL_BASELINE, seed=0)
    assert "continuous_weights" in result.extra
    assert len(result.extra["continuous_weights"]) == 6
    assert int(result.bitstring.sum()) == 3


def test_training_pipeline_tqa_reports_real_tqa_diagnostics():
    spec = _toy_spec(n=6, k=3, seed=9)
    result = solve_portfolio(spec, SolverChoice.TRAINING_PIPELINE_TQA, seed=3)
    assert "tqa_dt" in result.extra and "tqa_energy" in result.extra
    assert result.extra["shots"] > 0


def test_compare_solvers_all_eight_reports_genuinely_different_numbers_and_feasibility():
    spec = _toy_spec(n=6, k=3, seed=10)
    comparison = compare_solvers(spec, seed=11, choices=list(ALL_SOLVER_CHOICES))
    assert set(comparison.results) == {c.value for c in ALL_SOLVER_CHOICES}
    costs = {name: r.cost for name, r in comparison.results.items()}
    assert len(set(round(c, 6) for c in costs.values())) >= 3, "eight independent architectures should not all agree exactly"
    assert comparison.best in comparison.results
    # every result's feasibility must come from an independent post-hoc check,
    # not the solver's own claim -- verify it actually matches re-validation
    for name, r in comparison.results.items():
        if len(r.bitstring) == spec.n_assets():
            v = validate_bitstring(spec, r.bitstring)
            assert v.ok == r.feasible


def test_gen1_qaoa_requires_cardinality_constraint():
    from applications.portfolio.gen1_backend import solve_gen1_qaoa
    from core.errors import ValidationError

    spec = _toy_spec(n=5, k=2, seed=12)
    spec.constraints = []  # strip cardinality
    with pytest.raises(ValidationError):
        solve_gen1_qaoa(spec, seed=0)
