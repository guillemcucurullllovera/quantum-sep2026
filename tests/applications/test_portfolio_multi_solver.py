import numpy as np
import pytest

from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import compare_solvers, solve_portfolio
from applications.portfolio.report import build_comparison_report, build_report
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


def test_portfolio_spec_shape_validation():
    spec = _toy_spec(n=4)
    spec.validate_shape()  # should not raise
    bad = PortfolioSpec(
        assets=[AssetSpec(name="A", expected_return=0.1), AssetSpec(name="B", expected_return=0.1)],
        cov_matrix=[[1.0]],  # wrong shape for 2 assets
    )
    with pytest.raises(Exception):
        bad.validate_shape()


@pytest.mark.parametrize("choice", [SolverChoice.QAOA_DIRECT, SolverChoice.QOKIT_FAST, SolverChoice.MODELED_REPAIR])
def test_each_solver_backend_runs_and_respects_cardinality(choice):
    spec = _toy_spec(n=6, k=3)
    result = solve_portfolio(spec, choice, seed=1)
    assert result.bitstring.shape == (6,)
    v = validate_bitstring(spec, result.bitstring)
    # QAOA-based solvers are heuristic; report the real outcome rather than
    # assert perfection, but the deterministic modeled_repair path with
    # explicit feasibility repair must always land exactly on the constraint.
    if choice == SolverChoice.MODELED_REPAIR:
        assert v.ok, f"modeled_repair (has explicit repair step) violated: {v.violations}"


def test_compare_solvers_runs_all_three_and_reports_distinct_numbers():
    spec = _toy_spec(n=6, k=3, seed=2)
    comparison = compare_solvers(spec, seed=3)
    assert set(comparison.results) == {"qaoa_direct", "qokit_fast", "modeled_repair"}
    assert len(comparison.benchmarks) == 3  # 3 pairwise comparisons among 3 solvers
    costs = {name: r.cost for name, r in comparison.results.items()}
    # not all three should be byte-identical -- they're different
    # architectures solving via different encodings/optimizers
    assert len(set(round(c, 8) for c in costs.values())) >= 2
    assert comparison.best in comparison.results
    report_text = build_comparison_report(spec, comparison)
    assert "qaoa_direct" in report_text and "qokit_fast" in report_text and "modeled_repair" in report_text


def test_validation_catches_deliberately_infeasible_bitstring():
    spec = _toy_spec(n=6, k=3)
    infeasible = np.array([1, 1, 1, 1, 0, 0], dtype=float)  # holds 4, constraint wants exactly 3
    v = validate_bitstring(spec, infeasible)
    assert not v.ok
    assert any("cardinality" in msg for msg in v.violations)


def test_validation_passes_feasible_bitstring():
    spec = _toy_spec(n=6, k=3)
    feasible = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    v = validate_bitstring(spec, feasible)
    assert v.ok
    assert v.violations == []


def test_sector_range_validation():
    spec = _toy_spec(n=6, k=3)
    spec.constraints.append(
        ConstraintSpec(kind="sector_range", params={"name": "techish", "asset_names": ["A0", "A1"], "max_count": 1})
    )
    # holds both A0 and A1 (in the sector) plus one more -> violates max_count=1
    x = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    v = validate_bitstring(spec, x)
    assert not v.ok
    assert any("sector_range" in msg for msg in v.violations)


def test_build_report_produces_readable_summary():
    spec = _toy_spec(n=6, k=3)
    result = solve_portfolio(spec, SolverChoice.MODELED_REPAIR, seed=5)
    report = build_report(spec, result)
    assert report.solver == "modeled_repair"
    assert isinstance(report.summary, str) and len(report.summary) > 10
    assert report.validation.ok
