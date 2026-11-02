"""End-to-end example: build a small portfolio problem and solve it across
every solver generation this application layer exposes (Gen1/Gen2 QAOA,
QOKit-fast, qiskit-optimization-finance modeled+repair,
qiskit-optimization-asymmetrica's real QuadraticProgram, a cardinality-
preserving XY-mixer circuit on Aer, IBM's QAOA-training-pipeline TQA
methodology, and a classical Markowitz baseline), then runs a walk-forward
backtest comparing a subset of them out-of-sample.

Run: python -m examples.qaoa_portfolio_example
"""
from __future__ import annotations

from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import ALL_SOLVER_CHOICES, compare_solvers
from applications.portfolio.report import build_comparison_report, build_walk_forward_report
from applications.portfolio.scenario import compare_walk_forward, generate_multi_period_market
from applications.portfolio.synthetic_market import generate_synthetic_market


def single_shot_comparison() -> None:
    n_assets = 6
    mu, cov, names = generate_synthetic_market(n_assets, seed=1)
    spec = PortfolioSpec(
        assets=[AssetSpec(name=n, expected_return=float(r)) for n, r in zip(names, mu)],
        cov_matrix=cov.tolist(),
        risk_aversion=0.5,
        constraints=[ConstraintSpec(kind="cardinality", params={"k": 3})],
    )

    print(f"Portfolio: {n_assets} assets, cardinality=3")
    print(f"Solver generations available through one interface: {len(ALL_SOLVER_CHOICES)}")
    comparison = compare_solvers(spec, seed=42, choices=list(ALL_SOLVER_CHOICES))
    print(build_comparison_report(spec, comparison))


def walk_forward_comparison() -> None:
    n_assets = 6
    realized, names, _sectors = generate_multi_period_market(n_assets, n_periods=32, seed=3, n_sectors=2)
    results = compare_walk_forward(
        realized, names, cardinality_k=3,
        solvers=[SolverChoice.QAOA_DIRECT, SolverChoice.QOKIT_FAST, SolverChoice.MODELED_REPAIR, SolverChoice.CLASSICAL_BASELINE],
        n_splits=4, turnover_limit=2, seed=7,
    )
    print()
    print(build_walk_forward_report(results))


def main() -> None:
    single_shot_comparison()
    walk_forward_comparison()


if __name__ == "__main__":
    main()
