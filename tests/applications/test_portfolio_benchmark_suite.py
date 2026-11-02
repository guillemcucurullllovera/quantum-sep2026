"""Real tests for cross-generation benchmark integration: every run must
actually be logged through core.experiment_registry, and a genuinely
inapplicable solver must be reported in `failures`, not silently dropped or
allowed to abort the whole benchmark.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.benchmark_suite import benchmark_pair, run_full_generation_benchmark
from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.exceptions import SolverFailureError
from applications.portfolio.multi_solver import ALL_SOLVER_CHOICES
from core.errors import ValidationError
from core.experiment_registry import ExperimentRegistry


def _toy_spec(n=5, k=2, seed=0) -> PortfolioSpec:
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(n)]
    mu = rng.uniform(0.02, 0.15, size=n)
    L = rng.normal(scale=0.05, size=(n, n))
    cov = (L @ L.T + np.eye(n) * 0.01).tolist()
    return PortfolioSpec(
        assets=[AssetSpec(name=nm, expected_return=float(m)) for nm, m in zip(names, mu)],
        cov_matrix=cov, constraints=[ConstraintSpec(kind="cardinality", params={"k": k})],
    )


def test_run_full_generation_benchmark_logs_every_run_to_the_registry(tmp_path):
    spec = _toy_spec(n=5, k=2)
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    results, failures = run_full_generation_benchmark(spec, seed=1, registry=registry)

    assert failures == {}
    assert len(results) == len(ALL_SOLVER_CHOICES) == 8
    assert registry.count() == 8
    for r in results:
        fetched = registry.get(r.metadata.run_id)
        assert fetched.metrics["cost"] == r.metrics["cost"]
        assert fetched.config.application == "portfolio"


def test_run_full_generation_benchmark_reports_inapplicable_solvers_without_aborting(tmp_path):
    """A spec with NO cardinality constraint is inapplicable to every
    cardinality-requiring solver -- each should land in `failures` with its
    real error message, and the benchmark must still complete rather than
    raising on the first inapplicable one."""
    spec = PortfolioSpec(
        assets=[AssetSpec(name=f"A{i}", expected_return=0.05) for i in range(4)],
        cov_matrix=(np.eye(4) * 0.01).tolist(), constraints=[],
    )
    registry = ExperimentRegistry(db_path=tmp_path / "reg2.sqlite3")
    results, failures = run_full_generation_benchmark(spec, seed=1, registry=registry)
    assert len(failures) > 0
    assert all(isinstance(msg, str) and msg for msg in failures.values())
    assert len(results) + len(failures) == len(ALL_SOLVER_CHOICES)


def test_benchmark_pair_uses_shared_runner_and_reports_real_cost_values():
    spec = _toy_spec(n=5, k=2, seed=2)
    result = benchmark_pair(spec, SolverChoice.QAOA_DIRECT, SolverChoice.CLASSICAL_BASELINE, seed=3)
    assert result.method_a == "qaoa_direct"
    assert result.method_b == "classical_baseline"
    assert result.metric_name == "portfolio_qubo_cost"
    assert np.isfinite(result.value_a) and np.isfinite(result.value_b)
    assert result.wall_time_a_s is not None and result.wall_time_b_s is not None
