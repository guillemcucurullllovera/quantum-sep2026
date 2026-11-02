"""Coverage for the multi-generation solver access and walk-forward
evaluation added to `AssetClassAllocator` recently
(`compare_all_solvers`, `walk_forward_allocate`, `compare_walk_forward_allocate`).
"""
from __future__ import annotations

import numpy as np

from applications.asset_allocation.allocator import AssetClassAllocator, AssetClassBucket
from applications.portfolio.domain import SolverChoice
from applications.portfolio.scenario import generate_multi_period_market


def _toy_allocator(seed=0):
    rng = np.random.default_rng(seed)
    n = 6
    mu = rng.uniform(0.02, 0.15, size=n)
    L = rng.normal(scale=0.05, size=(n, n))
    cov = L @ L.T + np.eye(n) * 0.01
    buckets = [
        AssetClassBucket(name="equities", asset_indices=[0, 1, 2], min_holdings=1, max_holdings=2),
        AssetClassBucket(name="bonds", asset_indices=[3, 4, 5], min_holdings=0, max_holdings=2),
    ]
    return AssetClassAllocator(mu, cov, buckets)


def test_compare_all_solvers_runs_all_eight_generations():
    alloc = _toy_allocator(seed=1)
    result = alloc.compare_all_solvers(total_cardinality=3, seed=2)
    assert set(result["solvers"]) == {
        "qaoa_direct", "gen1_qaoa", "qokit_fast", "modeled_repair",
        "qp_asymmetrica", "xy_mixer_qaoa", "training_pipeline_tqa", "classical_baseline",
    }
    assert result["best"] in result["solvers"]


def test_compare_all_solvers_reports_per_bucket_compliance_for_every_generation():
    alloc = _toy_allocator(seed=2)
    result = alloc.compare_all_solvers(total_cardinality=3, seed=3)
    for name, info in result["solvers"].items():
        assert "by_bucket" in info
        assert set(info["by_bucket"]) == {"equities", "bonds"}
        for bucket_info in info["by_bucket"].values():
            assert "within_range" in bucket_info


def test_walk_forward_allocate_carries_bucket_constraints_across_periods():
    alloc = _toy_allocator(seed=3)
    realized, names, _ = generate_multi_period_market(6, n_periods=24, seed=4, n_sectors=2)
    result = alloc.walk_forward_allocate(
        realized, total_cardinality=3, solver=SolverChoice.CLASSICAL_BASELINE,
        n_splits=3, turnover_limit=3, seed=5,
    )
    assert len(result.periods) == 3
    for period in result.periods:
        sector_kinds = [c.kind for c in period.spec.constraints]
        assert sector_kinds.count("sector_range") == 2  # equities + bonds bucket ranges present every period


def test_compare_walk_forward_allocate_runs_multiple_solvers():
    alloc = _toy_allocator(seed=4)
    realized, names, _ = generate_multi_period_market(6, n_periods=24, seed=5, n_sectors=2)
    results = alloc.compare_walk_forward_allocate(
        realized, total_cardinality=3,
        solvers=[SolverChoice.CLASSICAL_BASELINE, SolverChoice.QOKIT_FAST],
        n_splits=3, seed=6,
    )
    assert set(results) == {"classical_baseline", "qokit_fast"}
    for r in results.values():
        assert len(r.periods) == 3
