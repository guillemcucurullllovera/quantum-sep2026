import numpy as np

from applications.asset_allocation.allocator import AssetClassAllocator, AssetClassBucket
from applications.portfolio.domain import SolverChoice


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


def test_to_portfolio_spec_round_trips_bucket_constraints():
    alloc = _toy_allocator()
    spec = alloc.to_portfolio_spec(total_cardinality=3)
    kinds = {c.kind for c in spec.constraints}
    assert "cardinality" in kinds
    assert "sector_range" in kinds
    assert spec.n_assets() == 6


def test_allocate_with_modeled_repair_respects_bucket_ranges():
    alloc = _toy_allocator(seed=1)
    result = alloc.allocate_with(total_cardinality=3, solver=SolverChoice.MODELED_REPAIR, seed=2)
    assert result["solver"] == "modeled_repair"
    for name, info in result["by_bucket"].items():
        assert info["within_range"], f"bucket {name} out of range: {info}"


def test_allocate_with_qokit_fast_runs():
    alloc = _toy_allocator(seed=2)
    result = alloc.allocate_with(total_cardinality=3, solver=SolverChoice.QOKIT_FAST, seed=3)
    assert result["solver"] == "qokit_fast"
    assert isinstance(result["cost"], float)


def test_original_allocate_path_still_works():
    # regression guard: the pre-existing hand-rolled path must be untouched
    alloc = _toy_allocator(seed=3)
    result = alloc.allocate(total_cardinality=3, qaoa_n_iter=50)
    assert "qaoa_bitstring" in result and "by_bucket" in result
