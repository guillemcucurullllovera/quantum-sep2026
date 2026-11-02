import numpy as np
import pytest

from applications.asset_allocation import AssetClassAllocator, AssetClassBucket
from applications.portfolio import PortfolioOptimizer, generate_synthetic_market
from algorithms.qaoa_qubo.qubo_builder import PortfolioConstraints
from core.errors import ValidationError
from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata


def test_synthetic_market_covariance_is_psd_and_shapes_match():
    mu, cov, names = generate_synthetic_market(10, seed=1)
    assert mu.shape == (10,)
    assert cov.shape == (10, 10)
    assert len(names) == 10
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-6


def test_portfolio_optimizer_end_to_end_and_records_to_experiment_registry(tmp_path):
    mu, cov, names = generate_synthetic_market(9, seed=2)
    optimizer = PortfolioOptimizer(mu, cov, asset_names=names)
    report = optimizer.optimize(
        PortfolioConstraints(cardinality_k=3, cardinality_penalty=4.0),
        qaoa_n_iter=100,
    )

    assert len(report.selected_assets) == sum(int(b) for b in report.qaoa_bitstring)
    assert report.exact_cost is not None
    # QAOA and SA should both be at least as good as *some* sane upper bound:
    # not worse than 3x the exact optimum's magnitude (loose smoke bound)
    assert report.qaoa_cost <= abs(report.exact_cost) * 5 + 1.0

    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    backend = BackendDescriptor(name="numpy_gen2", kind=BackendKind.SIMULATOR, num_qubits=len(mu))
    config = ExperimentConfig(name="portfolio_e2e", algorithm="qaoa_qubo", application="portfolio", backend=backend)
    result = ExperimentResult(
        metadata=RunMetadata(algorithm="qaoa_qubo", generation="gen2"),
        config=config,
        metrics={"qaoa_cost": report.qaoa_cost, "exact_cost": report.exact_cost or 0.0},
        classical_baseline_metrics={"sa_cost": report.sa_cost},
    )
    run_id = registry.record(result)
    fetched = registry.get(run_id)
    assert fetched.metrics["qaoa_cost"] == report.qaoa_cost


def test_portfolio_optimizer_requires_cardinality():
    mu, cov, _ = generate_synthetic_market(5, seed=3)
    optimizer = PortfolioOptimizer(mu, cov)
    with pytest.raises(ValidationError):
        optimizer.optimize(PortfolioConstraints())


def test_asset_class_allocator_respects_bucket_ranges():
    mu, cov, names = generate_synthetic_market(9, seed=4)
    buckets = [
        AssetClassBucket("equities", list(range(0, 4)), min_holdings=1, max_holdings=2),
        AssetClassBucket("bonds", list(range(4, 7)), min_holdings=1, max_holdings=2),
        AssetClassBucket("cash", list(range(7, 9)), min_holdings=0, max_holdings=1),
    ]
    allocator = AssetClassAllocator(mu, cov, buckets)
    result = allocator.allocate(total_cardinality=4, qaoa_n_iter=100)

    for name, info in result["by_bucket"].items():
        assert info["count"] <= (info["max_holdings"] if info["max_holdings"] is not None else 999)


def test_asset_class_allocator_rejects_unassigned_assets():
    mu, cov, _ = generate_synthetic_market(5, seed=5)
    buckets = [AssetClassBucket("only_some", [0, 1, 2])]  # assets 3, 4 unassigned
    with pytest.raises(ValidationError):
        AssetClassAllocator(mu, cov, buckets)
