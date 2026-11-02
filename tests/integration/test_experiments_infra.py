import tempfile
from pathlib import Path

import pytest

from core.errors import ValidationError
from core.experiment_registry import ExperimentRegistry
from core.registry import ALGORITHM_REGISTRY
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from experiments.analysis import compare_quantum_vs_classical, summarize_algorithm, summarize_metric
from experiments.parameter_search import grid, random_sample, run_sweep
from experiments.reporting import render_markdown_summary
from experiments.result_processing import aggregate_by_algorithm, to_rows


@pytest.fixture()
def registry():
    with tempfile.TemporaryDirectory() as tmp:
        yield ExperimentRegistry(db_path=Path(tmp) / "reg.sqlite3")


class _FakeAlgo:
    """Deterministic fake algorithm registered just for this test module."""

    def run(self, config: ExperimentConfig, *, seed: int) -> ExperimentResult:
        x = config.params.get("x", 0)
        return ExperimentResult(
            metadata=RunMetadata(algorithm=config.algorithm, generation="test"),
            config=config,
            metrics={"objective": float(x) ** 2},
            classical_baseline_metrics={"objective": float(x)},
        )


@pytest.fixture(autouse=True)
def _register_fake_algo():
    name = "_test_experiments_infra_algo"
    if name not in ALGORITHM_REGISTRY.list():
        ALGORITHM_REGISTRY.register(name)(_FakeAlgo())
    yield name


def _base_config(name: str) -> ExperimentConfig:
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4)
    return ExperimentConfig(name="t", algorithm=name, backend=backend)


def test_grid_is_full_cartesian_product():
    space = {"x": [1, 2], "y": ["a", "b"]}
    combos = grid(space)
    assert len(combos) == 4
    assert {"x": 1, "y": "a"} in combos
    assert {"x": 2, "y": "b"} in combos


def test_random_sample_size_and_determinism():
    space = {"x": [1, 2, 3, 4, 5]}
    a = random_sample(space, n=10, seed=1)
    b = random_sample(space, n=10, seed=1)
    assert len(a) == 10
    assert a == b  # same seed -> same sample


def test_run_sweep_logs_every_run_and_collects_errors(registry, _register_fake_algo):
    cfg = _base_config(_register_fake_algo)
    param_sets = [{"x": 1}, {"x": 2}, {"x": 3}]
    results = run_sweep(cfg, param_sets, registry=registry)
    assert len(results) == 3
    assert all(isinstance(r, ExperimentResult) for r in results)
    assert registry.count() == 3
    objectives = sorted(r.metrics["objective"] for r in results)
    assert objectives == [1.0, 4.0, 9.0]


def test_run_sweep_collects_unregistered_algorithm_error_instead_of_aborting(registry):
    cfg = ExperimentConfig(
        name="t",
        algorithm="_definitely_not_registered_xyz",
        backend=BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4),
    )
    results = run_sweep(cfg, [{"x": 1}, {"x": 2}], registry=registry, on_error="collect")
    assert len(results) == 2
    assert all(isinstance(r, ValidationError) for r in results)


def test_analysis_and_reporting_roundtrip(registry, _register_fake_algo):
    cfg = _base_config(_register_fake_algo)
    run_sweep(cfg, grid({"x": [1, 2, 3]}), registry=registry)

    results = summarize_algorithm(_register_fake_algo, registry=registry)
    assert len(results) == 3

    summary = summarize_metric(results, "objective")
    assert summary.n == 3
    assert summary.minimum == 1.0
    assert summary.maximum == 9.0

    diff = compare_quantum_vs_classical(results[0])
    assert diff is not None and "objective" in diff

    md = render_markdown_summary("Test Sweep", [summary])
    assert "Test Sweep" in md
    assert "objective" in md


def test_result_processing_rows_and_aggregate(registry, _register_fake_algo):
    cfg = _base_config(_register_fake_algo)
    run_sweep(cfg, grid({"x": [1, 2]}), registry=registry)

    results = summarize_algorithm(_register_fake_algo, registry=registry)
    rows = to_rows(results)
    assert len(rows) == 2
    assert all("metric.objective" in row for row in rows)
    assert all("classical.objective" in row for row in rows)

    by_algo = aggregate_by_algorithm(registry=registry)
    assert _register_fake_algo in by_algo
    assert len(by_algo[_register_fake_algo]) == 2
