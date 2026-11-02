import tempfile
from pathlib import Path

import pytest

from core.errors import ExperimentNotFoundError
from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata


@pytest.fixture()
def registry():
    with tempfile.TemporaryDirectory() as tmp:
        yield ExperimentRegistry(db_path=Path(tmp) / "reg.sqlite3")


def _result(algorithm: str) -> ExperimentResult:
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4)
    config = ExperimentConfig(name="t", algorithm=algorithm, backend=backend)
    meta = RunMetadata(algorithm=algorithm)
    return ExperimentResult(metadata=meta, config=config, metrics={"x": 1.0})


def test_record_and_get_roundtrip(registry):
    r = _result("qaoa_qubo")
    run_id = registry.record(r)
    fetched = registry.get(run_id)
    assert fetched.metadata.algorithm == "qaoa_qubo"
    assert fetched.metrics["x"] == 1.0


def test_missing_run_raises(registry):
    with pytest.raises(ExperimentNotFoundError):
        registry.get("does-not-exist")


def test_list_by_algorithm_and_count(registry):
    registry.record(_result("qaoa_qubo"))
    registry.record(_result("qaoa_qubo"))
    registry.record(_result("vqc_qml"))
    assert len(registry.list_by_algorithm("qaoa_qubo")) == 2
    assert registry.count() == 3
