import tempfile
from pathlib import Path

from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from tools.result_aggregation import aggregate
from tools.validation import validate, _ALGO_TEST_PATHS


def test_aggregate_returns_recorded_runs(monkeypatch, tmp_path):
    db = tmp_path / "reg.sqlite3"
    monkeypatch.setattr("tools.result_aggregation.ExperimentRegistry", lambda: ExperimentRegistry(db_path=db))

    reg = ExperimentRegistry(db_path=db)
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=4)
    config = ExperimentConfig(name="t", algorithm="_test_algo_agg", backend=backend)
    meta = RunMetadata(algorithm="_test_algo_agg")
    reg.record(ExperimentResult(metadata=meta, config=config, metrics={"x": 1.0}))

    rows = aggregate("_test_algo_agg")
    assert len(rows) == 1
    assert rows[0]["metrics"]["x"] == 1.0


def test_aggregate_empty_for_unknown_algorithm(monkeypatch, tmp_path):
    db = tmp_path / "reg2.sqlite3"
    monkeypatch.setattr("tools.result_aggregation.ExperimentRegistry", lambda: ExperimentRegistry(db_path=db))
    assert aggregate("nonexistent_algorithm_xyz") == []


def test_validate_unknown_algorithm_fails_clearly():
    passed, msg = validate("totally_unknown_algorithm")
    assert passed is False
    assert "No known test mapping" in msg


def test_validate_known_algorithm_paths_exist():
    for algo, paths in _ALGO_TEST_PATHS.items():
        assert any(Path(p).exists() for p in paths), f"none of {paths} exist for {algo}"
