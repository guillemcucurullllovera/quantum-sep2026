import json

import pytest

from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from hardware.provenance import build_reproducibility_record
from tools.provenance_tool import main as provenance_main


def test_provenance_tool_cli_reports_missing_fields(monkeypatch, tmp_path, capsys):
    db = tmp_path / "reg.sqlite3"
    reg = ExperimentRegistry(db_path=db)
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=2)
    config = ExperimentConfig(name="t", algorithm="_test_prov_algo", backend=backend)
    meta = RunMetadata(algorithm="_test_prov_algo")
    result = ExperimentResult(metadata=meta, config=config, metrics={"x": 1.0})
    run_id = reg.record(result)

    monkeypatch.setattr("tools.provenance_tool.ExperimentRegistry", lambda: ExperimentRegistry(db_path=db))
    rc = provenance_main(["--run-id", run_id])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["run_id"] == run_id


def test_provenance_tool_unknown_run_id_fails_cleanly(monkeypatch, tmp_path):
    db = tmp_path / "reg2.sqlite3"
    monkeypatch.setattr("tools.provenance_tool.ExperimentRegistry", lambda: ExperimentRegistry(db_path=db))
    rc = provenance_main(["--run-id", "does-not-exist"])
    assert rc == 1


def test_build_reproducibility_record_flags_missing_hardware_context():
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=2)
    config = ExperimentConfig(name="t", algorithm="x", backend=backend)
    meta = RunMetadata(algorithm="x")
    result = ExperimentResult(metadata=meta, config=config, metrics={})
    record = build_reproducibility_record(result)
    # no hardware.experiment_tracking context was attached -> honestly incomplete
    assert record.reproducible is False
    assert len(record.missing_fields) > 0
