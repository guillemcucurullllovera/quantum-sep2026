"""hardware.provenance.runtime_diagnostics -- capture which
backend/cloud/fallback-path was actually used for a given run, wired into
core.experiment_registry, so a result read back out of the registry months
later reveals whether it silently ran somewhere other than requested."""
from __future__ import annotations

import json

from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from hardware.backend_selection.fallback import resolve_with_fallback
from hardware.provenance.runtime_diagnostics import (
    build_runtime_diagnostics,
    record_with_runtime_diagnostics,
    stamp_runtime_diagnostics,
)


def _fresh_result() -> ExperimentResult:
    backend = BackendDescriptor(name="FakeTorino", kind=BackendKind.FAKE, num_qubits=133, calibration_source="fake_backend:FakeTorino")
    config = ExperimentConfig(name="t", algorithm="qaoa_qubo", backend=backend)
    meta = RunMetadata(algorithm="qaoa_qubo", seed=7)
    return ExperimentResult(metadata=meta, config=config)


def test_build_runtime_diagnostics_from_a_no_fallback_outcome():
    outcome = resolve_with_fallback("fake_heron")
    diagnostics = build_runtime_diagnostics(outcome)
    assert diagnostics.requested_backend == "fake_heron"
    assert diagnostics.fell_back is False
    assert diagnostics.actual_backend == outcome.actual.step_name
    assert diagnostics.actual_cloud == "ibm_quantum"
    assert len(diagnostics.attempts) == 1


def test_build_runtime_diagnostics_from_a_fallback_outcome():
    outcome = resolve_with_fallback("unknown_backend_for_diagnostics_test")
    diagnostics = build_runtime_diagnostics(outcome)
    assert diagnostics.fell_back is True
    assert diagnostics.requested_backend == "unknown_backend_for_diagnostics_test"
    assert diagnostics.actual_backend != diagnostics.requested_backend
    assert len(diagnostics.attempts) >= 2
    assert diagnostics.attempts[0]["ok"] is False
    assert diagnostics.attempts[-1]["ok"] is True


def test_stamp_runtime_diagnostics_preserves_existing_notes_payload():
    result = _fresh_result().model_copy(update={"notes": json.dumps({"backend_name": "FakeTorino", "mitigation_kind": "none"})})
    outcome = resolve_with_fallback("fake_heron")
    diagnostics = build_runtime_diagnostics(outcome)
    stamped = stamp_runtime_diagnostics(result, diagnostics)

    assert stamped is not result  # immutable-style: original untouched
    assert result.notes is not None and "runtime_diagnostics" not in result.notes

    payload = json.loads(stamped.notes)
    assert payload["backend_name"] == "FakeTorino"  # preserved
    assert payload["runtime_diagnostics"]["requested_backend"] == "fake_heron"
    assert payload["runtime_diagnostics"]["fell_back"] is False


def test_record_with_runtime_diagnostics_persists_to_experiment_registry(tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "diagnostics_test_registry.sqlite3")
    result = _fresh_result()
    outcome = resolve_with_fallback("unknown_backend_for_registry_test")

    stamped = record_with_runtime_diagnostics(result, outcome, registry=registry)

    fetched = registry.get(result.metadata.run_id)
    assert fetched.metadata.run_id == stamped.metadata.run_id
    payload = json.loads(fetched.notes)
    diag = payload["runtime_diagnostics"]
    assert diag["requested_backend"] == "unknown_backend_for_registry_test"
    assert diag["fell_back"] is True
    assert diag["actual_cloud"] == "ibm_quantum"  # falls through to fake_heron by default
