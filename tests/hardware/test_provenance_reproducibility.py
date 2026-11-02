from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    MitigationConfig,
    MitigationKind,
    RunMetadata,
)
from hardware.experiment_tracking import hardware_context
from hardware.provenance import build_reproducibility_record


def test_reproducibility_record_complete_when_context_present():
    backend = BackendDescriptor(
        name="FakeSherbrooke", kind=BackendKind.FAKE, num_qubits=127, calibration_source="fake_backend:FakeSherbrooke"
    )
    config = ExperimentConfig(
        name="t", algorithm="pqk", backend=backend, mitigation=MitigationConfig(kind=MitigationKind.READOUT_M3), seed=7
    )
    meta = RunMetadata(algorithm="pqk", seed=7)
    ctx = hardware_context(backend, transpilation_strategy="depth_optimized", mitigation=config.mitigation)
    import json

    result = ExperimentResult(metadata=meta, config=config, notes=json.dumps(ctx))
    record = build_reproducibility_record(result)
    assert record.reproducible is True
    assert record.missing_fields == []
    assert record.backend_name == "FakeSherbrooke"
    assert record.transpilation_strategy == "depth_optimized"
    assert record.seed == 7


def test_reproducibility_record_flags_missing_fields_honestly():
    backend = BackendDescriptor(name="unknown", kind=BackendKind.SIMULATOR, num_qubits=4, calibration_source=None)
    config = ExperimentConfig(name="t", algorithm="pqk", backend=backend)
    meta = RunMetadata(algorithm="pqk")
    result = ExperimentResult(metadata=meta, config=config)
    record = build_reproducibility_record(result)
    assert record.reproducible is False
    assert "calibration_source" in record.missing_fields
