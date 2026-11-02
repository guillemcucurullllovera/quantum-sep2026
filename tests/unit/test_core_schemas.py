from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)


def _make_result() -> ExperimentResult:
    backend = BackendDescriptor(name="aer_simulator", kind=BackendKind.SIMULATOR, num_qubits=8)
    config = ExperimentConfig(name="t1", algorithm="qaoa_qubo", backend=backend)
    meta = RunMetadata(algorithm="qaoa_qubo", generation="gen2")
    return ExperimentResult(metadata=meta, config=config, metrics={"objective": 1.23})


def test_experiment_result_roundtrip_json():
    result = _make_result()
    payload = result.model_dump_json()
    restored = ExperimentResult.model_validate_json(payload)
    assert restored.metadata.run_id == result.metadata.run_id
    assert restored.metrics["objective"] == 1.23


def test_run_id_is_unique():
    a = RunMetadata(algorithm="x")
    b = RunMetadata(algorithm="x")
    assert a.run_id != b.run_id
