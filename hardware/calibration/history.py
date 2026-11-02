"""Track calibration snapshots over time via `core.experiment_registry`, so
calibration drift for a given backend becomes queryable rather than only
ever visible as a single point-in-time value.
"""
from __future__ import annotations

from core.experiment_registry import ExperimentRegistry
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from hardware.calibration.ingest import CalibrationSnapshot, extract_calibration


class CalibrationHistory:
    """Thin wrapper over `ExperimentRegistry` specialized for calibration
    snapshots — every `record()` call is logged as an `ExperimentResult`
    tagged `algorithm="calibration_snapshot"`, so `list_snapshots` can pull
    a backend's calibration history back out for drift analysis.
    """

    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    def record(self, backend) -> CalibrationSnapshot:
        snapshot = extract_calibration(backend)
        descriptor = BackendDescriptor(
            name=snapshot.backend_name, kind=BackendKind.FAKE, num_qubits=snapshot.n_qubits
        )
        config = ExperimentConfig(name="calibration_snapshot", algorithm="calibration_snapshot", backend=descriptor)
        meta = RunMetadata(algorithm="calibration_snapshot")
        result = ExperimentResult(
            metadata=meta,
            config=config,
            metrics={
                k: v
                for k, v in {
                    "t1_mean_us": snapshot.t1_mean_us,
                    "two_qubit_error_mean": snapshot.two_qubit_error_mean,
                }.items()
                if v is not None
            },
            notes=snapshot.model_dump_json(),
        )
        self.registry.record(result)
        return snapshot

    def list_snapshots(self, backend_name: str) -> list[CalibrationSnapshot]:
        run_ids = self.registry.list_by_algorithm("calibration_snapshot")
        out = []
        for run_id in run_ids:
            result = self.registry.get(run_id)
            snap = CalibrationSnapshot.model_validate_json(result.notes)
            if snap.backend_name == backend_name:
                out.append(snap)
        return sorted(out, key=lambda s: s.captured_at)

    def drift(self, backend_name: str) -> dict:
        """t1_mean_us / two_qubit_error_mean delta between the earliest and
        latest recorded snapshot for a backend — real drift measurement,
        not a schema placeholder."""
        snaps = self.list_snapshots(backend_name)
        if len(snaps) < 2:
            return {"n_snapshots": len(snaps), "drift_detectable": False}
        first, last = snaps[0], snaps[-1]
        out = {"n_snapshots": len(snaps), "drift_detectable": True}
        if first.t1_mean_us is not None and last.t1_mean_us is not None:
            out["t1_mean_us_delta"] = last.t1_mean_us - first.t1_mean_us
        if first.two_qubit_error_mean is not None and last.two_qubit_error_mean is not None:
            out["two_qubit_error_mean_delta"] = last.two_qubit_error_mean - first.two_qubit_error_mean
        return out
