"""Stamp hardware run context (backend, calibration snapshot, transpilation
strategy, mitigation applied) onto an ExperimentResult so a benchmark
result read out of the registry months later is self-describing about
exactly what hardware conditions produced it — no tribal knowledge needed.
"""
from __future__ import annotations

from core.schemas import BackendDescriptor, MitigationConfig, NoiseModelDescriptor


def hardware_context(
    backend_descriptor: BackendDescriptor,
    *,
    transpilation_strategy: str | None = None,
    noise_model: NoiseModelDescriptor | None = None,
    mitigation: MitigationConfig | None = None,
    circuit_depth: int | None = None,
    two_qubit_gate_count: int | None = None,
) -> dict:
    """Build the `context` payload for a `core.schemas.BenchmarkResult`, or
    a value to stash under `ExperimentResult.artifacts`/notes — a plain
    dict so it composes with whichever schema field the caller is filling.
    """
    return {
        "backend_name": backend_descriptor.name,
        "backend_kind": backend_descriptor.kind.value,
        "calibration_source": backend_descriptor.calibration_source,
        "transpilation_strategy": transpilation_strategy,
        "noise_model_source": noise_model.source if noise_model else None,
        "noise_readout_error_mean": noise_model.readout_error_mean if noise_model else None,
        "noise_gate_error_mean": noise_model.gate_error_mean if noise_model else None,
        "mitigation_kind": mitigation.kind.value if mitigation else "none",
        "circuit_depth": circuit_depth,
        "two_qubit_gate_count": two_qubit_gate_count,
    }


__all__ = ["hardware_context"]
