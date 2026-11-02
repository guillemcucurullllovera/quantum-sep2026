"""Integration layer: wire the vendored qiskit-experiments framework into
this platform's own `core.experiment_registry` / `hardware.calibration`
infrastructure, rather than letting ExperimentData live in a parallel silo.

This is the Asymmetrica-original glue code for this adaptation — everything
it imports from `.framework` / `.library.characterization` /
`.library.randomized_benchmarking` is the vendored upstream code (see
THIRD_PARTY_NOTICES.md); this module itself has no upstream equivalent.
"""
from __future__ import annotations

import numpy as np

from core.experiment_registry import ExperimentRegistry
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from hardware.calibration.history import CalibrationHistory
from hardware.calibration.ingest import CalibrationSnapshot

from algorithms_extended.qiskit_experiments_asymmetrica.library.characterization.t1 import T1
from algorithms_extended.qiskit_experiments_asymmetrica.library.characterization.t2ramsey import T2Ramsey
from algorithms_extended.qiskit_experiments_asymmetrica.library.randomized_benchmarking.standard_rb import (
    StandardRB,
)


def _noisy_simulator_for(fake_backend):
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(fake_backend)


def run_t1_experiment(fake_backend, physical_qubit: int = 0, registry: ExperimentRegistry | None = None) -> ExperimentResult:
    """Run the real, vendored T1 experiment against a noisy simulator built
    from `fake_backend`'s own calibration data, and log a real, fitted T1
    estimate — not the backend's declared T1 read back verbatim, an
    independently *measured* one from actual simulated decay curves.
    """
    sim = _noisy_simulator_for(fake_backend)
    declared_t1_s = fake_backend.target.qubit_properties[physical_qubit].t1
    # delays spanning roughly 0 to ~3x the declared T1, so the exponential
    # decay is actually visible in the sampled range
    delays = np.linspace(1e-6, max(3 * declared_t1_s, 3e-6), 30)
    exp = T1(physical_qubits=(physical_qubit,), delays=delays, backend=sim)
    exp_data = exp.run().block_for_results()
    df = exp_data.analysis_results(dataframe=True)
    t1_row = df[df["name"] == "T1"].iloc[0]
    measured_t1_s = float(t1_row["value"].nominal_value)

    descriptor = BackendDescriptor(
        name=getattr(fake_backend, "name", str(fake_backend)),
        kind=BackendKind.FAKE,
        num_qubits=fake_backend.num_qubits,
        calibration_source=f"qiskit_experiments_asymmetrica.T1 on qubit {physical_qubit}",
    )
    config = ExperimentConfig(
        name="t1_characterization", algorithm="qiskit_experiments_t1", backend=descriptor,
        params={"physical_qubit": physical_qubit},
    )
    result = ExperimentResult(
        metadata=RunMetadata(algorithm="qiskit_experiments_t1", generation=None,
                              adapted_from="qiskit-experiments (Apache-2.0)"),
        config=config,
        metrics={
            "measured_t1_us": measured_t1_s * 1e6,
            "declared_t1_us": declared_t1_s * 1e6,
            "relative_error": abs(measured_t1_s - declared_t1_s) / declared_t1_s,
        },
        notes=f"quality={t1_row.get('quality', 'unknown')}",
    )
    (registry or ExperimentRegistry()).record(result)

    # feed the same run into CalibrationHistory as a richer, *measured*
    # (not just declared) ingestion source
    ch = CalibrationHistory(registry=registry or ExperimentRegistry())
    snap = CalibrationSnapshot(
        backend_name=descriptor.name,
        n_qubits=fake_backend.num_qubits,
        t1_us={physical_qubit: measured_t1_s * 1e6},
        t1_mean_us=measured_t1_s * 1e6,
    )
    ch.registry.record(ExperimentResult(
        metadata=RunMetadata(algorithm="calibration_snapshot"),
        config=ExperimentConfig(name="calibration_snapshot", algorithm="calibration_snapshot", backend=descriptor),
        metrics={"t1_mean_us": snap.t1_mean_us},
        notes=snap.model_dump_json(),
    ))
    return result


def run_t2_experiment(fake_backend, physical_qubit: int = 0, registry: ExperimentRegistry | None = None) -> ExperimentResult:
    """Real, vendored T2Ramsey experiment — measured T2*, compared against
    the backend's declared T2."""
    sim = _noisy_simulator_for(fake_backend)
    declared_t2_s = fake_backend.target.qubit_properties[physical_qubit].t2
    delays = np.linspace(1e-6, max(2 * declared_t2_s, 2e-6), 40)
    exp = T2Ramsey(physical_qubits=(physical_qubit,), delays=delays, backend=sim, osc_freq=2e5)
    exp_data = exp.run().block_for_results()
    df = exp_data.analysis_results(dataframe=True)
    t2_row = df[df["name"] == "T2star"].iloc[0]
    measured_t2_s = float(t2_row["value"].nominal_value)

    descriptor = BackendDescriptor(
        name=getattr(fake_backend, "name", str(fake_backend)),
        kind=BackendKind.FAKE,
        num_qubits=fake_backend.num_qubits,
        calibration_source=f"qiskit_experiments_asymmetrica.T2Ramsey on qubit {physical_qubit}",
    )
    result = ExperimentResult(
        metadata=RunMetadata(algorithm="qiskit_experiments_t2", adapted_from="qiskit-experiments (Apache-2.0)"),
        config=ExperimentConfig(name="t2_characterization", algorithm="qiskit_experiments_t2",
                                 backend=descriptor, params={"physical_qubit": physical_qubit}),
        metrics={
            "measured_t2_us": measured_t2_s * 1e6,
            "declared_t2_us": declared_t2_s * 1e6,
            "relative_error": abs(measured_t2_s - declared_t2_s) / declared_t2_s,
        },
        notes=f"quality={t2_row.get('quality', 'unknown')}",
    )
    (registry or ExperimentRegistry()).record(result)
    return result


def run_standard_rb(
    fake_backend, physical_qubits: tuple[int, ...] = (0,),
    lengths: list[int] | None = None, num_samples: int = 3,
    registry: ExperimentRegistry | None = None,
) -> ExperimentResult:
    """Real, vendored standard randomized benchmarking — returns a genuine
    fitted per-Clifford error-per-gate (EPG) estimate from actual noisy
    simulation, the first RB-derived fidelity number anywhere in this
    platform."""
    sim = _noisy_simulator_for(fake_backend)
    lengths = lengths or [1, 10, 30, 50, 80]
    exp = StandardRB(physical_qubits=physical_qubits, lengths=lengths, num_samples=num_samples, seed=42, backend=sim)
    exp_data = exp.run().block_for_results()
    df = exp_data.analysis_results(dataframe=True)
    epc_row = df[df["name"] == "EPC"].iloc[0]
    epc = float(epc_row["value"].nominal_value)

    descriptor = BackendDescriptor(
        name=getattr(fake_backend, "name", str(fake_backend)),
        kind=BackendKind.FAKE,
        num_qubits=fake_backend.num_qubits,
        calibration_source=f"qiskit_experiments_asymmetrica.StandardRB on qubits {physical_qubits}",
    )
    result = ExperimentResult(
        metadata=RunMetadata(algorithm="qiskit_experiments_rb", adapted_from="qiskit-experiments (Apache-2.0)"),
        config=ExperimentConfig(name="randomized_benchmarking", algorithm="qiskit_experiments_rb",
                                 backend=descriptor, params={"physical_qubits": list(physical_qubits), "lengths": lengths}),
        metrics={"error_per_clifford": epc, "fidelity_per_clifford": 1.0 - epc},
        notes=f"quality={epc_row.get('quality', 'unknown')}",
    )
    (registry or ExperimentRegistry()).record(result)
    return result
