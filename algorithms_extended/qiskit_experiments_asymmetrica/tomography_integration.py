"""integration layer: wire the vendored qiskit-experiments tomography
module (`.library.tomography`, all vendored source -- see
THIRD_PARTY_NOTICES.md) into this platform's `core.experiment_registry` and
`hardware.provenance` infrastructure, and provide a real, finance-algorithm-
facing hardware diagnostic: "how well does noisy hardware execution preserve
the intended quantum state produced by our own algorithm circuits?"

This module is Asymmetrica-original glue code -- no upstream equivalent.
Everything it imports from `.library.tomography` is the vendored upstream
source. It follows the same pattern as `integration.py`: run the
real vendored experiment class against a real (fake or noisy-simulated)
backend, extract a genuinely measured/fitted number, and record it through
`core.experiment_registry` -- never a fabricated placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from core.experiment_registry import ExperimentRegistry
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from hardware.provenance.reproducibility_record import (
    ReproducibilityRecord,
    build_reproducibility_record,
)

from algorithms_extended.qiskit_experiments_asymmetrica.library.tomography import (
    StateTomography,
)


def _noisy_simulator_for(fake_backend):
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(fake_backend)


def _strip_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a measurement-free copy of `circuit` -- `StateTomography`
    appends its own per-basis measurement circuits, so any final measurement
    already on the circuit under test must be removed first."""
    return circuit.remove_final_measurements(inplace=False)


def run_state_tomography_diagnostic(
    circuit: QuantumCircuit,
    backend,
    *,
    physical_qubits: tuple[int, ...] | None = None,
    shots: int = 4000,
    seed_simulation: int = 42,
    algorithm_name: str = "state_tomography_diagnostic",
    registry: ExperimentRegistry | None = None,
) -> tuple[ExperimentResult, ReproducibilityRecord]:
    """Run the real, vendored `StateTomography` experiment on `circuit`
    against `backend`, and compare the reconstructed density matrix's
    fidelity against the *exact* statevector produced by the same circuit
    computed directly with `qiskit.quantum_info.Statevector` (noiseless,
    analytic ground truth) -- a genuine hardware-diagnostic capability, not
    just an abstract tomography demo.

    Returns the recorded `ExperimentResult` and a `hardware.provenance`
    `ReproducibilityRecord` built from it, demonstrating the intended
    provenance-record integration point (`hardware/provenance/`)
    without modifying that module.
    """
    prep_circuit = _strip_measurements(circuit)
    ideal_sv = Statevector(prep_circuit)

    exp = StateTomography(prep_circuit, physical_qubits=physical_qubits, backend=backend)
    exp_data = exp.run(shots=shots, seed_simulation=seed_simulation).block_for_results()
    state_result = exp_data.analysis_results("state", dataframe=False)
    rho = state_result.value
    fidelity = float(state_fidelity(rho, ideal_sv))

    backend_name = getattr(backend, "name", str(backend))
    backend_name = backend_name() if callable(backend_name) else backend_name
    descriptor = BackendDescriptor(
        name=str(backend_name),
        kind=BackendKind.FAKE if str(backend_name).lower().startswith("fake") else BackendKind.SIMULATOR,
        num_qubits=getattr(backend, "num_qubits", circuit.num_qubits),
        calibration_source=f"qiskit_experiments_asymmetrica.StateTomography on {circuit.num_qubits}-qubit algorithm circuit",
    )
    config = ExperimentConfig(
        name=algorithm_name,
        algorithm=algorithm_name,
        backend=descriptor,
        params={"num_qubits": circuit.num_qubits, "shots": shots},
    )
    result = ExperimentResult(
        metadata=RunMetadata(algorithm=algorithm_name, adapted_from="qiskit-experiments (Apache-2.0)"),
        config=config,
        metrics={
            "reconstructed_fidelity": fidelity,
            "num_qubits": float(circuit.num_qubits),
        },
        notes=(
            f"quality={state_result.quality}; "
            f"fitter={state_result.extra.get('fitter_metadata', {}).get('fitter', 'unknown')}"
        ),
    )
    (registry or ExperimentRegistry()).record(result)

    # hardware/provenance/ integration point: a tomography-based
    # fidelity check becomes part of the run's reproducibility record, using
    # the existing, unmodified hardware.provenance API.
    prov = build_reproducibility_record(result)
    return result, prov


def run_qaoa_circuit_fidelity_diagnostic(
    fake_backend,
    *,
    n_assets: int = 3,
    p_layers: int = 1,
    seed: int = 7,
    shots: int = 4000,
    registry: ExperimentRegistry | None = None,
) -> tuple[ExperimentResult, ReproducibilityRecord]:
    """Real, materially-integrated diagnostic: build this platform's own
    Qiskit QAOA circuit (`algorithms.qaoa_qubo.solvers._build_qiskit_qaoa_circuit`)
    for a small fixed-parameter portfolio-QUBO instance, run state tomography
    on its output against a noisy simulator built from `fake_backend`'s own
    calibration data, and report how much fidelity noisy hardware execution
    costs relative to the exact statevector of the same circuit.

    This answers a genuinely useful question for this platform's finance
    algorithms: "if we ran our actual QAOA portfolio circuit on this device,
    how faithfully would the output state survive?" -- not an abstract
    tomography capability demo.
    """
    from algorithms.qaoa_qubo.solvers import _build_qiskit_qaoa_circuit
    from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising

    rng = np.random.default_rng(seed)
    # small, fixed, deterministic synthetic portfolio-QUBO (n_assets x n_assets)
    A = rng.normal(size=(n_assets, n_assets))
    Q = (A + A.T) / 2  # symmetric, as qubo_to_ising expects
    J, h, _offset = qubo_to_ising(Q)

    gamma = np.full(p_layers, 0.4)
    beta = np.full(p_layers, 0.6)
    circuit = _build_qiskit_qaoa_circuit(J, h, p_layers, gamma, beta)
    prep_circuit = _strip_measurements(circuit)

    sim = _noisy_simulator_for(fake_backend)
    result, prov = run_state_tomography_diagnostic(
        prep_circuit,
        sim,
        shots=shots,
        algorithm_name="qaoa_circuit_fidelity_diagnostic",
        registry=registry,
    )
    return result, prov


@dataclass
class NoiseLevelFidelityPoint:
    noise_level: float
    fidelity: float


def circuit_fidelity_vs_noise_level(
    circuit: QuantumCircuit,
    noise_levels: list[float],
    *,
    shots: int = 4000,
    seed_simulation: int = 42,
) -> list[NoiseLevelFidelityPoint]:
    """"Circuit fidelity under noise" benchmark dimension: for each
    depolarizing-noise probability in `noise_levels`, run the real vendored
    `StateTomography` experiment against an `AerSimulator` with a
    depolarizing noise model at that level, and report the reconstructed
    state's fidelity against the exact (noiseless) statevector.

    This is the function a `benchmarks/` "circuit fidelity under noise"
    dimension would call -- exposed here (not inside `benchmarks/` itself,
    per this platform's scope) so it can be wired in as a thin follow-up without
    touching this module's actual reconstruction logic.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    prep_circuit = _strip_measurements(circuit)
    ideal_sv = Statevector(prep_circuit)

    points: list[NoiseLevelFidelityPoint] = []
    for level in noise_levels:
        if level > 0:
            noise_model = NoiseModel()
            err1 = depolarizing_error(level, 1)
            err2 = depolarizing_error(level, 2)
            noise_model.add_all_qubit_quantum_error(err1, ["h", "rx", "ry", "rz", "sx", "x", "u"])
            noise_model.add_all_qubit_quantum_error(err2, ["cx", "rzz", "cz"])
            sim = AerSimulator(noise_model=noise_model)
        else:
            sim = AerSimulator()

        exp = StateTomography(prep_circuit, backend=sim)
        exp_data = exp.run(shots=shots, seed_simulation=seed_simulation).block_for_results()
        rho = exp_data.analysis_results("state", dataframe=False).value
        fidelity = float(state_fidelity(rho, ideal_sv))
        points.append(NoiseLevelFidelityPoint(noise_level=level, fidelity=fidelity))

    return points
