"""Sampler/Estimator abstraction: one interface (`run_sampler`,
`run_estimator`) that works identically against a local Aer simulator or a
real Qiskit Runtime primitive (once credentials exist) — the caller never
branches on backend kind.

Matches the installed qiskit-ibm-runtime 0.49.0 SamplerV2/EstimatorV2 API
(`primitive.run([pub])` with a PUB — Primitive Unified Bloc — tuple).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import BackendDescriptor, BackendKind


@dataclass
class SamplerRunResult:
    counts: dict[str, int]
    backend_name: str
    shots: int


def _counts_from_aer_samplerv2_result(result) -> dict[str, int]:
    pub_result = result[0]
    data_bin = pub_result.data
    reg_name = "meas" if hasattr(data_bin, "meas") else list(data_bin.__dict__.keys())[0]
    return getattr(data_bin, reg_name).get_counts()


def _truncate_to_active_qubits(circuit):
    """Bug fix (diagnosed via tests/algorithms_extended/test_qiskit_finance_extensions.py
    ::test_execution_wiring_readout_m3_on_fake_backend, previously xfail strict=True):

    A circuit transpiled against a real fake-device backend (e.g. FakeSherbrooke,
    127 qubits) carries a qubit register sized to the WHOLE device even though a
    typical algorithm circuit only ever touches a handful of those physical
    qubits -- the rest are idle placeholders introduced purely by the device
    layout. Simulating that full-width register forces AerSimulator to allocate
    a dense statevector/density-matrix sized to all 127/133 qubits, regardless
    of how few are actually entangled. For an all-Clifford circuit (e.g. a bare
    Bell state) Aer silently auto-selects its cheap stabilizer method regardless
    of qubit count, which is why the platform's existing FAKE-backend tests
    never caught this. But any circuit containing a non-Clifford rotation gate
    (every real algorithm circuit here -- QAOA, VQC, QAE, payoff circuits, ...)
    forces a full statevector/density-matrix fallback sized to the whole
    backend, which exceeds max_memory_mb and fails with "a circuit requires
    more memory than max_memory_mb" (reproduced directly: a 4-qubit circuit
    with one non-Clifford rotation, transpiled onto FakeSherbrooke and then
    measured with `measure_all()` -- which measures all 127 physical qubits,
    not just the 4 active ones -- OOMs; the same circuit with only its 4
    active qubits measured runs in ~1.6s).

    The fix: build a fresh, minimal circuit containing only the qubits the
    circuit actually touches with a real gate (plus their measurements),
    remapped onto a dense 0..k-1 register, and drop measurements on every
    other (idle) qubit. This is exact, not an approximation -- an idle qubit
    with no gate applied to it and no scheduled delay/idle noise (this
    platform never calls a scheduling pass that would insert Delay
    instructions) stays deterministically |0>, so its measurement outcome is
    always '0' and dropping it changes nothing about the physics, only how
    much of the Hilbert space Aer has to materialize.

    The source of truth for "actually touches" is always a direct scan of
    the circuit's own instructions for gate-bearing qubits (excluding
    measure/barrier/delay) -- NOT `layout.final_index_layout()` alone. This
    matters because this FAKE path re-transpiles whatever circuit it is
    given (see the `transpile(...)` call right above the call site): if the
    caller already transpiled+measured the circuit against the full device
    (as algorithms_extended.qiskit_finance_extensions.execution does, via
    `measure_all()` on an already-127-wide circuit -- which measures every
    physical qubit, not just the active ones), that measurement on every
    qubit makes the *second* transpile's own `final_index_layout()` a
    trivial full-width permutation again, silently defeating a
    layout-only approach. A direct instruction scan sees straight through
    that: gate-bearing qubits stay few no matter how many times the circuit
    gets re-transpiled or how many idle qubits picked up a measurement
    along the way. When a (non-trivial, already-reduced) `layout` is also
    present, its `final_index_layout()` order is used only to order the
    scanned active qubits -- matching the order downstream mitigation code
    (e.g. algorithms_extended.qiskit_finance_extensions.execution.
    _physical_qubits_measured) derives the same way -- not to select them.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    try:
        seen: list[int] = []
        for instr in circuit.data:
            if instr.operation.name in ("measure", "barrier", "delay"):
                continue
            for q in instr.qubits:
                idx = circuit.find_bit(q).index
                if idx not in seen:
                    seen.append(idx)
        active_physical: list[int] = seen

        layout = getattr(circuit, "layout", None)
        if layout is not None and active_physical:
            try:
                layout_order = [int(q) for q in layout.final_index_layout()]
                active_set_for_order = set(active_physical)
                ordered = [p for p in layout_order if p in active_set_for_order]
                if len(ordered) == len(active_physical):
                    active_physical = ordered
            except Exception:
                pass

        if not active_physical or len(active_physical) >= circuit.num_qubits:
            return circuit  # nothing to truncate, or nothing recognizably active

        active_set = set(active_physical)
        dense_index = {phys: i for i, phys in enumerate(active_physical)}

        measured_phys: list[int] = []
        for instr in circuit.data:
            if instr.operation.name == "measure":
                phys = circuit.find_bit(instr.qubits[0]).index
                if phys in active_set and phys not in measured_phys:
                    measured_phys.append(phys)
        # Preserve logical-qubit order (matches final_index_layout()'s order),
        # not raw physical-index order, so downstream mitigation code that
        # derives its own qubit list the same way stays aligned with this one.
        measured_phys_ordered = [p for p in active_physical if p in measured_phys]

        qreg = QuantumRegister(len(active_physical), "q")
        reduced = QuantumCircuit(qreg, name=circuit.name)
        creg = None
        clbit_index: dict[int, int] = {}
        if measured_phys_ordered:
            creg = ClassicalRegister(len(measured_phys_ordered), "meas")
            reduced.add_register(creg)
            clbit_index = {phys: i for i, phys in enumerate(measured_phys_ordered)}

        for instr in circuit.data:
            op = instr.operation
            if op.name in ("barrier", "delay"):
                continue
            if op.name == "measure":
                phys = circuit.find_bit(instr.qubits[0]).index
                if phys not in active_set:
                    continue  # idle qubit: deterministically |0>, drop it
                reduced.measure(qreg[dense_index[phys]], creg[clbit_index[phys]])
                continue
            new_qubits = [qreg[dense_index[circuit.find_bit(q).index]] for q in instr.qubits]
            reduced.append(op, new_qubits, instr.clbits)
        return reduced
    except Exception:
        # Defensive fallback: never let the truncation itself break a
        # previously-working call path -- run the untouched circuit as
        # before (pre-fix behavior) if anything here doesn't apply.
        return circuit


def submit_qpu_sampler_job(circuit, backend_descriptor: BackendDescriptor, backend, shots: int = 4096):
    """QPU submission, SEPARATED from waiting for the result (see
    `fetch_qpu_sampler_result` and `hardware/qpu_execution/execute.py`'s
    module docstring for why this split exists): once this call returns a
    job handle, a real job exists and may already be billed/queued. A
    caller that wants to retry a transient failure must retry
    `fetch_qpu_sampler_result` against the SAME returned job, never call
    this function again for the same logical request — calling it twice
    creates two real, independent jobs.
    """
    if backend_descriptor.kind != BackendKind.QPU:
        raise ValueError(
            f"submit_qpu_sampler_job is QPU-only, got backend_descriptor.kind={backend_descriptor.kind!r}"
        )
    from core.config import get_settings
    from qiskit_ibm_runtime import SamplerV2

    get_settings().ibm.require()
    sampler = SamplerV2(mode=backend)
    return sampler.run([circuit], shots=shots)


def fetch_qpu_sampler_result(job, backend_descriptor: BackendDescriptor, shots: int) -> SamplerRunResult:
    """Wait for and parse an ALREADY-SUBMITTED job's result. Never submits
    anything itself — safe to retry (re-poll/re-fetch the same job object)
    without any risk of duplicate real-hardware spend, unlike retrying a
    function that both submits and waits in one call."""
    result = job.result()
    counts = result[0].data.meas.get_counts()
    return SamplerRunResult(counts=counts, backend_name=backend_descriptor.name, shots=shots)


def run_sampler(circuit, backend_descriptor: BackendDescriptor, backend, shots: int = 4096) -> SamplerRunResult:
    if backend_descriptor.kind == BackendKind.QPU:
        job = submit_qpu_sampler_job(circuit, backend_descriptor, backend, shots=shots)
        return fetch_qpu_sampler_result(job, backend_descriptor, shots)

    elif backend_descriptor.kind == BackendKind.FAKE:
        # A FAKE backend's whole purpose is device-calibrated noise — route
        # through AerSimulator.from_backend(backend) so that noise is
        # actually applied, rather than silently falling back to a clean
        # simulator (which would make every "noisy" experiment secretly
        # noiseless — the exact silent-fallback failure mode this platform
        # explicitly rejects for credentials, and must reject here too).
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        noisy_backend = AerSimulator.from_backend(backend)
        transpiled = transpile(circuit, backend=backend, seed_transpiler=42)
        transpiled = _truncate_to_active_qubits(transpiled)
        result = noisy_backend.run(transpiled, shots=shots).result()
        counts = result.get_counts()

    else:
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

        sampler = AerSamplerV2()
        job = sampler.run([circuit], shots=shots)
        result = job.result()
        counts = _counts_from_aer_samplerv2_result(result)

    return SamplerRunResult(counts=counts, backend_name=backend_descriptor.name, shots=shots)


@dataclass
class EstimatorRunResult:
    expectation_values: list[float]
    backend_name: str


def run_estimator(circuit, observables: list, backend_descriptor: BackendDescriptor, backend) -> EstimatorRunResult:
    if backend_descriptor.kind == BackendKind.QPU:
        from core.config import get_settings
        from qiskit_ibm_runtime import EstimatorV2

        get_settings().ibm.require()
        estimator = EstimatorV2(mode=backend)
        job = estimator.run([(circuit, observables)])
        result = job.result()
        evs = list(result[0].data.evs)
    elif backend_descriptor.kind == BackendKind.FAKE:
        # As with the sampler path: the noise model only has entries for
        # the backend's native basis gates. Passing an untranspiled circuit
        # (e.g. containing "h"/"cx") means those gate names never match a
        # noise instruction and the run is silently noiseless again. Must
        # transpile to the backend first, and correspondingly permute each
        # observable onto the transpiled circuit's physical layout — the
        # standard EstimatorV2 ISA pattern — or the expectation value would
        # be computed against the wrong physical qubits.
        from qiskit import transpile
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        noise_model = NoiseModel.from_backend(backend)
        transpiled = transpile(circuit, backend=backend, seed_transpiler=42)
        laid_out_observables = [obs.apply_layout(transpiled.layout) for obs in observables]
        estimator = AerEstimatorV2(options={"backend_options": {"noise_model": noise_model}})
        job = estimator.run([(transpiled, laid_out_observables)])
        result = job.result()
        evs = list(result[0].data.evs)

    else:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

        estimator = AerEstimatorV2()
        job = estimator.run([(circuit, observables)])
        result = job.result()
        evs = list(result[0].data.evs)

    return EstimatorRunResult(expectation_values=evs, backend_name=backend_descriptor.name)


__all__ = ["run_sampler", "run_estimator", "SamplerRunResult", "EstimatorRunResult"]
