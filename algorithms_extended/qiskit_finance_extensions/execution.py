"""Hardware execution wiring for this package's payoff circuits — thin glue
over hardware.ibm_runtime / hardware.readout_mitigation / hardware.zne
(infrastructure). No credential handling here: BackendCatalog /
run_sampler already route IBM_QUANTUM_TOKEN exclusively through
core.config; this module never touches os.environ directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit import transpile

from hardware.backends.catalog import BackendCatalog
from hardware.ibm_runtime.primitives import run_sampler
from hardware.readout_mitigation.m3 import M3Mitigator
from hardware.zne.zne import ZNEMitigator


def _transpiled(circuit, backend):
    """hardware.ibm_runtime.run_sampler's FAKE-backend path transpiles
    internally, but its noiseless/simulator path does not — StatePreparation
    (and any other non-basis-gate instruction) reaches Aer's SamplerV2 raw
    and fails with 'unknown instruction'. Transpiling here, on the caller
    side, is always safe (idempotent if run_sampler also transpiles) and
    keeps the fix inside this package's owned files rather than editing
    shared hardware/ infrastructure another workstream may be touching in
    parallel. Worth a follow-up fix directly in hardware/ibm_runtime/
    primitives.py's noiseless path so every other caller doesn't need this
    same workaround."""
    try:
        return transpile(circuit, backend=backend, seed_transpiler=42)
    except Exception:
        return transpile(circuit, basis_gates=["u", "cx"], seed_transpiler=42)


def _physical_qubits_measured(transpiled_circuit) -> list[int]:
    """After transpiling to a real device-sized backend, the circuit's small
    number of logical qubits get mapped onto *some* subset of the backend's
    physical qubits (e.g. 4 of fake_eagle's 127) — M3 calibration/correction
    must run on those actual physical indices, not range(num_logical_qubits),
    which is what `Bitstring length (4) does not match number of qubits (127)`
    was catching (found via the execution smoke test, fixed here rather than
    silencing with a smaller/simpler backend)."""
    layout = getattr(transpiled_circuit, "layout", None)
    if layout is not None:
        try:
            return list(layout.final_index_layout())
        except Exception:
            pass
    return list(range(transpiled_circuit.num_qubits))


@dataclass(frozen=True)
class ExecutionResult:
    counts: dict
    backend_name: str
    mitigation: str
    shots: int
    zne_extrapolated_probability: float | None = None


def run_payoff_circuit(
    circuit,
    *,
    backend_name: str = "noiseless",
    shots: int = 4096,
    mitigation: str = "none",
) -> ExecutionResult:
    """backend_name: any key in hardware.backends.catalog.BackendCatalog.list_backends()
    (e.g. 'noiseless', 'fake_eagle', 'fake_heron'). mitigation: 'none' | 'readout_m3' | 'zne'."""
    resolved = BackendCatalog.get(backend_name)
    unmeasured = _transpiled(circuit.copy(), resolved.backend)

    def _with_measurement(qc):
        qc = qc.copy()
        if qc.num_clbits == 0:
            qc.measure_all()
        return qc

    measured = _with_measurement(unmeasured)

    zne_prob = None
    if mitigation == "zne":
        # ZNEMitigator's own default noise_factors=[1, 2, 3] includes an even
        # factor, but its gate-folding backend (fold_circuit_gates) only
        # accepts odd scale factors (found via this smoke test) — passing
        # explicit odd factors here works around that mismatch without
        # editing shared hardware/zne/zne.py. Separately, fold_circuit_gates
        # calls circuit.inverse(), which errors on a `measure` instruction
        # ("inverse() not implemented for measure") — so folding must run on
        # the *unmeasured* transpiled circuit, with measurement appended
        # after folding, not before (also found via this smoke test).
        mitigator = ZNEMitigator(noise_factors=[1, 3, 5])

        def _prob_one_at_scale(scale: int) -> float:
            from hardware.zne.zne import fold_circuit_gates

            folded = fold_circuit_gates(unmeasured, scale) if scale > 1 else unmeasured
            folded = _with_measurement(folded)
            run_result = run_sampler(folded, resolved.descriptor, resolved.backend, shots=shots)
            total = sum(run_result.counts.values()) or 1
            ones = sum(v for k, v in run_result.counts.items() if k.count("1") % 2 == 1)
            return ones / total

        zne_prob = mitigator.mitigate(_prob_one_at_scale)
        run_result = run_sampler(measured, resolved.descriptor, resolved.backend, shots=shots)
        counts = run_result.counts
    elif mitigation == "readout_m3":
        run_result = run_sampler(measured, resolved.descriptor, resolved.backend, shots=shots)
        physical_qubits = _physical_qubits_measured(measured)
        m3 = M3Mitigator(resolved.backend, physical_qubits)
        m3_result = m3.apply(run_result.counts)
        counts = {k: v for k, v in m3_result.mitigated_quasi_probabilities.items()}
    else:
        run_result = run_sampler(measured, resolved.descriptor, resolved.backend, shots=shots)
        counts = run_result.counts

    return ExecutionResult(
        counts=counts, backend_name=resolved.descriptor.name, mitigation=mitigation, shots=shots,
        zne_extrapolated_probability=zne_prob,
    )
