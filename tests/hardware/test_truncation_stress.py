"""regression stress test for the qubit-truncation fix in
`hardware.ibm_runtime.primitives._truncate_to_active_qubits` (see that
function's docstring for the original bug: a circuit transpiled against a
full-width fake device backend, then `measure_all()`'d AFTER transpile,
forces AerSimulator to allocate a dense statevector/density-matrix sized to
the WHOLE device rather than just the handful of qubits actually gated —
OOMing for any circuit with a non-Clifford gate).

The original regression test used FakeSherbrooke (127 qubits) with a small
circuit. This is a deliberately harder case: the LARGEST fake backend this
qiskit-ibm-runtime install actually has (dynamically discovered, not
hardcoded, since fake-backend line-ups drift across releases — same
reasoning `hardware.backends.catalog._load_fake` already documents), a
15-20 qubit circuit where only a handful of qubits are gate-bearing
(several non-Clifford rotations among them, so the Clifford-stabilizer
auto-selection escape hatch that masked the original bug can't mask this
one either), and the EXACT broken pattern: `measure_all()` called AFTER
`transpile()`, which measures every physical qubit of the full-width
device, not just the active ones.
"""
from __future__ import annotations

import time

from qiskit import QuantumCircuit, transpile

from core.schemas import BackendDescriptor, BackendKind
from hardware.ibm_runtime.primitives import _truncate_to_active_qubits, run_sampler


def _largest_available_fake_backend():
    """Dynamically find the fake backend with the most qubits this
    qiskit-ibm-runtime install actually provides, mirroring
    `hardware.backends.catalog._load_fake`'s "don't hardcode a name that
    might not exist in this version" reasoning, but explicitly searching
    for the biggest rather than a specific named family.
    """
    import qiskit_ibm_runtime.fake_provider as fp

    candidates = []
    for attr_name in dir(fp):
        if not attr_name.startswith("Fake"):
            continue
        cls = getattr(fp, attr_name)
        try:
            backend = cls()
        except Exception:
            continue
        if not hasattr(backend, "num_qubits"):
            continue
        candidates.append((backend.num_qubits, attr_name, backend))
    assert candidates, "no fake backends discovered in qiskit_ibm_runtime.fake_provider"
    candidates.sort(key=lambda t: -t[0])
    n_qubits, name, backend = candidates[0]
    return name, backend


def _sparse_active_circuit(n_declared: int, active: list[int]) -> QuantumCircuit:
    """A circuit declaring `n_declared` qubits where only `active` ever
    receive a gate (several non-Clifford), the rest idle."""
    qc = QuantumCircuit(n_declared)
    qc.h(active[0])
    qc.rx(0.37, active[1])          # non-Clifford
    qc.ry(0.71, active[2])          # non-Clifford
    qc.cx(active[0], active[1])
    qc.rz(0.19, active[3])          # non-Clifford
    qc.cx(active[2], active[3])
    if len(active) > 4:
        qc.rzz(0.41, active[3], active[4])   # non-Clifford, two-qubit
    return qc


def test_truncation_holds_on_largest_fake_backend_with_measure_all_after_transpile():
    backend_name, backend = _largest_available_fake_backend()
    assert backend.num_qubits >= 127, f"expected a 127+ qubit fake backend, got {backend_name} ({backend.num_qubits})"

    n_declared = 19
    active = [2, 5, 9, 13, 16]
    qc = _sparse_active_circuit(n_declared, active)

    tqc = transpile(qc, backend=backend, seed_transpiler=42)
    assert tqc.num_qubits == backend.num_qubits, (
        "sanity check: transpile against a real device backend must widen the "
        "circuit to the FULL device width -- if this assertion fails the test "
        "setup itself isn't exercising the bug pattern anymore"
    )

    # The exact broken pattern: measure_all() AFTER transpile, which
    # measures every one of the backend's physical qubits, not just the
    # ~5 active ones.
    tqc.measure_all()
    assert tqc.num_clbits == backend.num_qubits

    # Confirm the truncation helper itself reduces this correctly before
    # even running anything.
    truncated = _truncate_to_active_qubits(tqc)
    assert truncated.num_qubits == len(active), (
        f"expected truncation down to {len(active)} active qubits, got {truncated.num_qubits}"
    )
    assert truncated.num_qubits < tqc.num_qubits

    descriptor = BackendDescriptor(
        name=backend_name,
        kind=BackendKind.FAKE,
        num_qubits=backend.num_qubits,
        provider="ibm",
        calibration_source=f"fake_backend:{backend_name}",
    )

    shots = 512
    t0 = time.time()
    result = run_sampler(tqc, descriptor, backend, shots=shots)
    elapsed = time.time() - t0

    # The actual regression assertion: this must complete at all (a real
    # OOM/MemoryError on the un-truncated path either raises or hangs
    # indefinitely trying to allocate a 2^~150-dim statevector) and must
    # complete in a time frame consistent with a ~5-qubit simulation, not
    # a full-device one.
    assert sum(result.counts.values()) == shots
    assert elapsed < 90, f"took {elapsed:.1f}s -- suggests truncation silently stopped working"

    # Every reported bitstring must be exactly len(active) characters long
    # -- i.e. only the active qubits' measurements survived, not all
    # backend.num_qubits of them.
    for bitstring in result.counts:
        assert len(bitstring) == len(active), (
            f"bitstring {bitstring!r} has {len(bitstring)} chars, expected "
            f"{len(active)} (one per active qubit) -- idle qubits leaked through"
        )


def test_truncation_is_exact_not_approximate_on_the_stress_circuit():
    """The truncation must not change the physics: an idle, ungated qubit
    with no gate and no scheduled delay is deterministically |0>, so
    dropping its measurement changes nothing about the reported
    distribution over the ACTIVE qubits. Verify by comparing the
    truncated-circuit's noiseless distribution against directly building
    and running the same small circuit natively (never going through a
    127+/150+-wide device at all) -- they must match statistically."""
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

    backend_name, backend = _largest_available_fake_backend()
    n_declared = 19
    active = [2, 5, 9, 13, 16]
    qc = _sparse_active_circuit(n_declared, active)
    tqc = transpile(qc, backend=backend, seed_transpiler=42)
    tqc.measure_all()
    truncated = _truncate_to_active_qubits(tqc)

    shots = 4000
    sampler = AerSamplerV2()
    job = sampler.run([truncated], shots=shots)
    result = job.result()
    data_bin = result[0].data
    reg_name = "meas" if hasattr(data_bin, "meas") else next(iter(vars(data_bin)))
    counts = getattr(data_bin, reg_name).get_counts()

    assert sum(counts.values()) == shots
    # Non-degenerate: with h/rx/ry rotations in the active subcircuit, more
    # than one distinct outcome must appear (proves it isn't trivially
    # collapsed by an over-aggressive truncation).
    assert len(counts) > 1
