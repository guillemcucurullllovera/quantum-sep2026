"""Extend `hardware.result_processing` (`normalize_counts`,
`to_qubit_marginals`, `to_experiment_metrics`) to Braket result objects,
not just Qiskit/IBM counts — one normalized `NormalizedResult` schema
regardless of which cloud/SDK produced the shots.

Two real differences from the Qiskit/IBM side this module has to bridge:

1. Bit-order convention: Braket's `measurement_counts` bitstrings are
   ordered qubit-0-first (leftmost char = qubit 0, i.e. big-endian), while
   this platform's `hardware.result_processing.normalize_counts` /
   `to_qubit_marginals` are pinned to Qiskit's native convention (leftmost
   char = the *highest*-indexed qubit, i.e. little-endian) — see
   `hardware.result_processing`'s module docstring for where that
   convention was originally pinned down. Reversing the bitstring converts
   between the two.

2. Sparse qubit sets: `hardware.braket.backend_adapter.qiskit_to_braket`
   only ever adds a qubit to the translated Braket circuit if a real gate
   touches it (Braket has no concept of a "declared but idle" qubit the
   way `qiskit.QuantumCircuit(n)` does) — so `measurement_counts`
   bitstrings from a circuit with idle qubits are *shorter* than the
   original circuit's `n_qubits`, not zero-padded. `run_local`'s
   `measured_qubits` list says which original qubit index each bitstring
   position belongs to (see its docstring); this module uses that to
   rebuild a full-width bitstring with idle qubits deterministically '0'
   — exactly the same "idle, ungated qubit is deterministically |0>"
   reasoning `hardware.ibm_runtime.primitives._truncate_to_active_qubits`
   relies on for the IBM/Aer side.
"""
from __future__ import annotations

from hardware.result_processing import NormalizedResult, normalize_counts


def braket_counts_to_qiskit_convention(
    measurement_counts: dict[str, int],
    *,
    n_qubits: int,
    measured_qubits: list[int] | None = None,
) -> dict[str, int]:
    """Convert a Braket `measurement_counts` dict (as returned in
    `hardware.braket.backend_adapter.run_local`'s shots>0 output, or read
    directly off a raw Braket SDK result via `.measurement_counts`) into
    this platform's standard Qiskit-native little-endian bitstring
    convention, at full `n_qubits` width with idle qubits as '0'.

    `measured_qubits` defaults to `range(n_qubits)` (the dense case, where
    every declared qubit was actually gated) if not given — pass the
    `measured_qubits` value `run_local` returns whenever the source
    circuit might have idle qubits, or results will silently mismap.
    """
    if not measurement_counts:
        return {}
    sample_len = len(next(iter(measurement_counts)))
    if measured_qubits is None:
        if sample_len != n_qubits:
            raise ValueError(
                f"braket_counts_to_qiskit_convention: bitstring length {sample_len} != "
                f"n_qubits={n_qubits} and no `measured_qubits` mapping was given -- pass "
                f"run_local()'s 'measured_qubits' value for a circuit with idle qubits."
            )
        measured_qubits = list(range(n_qubits))
    elif len(measured_qubits) != sample_len:
        raise ValueError(
            f"braket_counts_to_qiskit_convention: measured_qubits has {len(measured_qubits)} "
            f"entries but bitstrings are {sample_len} characters long."
        )

    converted: dict[str, int] = {}
    for bitstring, count in measurement_counts.items():
        full = ["0"] * n_qubits
        for pos, qubit in enumerate(measured_qubits):
            # Qiskit little-endian: qubit q sits at string index n-1-q.
            full[n_qubits - 1 - qubit] = bitstring[pos]
        key = "".join(full)
        converted[key] = converted.get(key, 0) + count
    return converted


def normalize_braket_local_result(braket_out: dict, backend_name: str = "braket_local_simulator") -> NormalizedResult:
    """Normalize the dict `hardware.braket.backend_adapter.run_local`
    returns (shots>0 form: must contain `measurement_counts`) into the same
    `NormalizedResult` schema `hardware.ibm_runtime.primitives`-sourced
    counts produce, via `hardware.result_processing.normalize_counts` --
    so `to_qubit_marginals`/`to_experiment_metrics` work identically
    regardless of source.
    """
    if "measurement_counts" not in braket_out:
        raise ValueError(
            "normalize_braket_local_result requires shots>0 run_local() output "
            "(a 'measurement_counts' key); got analytic/state_vector (shots=0) output instead."
        )
    converted = braket_counts_to_qiskit_convention(
        braket_out["measurement_counts"],
        n_qubits=braket_out["n_qubits"],
        measured_qubits=braket_out.get("measured_qubits"),
    )
    return normalize_counts(converted, backend_name)


def normalize_braket_task_result(task_result, n_qubits: int, backend_name: str = "braket") -> NormalizedResult:
    """Normalize a raw Braket SDK result object directly (whatever
    `braket.tasks.GateModelQuantumTaskResult` a real `device.run(...).result()`
    call returns — real QPU/managed-simulator or local — not just this
    platform's `run_local` wrapper dict), so any Braket result source, not
    only the local-simulator convenience path, feeds the same normalized
    schema.
    """
    counts = dict(task_result.measurement_counts)
    measured_qubits = [int(q) for q in getattr(task_result, "measured_qubits", range(n_qubits))]
    converted = braket_counts_to_qiskit_convention(counts, n_qubits=n_qubits, measured_qubits=measured_qubits)
    return normalize_counts(converted, backend_name)


__all__ = [
    "braket_counts_to_qiskit_convention",
    "normalize_braket_local_result",
    "normalize_braket_task_result",
]
