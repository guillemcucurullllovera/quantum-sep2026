"""Simulator/fake-backend wiring for the QAE pipeline. Real IBM hardware access goes
through core.config's IBMQuantumCredentials — see `iqae.run_on_hardware`.

Kept self-contained in this package for now rather than importing a shared hardware/
package, since hardware/ hasn't been built out yet by the platform's hardware
workstream — expect this to be superseded by hardware.backends once that lands
(duplication accepted deliberately currently, see repo-level build notes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def ideal_sim():
    from qiskit_aer import AerSimulator
    return AerSimulator()


def kingston_backend():
    """FakeKingston calibration-backed backend (IBM Kingston, Heron r3) — a real device
    calibration snapshot shipped with qiskit-ibm-runtime, no credentials required."""
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def kingston_sim():
    """Aer simulator carrying FakeKingston's noise model + coupling map."""
    from qiskit_aer import AerSimulator
    return AerSimulator.from_backend(kingston_backend())


@dataclass(frozen=True)
class TranspiledCircuit:
    circuit: object
    depth: int
    gate_counts: Dict[str, int]
    twoq_gates: int
    measurement_mapping: Dict[int, int]
    ancilla_clbit: int = 2


def _twoq_count(gate_counts: Dict[str, int]) -> int:
    twoq = ("cx", "cz", "ecr", "cy", "swap", "rzz", "rzx")
    return sum(v for g, v in gate_counts.items() if g in twoq)


def transpile_for(circuit, backend, *, optimization_level: int = 0,
                   initial_layout: Optional[List[int]] = None, seed: int = 42) -> TranspiledCircuit:
    """Transpile to a backend's ISA (opt_level=0 by default - prevents gate-fusion
    cancellation that would corrupt the QAE reflection structure)."""
    from qiskit import transpile
    from mthree.utils import final_measurement_mapping

    t = transpile(circuit, backend=backend, optimization_level=optimization_level,
                  initial_layout=initial_layout, seed_transpiler=seed)
    gc = {k: int(v) for k, v in t.count_ops().items()}
    mapping = final_measurement_mapping(t)
    return TranspiledCircuit(
        circuit=t, depth=int(t.depth()), gate_counts=gc, twoq_gates=_twoq_count(gc),
        measurement_mapping={int(k): int(v) for k, v in mapping.items()},
    )


def run_counts(backend, circuit, shots: int = 4096, *, transpiled: bool = False,
               optimization_level: int = 0, seed: int = 42) -> Dict[str, int]:
    from qiskit import transpile
    circ = circuit
    if not transpiled:
        circ = transpile(circuit, backend=backend, optimization_level=optimization_level,
                         seed_transpiler=seed)
    result = backend.run(circ, shots=shots).result()
    return result.get_counts()


def ancilla_p1(counts: Dict[str, int], ancilla_clbit: int = 2) -> Tuple[int, int]:
    """Return (successes, shots): shots with the ancilla clbit == 1."""
    total = 0
    ones = 0
    for b, c in counts.items():
        s = b.replace(" ", "")
        total += c
        idx = len(s) - 1 - ancilla_clbit
        if 0 <= idx < len(s) and s[idx] == "1":
            ones += c
    return ones, total
