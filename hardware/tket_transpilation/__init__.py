"""Qiskit <-> tket circuit bridge and a third, architecturally distinct
transpilation strategy alongside two Qiskit-native ones
(hardware.transpilation.TranspilationStrategy).

tket (Quantinuum, Apache-2.0) is not vendored as source here -- its core is
C++ and ships only as a compiled wheel (`pytket`, installed as a normal
Python dependency, the same way this platform already depends on `qiskit`
itself rather than vendoring Qiskit's C++/Rust extensions). What IS
genuinely new here is the integration: a real circuit round-trip between
Qiskit's and tket's circuit representations, and a real depth/2Q-gate-count
comparison against Qiskit-only strategies, using tket's pass
composition architecture -- a design genuinely different from Qiskit's
PassManager (declarative pass sequencing vs. the FullPeepholeOptimise-style
combinator passes tket favors).
"""
from __future__ import annotations

from dataclasses import dataclass

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise, SequencePass
from pytket.predicates import CompilationUnit
from qiskit import QuantumCircuit

from core.errors import ValidationError


@dataclass
class TketTranspileResult:
    circuit: QuantumCircuit
    depth: int
    gate_counts: dict[str, int]
    two_qubit_gate_count: int


_TWO_QUBIT_GATE_NAMES = ("cx", "cz", "ecr", "cy", "swap", "rzz", "rzx", "cnot")


def qiskit_roundtrip(circuit: QuantumCircuit) -> QuantumCircuit:
    """Qiskit -> tket -> Qiskit, with no optimization applied.

    Used to validate the bridge itself is faithful before trusting any
    comparison built on top of it.
    """
    tk_circuit = qiskit_to_tk(circuit)
    return tk_to_qiskit(tk_circuit)


def transpile_via_tket(circuit: QuantumCircuit) -> TketTranspileResult:
    """Optimize `circuit` using tket's FullPeepholeOptimise pass sequence,
    then convert back to Qiskit for uniform downstream analysis.
    """
    if circuit.num_qubits == 0:
        raise ValidationError("Cannot transpile a zero-qubit circuit")

    tk_circuit = qiskit_to_tk(circuit)
    unit = CompilationUnit(tk_circuit)
    SequencePass([FullPeepholeOptimise()]).apply(unit)
    optimized = unit.circuit

    result_circuit = tk_to_qiskit(optimized)
    gate_counts: dict[str, int] = {}
    for instr in result_circuit.data:
        name = instr.operation.name
        gate_counts[name] = gate_counts.get(name, 0) + 1
    two_q = sum(n for g, n in gate_counts.items() if g in _TWO_QUBIT_GATE_NAMES)

    return TketTranspileResult(
        circuit=result_circuit,
        depth=result_circuit.depth(),
        gate_counts=gate_counts,
        two_qubit_gate_count=two_q,
    )


def compare_three_strategies(circuit: QuantumCircuit, backend) -> dict[str, dict]:
    """Depth / 2Q-gate-count comparison across two Qiskit-native
    strategies plus tket's FullPeepholeOptimise -- three architecturally
    distinct transpilers on the identical input circuit. `backend` is any
    object transpile_with_strategy accepts (e.g. a fake backend
    from hardware.fake_backends).
    """
    from hardware.transpilation.strategies import TranspilationStrategy, transpile_with_strategy

    out: dict[str, dict] = {}
    for strategy in TranspilationStrategy:
        r = transpile_with_strategy(circuit, backend, strategy)
        out[strategy.value] = {"depth": r.depth, "two_qubit_gate_count": r.two_qubit_gate_count}

    tket_r = transpile_via_tket(circuit)
    out["tket_full_peephole"] = {
        "depth": tket_r.depth,
        "two_qubit_gate_count": tket_r.two_qubit_gate_count,
    }
    return out
