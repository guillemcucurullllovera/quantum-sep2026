"""Post-transpile circuit metrics — depth, gate counts, 2-qubit gate count,
critical-path analysis. Ported/extended from Generation 2's
isa_pipeline.circuit_stats.
"""
from __future__ import annotations

from dataclasses import dataclass

_TWO_QUBIT_GATE_NAMES = ("cx", "cz", "ecr", "cy", "swap", "rzz", "rzx", "cnot")


@dataclass
class CircuitStats:
    n_qubits: int
    depth: int
    n_gates: int
    gate_counts: dict[str, int]
    two_qubit_gate_count: int
    two_qubit_depth: int


def analyze(circuit) -> CircuitStats:
    ops = {k: int(v) for k, v in circuit.count_ops().items()}
    two_q_names = [g for g in ops if g in _TWO_QUBIT_GATE_NAMES]
    two_q_depth = circuit.depth(lambda instr: instr.operation.name in _TWO_QUBIT_GATE_NAMES)
    return CircuitStats(
        n_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        n_gates=sum(ops.values()),
        gate_counts=ops,
        two_qubit_gate_count=sum(ops.get(g, 0) for g in two_q_names),
        two_qubit_depth=two_q_depth,
    )


def compare(before, after) -> dict[str, float]:
    """Relative change (after - before) / before for depth and 2q-gate
    count, useful for reporting transpilation-strategy impact."""
    b, a = analyze(before), analyze(after)

    def rel(x_before: int, x_after: int) -> float:
        return (x_after - x_before) / x_before if x_before else float("nan")

    return {
        "depth_before": b.depth,
        "depth_after": a.depth,
        "depth_relative_change": rel(b.depth, a.depth),
        "two_qubit_gates_before": b.two_qubit_gate_count,
        "two_qubit_gates_after": a.two_qubit_gate_count,
        "two_qubit_gates_relative_change": rel(b.two_qubit_gate_count, a.two_qubit_gate_count),
    }


__all__ = ["CircuitStats", "analyze", "compare"]
