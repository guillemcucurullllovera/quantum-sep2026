"""Named transpilation strategies over Qiskit's preset pass managers.

Ported/generalized from Generation 2's isa_pipeline.py (SABRE + ECR-basis
ISA transpilation), extended with a second named strategy so callers can
actually choose between "minimize depth" and "minimize 2-qubit gate count"
rather than only ever using optimization_level=3.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.errors import ValidationError


class TranspilationStrategy(str, Enum):
    MINIMIZE_DEPTH = "minimize_depth"
    MINIMIZE_TWO_QUBIT_GATES = "minimize_two_qubit_gates"


_STRATEGY_PARAMS: dict[TranspilationStrategy, dict] = {
    TranspilationStrategy.MINIMIZE_DEPTH: dict(
        optimization_level=3, layout_method="sabre", routing_method="sabre"
    ),
    TranspilationStrategy.MINIMIZE_TWO_QUBIT_GATES: dict(
        optimization_level=3, layout_method="dense", routing_method="lookahead"
    ),
}


@dataclass
class TranspileResult:
    circuit: object
    strategy: TranspilationStrategy
    depth: int
    gate_counts: dict[str, int]
    two_qubit_gate_count: int


_TWO_QUBIT_GATE_NAMES = ("cx", "cz", "ecr", "cy", "swap", "rzz", "rzx", "cnot")


def transpile_with_strategy(
    circuit,
    backend,
    strategy: TranspilationStrategy,
    initial_layout: list[int] | None = None,
    seed: int = 42,
) -> TranspileResult:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if circuit.num_qubits > backend.num_qubits:
        raise ValidationError(
            f"Circuit needs {circuit.num_qubits} qubits, backend {getattr(backend, 'name', backend)} "
            f"has {backend.num_qubits}."
        )
    params = _STRATEGY_PARAMS[strategy]
    pm = generate_preset_pass_manager(
        backend=backend, initial_layout=initial_layout, seed_transpiler=seed, **params
    )
    transpiled = pm.run(circuit)
    gate_counts = {k: int(v) for k, v in transpiled.count_ops().items()}
    two_q = sum(v for k, v in gate_counts.items() if k in _TWO_QUBIT_GATE_NAMES)
    return TranspileResult(
        circuit=transpiled,
        strategy=strategy,
        depth=transpiled.depth(),
        gate_counts=gate_counts,
        two_qubit_gate_count=two_q,
    )


def compare_strategies(circuit, backend, seed: int = 42) -> dict[str, TranspileResult]:
    """Run every known strategy on the same circuit+backend, so a caller can
    pick empirically rather than guessing which one is better for a given
    circuit shape."""
    return {
        strategy.value: transpile_with_strategy(circuit, backend, strategy, seed=seed)
        for strategy in TranspilationStrategy
    }


__all__ = ["TranspilationStrategy", "TranspileResult", "transpile_with_strategy", "compare_strategies"]
