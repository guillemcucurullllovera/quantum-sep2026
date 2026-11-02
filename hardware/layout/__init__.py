"""Layout selection: re-exports `hardware.qubit_selection.QubitSelector` (which
already does real calibration-driven layout scoring) under `hardware.layout`,
plus the genuinely new piece that didn't exist before this module — actually
wiring a calibration-driven layout INTO `hardware.transpilation`'s
`transpile_with_strategy`, so a caller gets a full transpile that specifically
avoids a device's worst-calibrated qubits, rather than leaving Qiskit's
default layout pass (which optimizes structurally, not against T1/error data)
to pick qubits blind to calibration quality.
"""
from __future__ import annotations

from hardware.qubit_selection.selector import QubitScore, QubitSelector
from hardware.transpilation.strategies import (
    TranspilationStrategy,
    TranspileResult,
    transpile_with_strategy,
)

__all__ = [
    "QubitScore",
    "QubitSelector",
    "calibration_aware_transpile",
    "layout_method_sweep",
]


def calibration_aware_transpile(
    circuit, backend, strategy: TranspilationStrategy = TranspilationStrategy.MINIMIZE_DEPTH, seed: int = 42
) -> TranspileResult:
    """Score every qubit's calibration quality, pick the best connected
    n-qubit subgraph for `circuit`, and transpile pinned to that layout —
    the calibration-aware counterpart to calling `transpile_with_strategy`
    with no `initial_layout` (which leaves layout choice to Qiskit's
    structural heuristics alone).
    """
    selector = QubitSelector(backend)
    initial_layout = selector.select_layout(circuit.num_qubits)
    return transpile_with_strategy(circuit, backend, strategy, initial_layout=initial_layout, seed=seed)


def layout_method_sweep(circuit, backend, methods: tuple[str, ...] = ("trivial", "dense", "sabre"), seed: int = 42):
    """Hold routing fixed (sabre) and vary only the layout method, isolating
    layout's specific contribution to depth/2Q-gate count — distinct from
    `transpilation.compare_strategies`, which varies layout+routing together
    as bundled named strategies."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    results = {}
    for method in methods:
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, layout_method=method,
            routing_method="sabre", seed_transpiler=seed,
        )
        transpiled = pm.run(circuit)
        ops = {k: int(v) for k, v in transpiled.count_ops().items()}
        results[method] = {
            "depth": transpiled.depth(),
            "two_qubit_gate_count": sum(
                v for k, v in ops.items()
                if k in ("cx", "cz", "ecr", "cy", "swap", "rzz", "rzx", "cnot")
            ),
        }
    return results
