"""Routing: re-exports `hardware.circuit_analysis.compare` (which already
reports before/after 2-qubit-gate overhead — the direct signal of how much a
router had to add) under `hardware.routing`, plus a genuinely new
`routing_method_sweep` that isolates routing's contribution the same way
`hardware.layout.layout_method_sweep` isolates layout's — holding layout
fixed (dense) and varying only the routing method.
"""
from __future__ import annotations

from hardware.circuit_analysis.metrics import CircuitStats, analyze, compare

__all__ = ["CircuitStats", "analyze", "compare", "routing_method_sweep"]


def routing_method_sweep(circuit, backend, methods: tuple[str, ...] = ("sabre", "stochastic", "lookahead"), seed: int = 42):
    """Hold layout fixed (dense) and vary only the routing method."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    results = {}
    for method in methods:
        try:
            pm = generate_preset_pass_manager(
                backend=backend, optimization_level=3, layout_method="dense",
                routing_method=method, seed_transpiler=seed,
            )
            transpiled = pm.run(circuit)
        except Exception as exc:  # some routing methods may be unavailable/incompatible with a given backend
            results[method] = {"error": str(exc)}
            continue
        ops = {k: int(v) for k, v in transpiled.count_ops().items()}
        results[method] = {
            "depth": transpiled.depth(),
            "two_qubit_gate_count": sum(
                v for k, v in ops.items()
                if k in ("cx", "cz", "ecr", "cy", "swap", "rzz", "rzx", "cnot")
            ),
        }
    return results
