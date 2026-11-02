"""Pre-flight circuit/backend compatibility checks: "will this circuit run
on this backend" answered BEFORE any execution attempt, using the
cloud-agnostic `hardware.backend_selection.capabilities.BackendCapabilities`
schema — so an incompatible pairing fails loudly and immediately with a
specific, actionable reason, rather than failing deep inside a simulator
call (a memory-allocation error, an `UnsupportedGateError` raised mid-
translation, or a job silently truncated) with no context about *why*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.errors import QuantumSep2026Error
from hardware.backend_selection.capabilities import BackendCapabilities


class IncompatibleCircuitError(QuantumSep2026Error):
    """Raised by `require_compatible` when a circuit cannot run on a given
    backend's capabilities. Carries the full `CompatibilityReport` so
    callers/logs get every reason at once, not just the first."""

    def __init__(self, report: "CompatibilityReport") -> None:
        self.report = report
        reasons = "; ".join(report.reasons)
        super().__init__(
            f"Circuit is not compatible with backend '{report.backend_name}': {reasons}"
        )


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    backend_name: str
    reasons: list[str] = field(default_factory=list)


def _circuit_gate_names(circuit) -> set[str]:
    return {instr.operation.name for instr in circuit.data}


def _has_mid_circuit_control_flow(circuit) -> bool:
    """True if the circuit contains a classically-conditioned/control-flow
    instruction (c_if, IfElseOp, WhileLoopOp, ...) that requires genuine
    mid-circuit measurement/conditioning support to execute correctly."""
    control_flow_names = {"if_else", "while_loop", "for_loop", "switch_case"}
    for instr in circuit.data:
        op = instr.operation
        if op.name in control_flow_names:
            return True
        condition = getattr(op, "condition", None)
        if condition is not None:
            return True
    return False


def check_compatibility(
    circuit,
    capabilities: BackendCapabilities,
    *,
    shots: int | None = None,
) -> CompatibilityReport:
    """Check `circuit` against `capabilities` and return a full report
    (never raises) — every failing dimension is checked and collected, not
    just the first one found, so a caller/log sees the whole picture."""
    reasons: list[str] = []

    if circuit.num_qubits > capabilities.num_qubits:
        reasons.append(
            f"circuit needs {circuit.num_qubits} qubits but backend "
            f"'{capabilities.name}' has only {capabilities.num_qubits}"
        )

    if capabilities.native_gates is not None:
        used = _circuit_gate_names(circuit)
        unsupported = sorted(used - capabilities.native_gates)
        if unsupported:
            reasons.append(
                f"gate(s) {unsupported} are not in backend '{capabilities.name}'s "
                f"translatable vocabulary ({sorted(capabilities.native_gates)})"
            )

    if not capabilities.supports_mid_circuit_measurement and _has_mid_circuit_control_flow(circuit):
        reasons.append(
            f"circuit uses classically-conditioned/mid-circuit control flow, "
            f"but backend '{capabilities.name}' does not support mid-circuit measurement"
        )

    if shots is not None and capabilities.max_shots is not None and shots > capabilities.max_shots:
        reasons.append(
            f"requested shots={shots} exceeds backend '{capabilities.name}'s "
            f"max_shots={capabilities.max_shots}"
        )

    return CompatibilityReport(
        compatible=len(reasons) == 0,
        backend_name=capabilities.name,
        reasons=reasons,
    )


def require_compatible(circuit, capabilities: BackendCapabilities, *, shots: int | None = None) -> CompatibilityReport:
    """Same check as `check_compatibility`, but raises `IncompatibleCircuitError`
    (carrying the full report) instead of returning a `compatible=False`
    report — the "fail loudly before execution" entry point callers should
    use right before dispatching to a real run_sampler/run_estimator/run_local
    call."""
    report = check_compatibility(circuit, capabilities, shots=shots)
    if not report.compatible:
        raise IncompatibleCircuitError(report)
    return report


__all__ = [
    "CompatibilityReport",
    "IncompatibleCircuitError",
    "check_compatibility",
    "require_compatible",
]
