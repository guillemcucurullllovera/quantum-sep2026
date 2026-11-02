"""Machine-checkable Hardware Day authorization gate.

**Remediates independent-review findings P2-3 and P2-4.**

P2-3 ("Hardware-day authorization is entirely procedural (a markdown
checklist), with no code-level gate tying real-service execution to the
capstone's decision"): no function anywhere parsed the capstone's decision
and refused to proceed if it wasn't `HARDWARE_DAY_READY` -- authorization
relied entirely on a human operator reading and self-certifying
`QUANTUM1B_HARDWARE_DAY_RUNBOOK.md`'s pre-flight checklist. This module is
that missing code-level gate: `assert_hardware_day_authorized()` raises
`HardwareDayNotAuthorizedError` unless every gate
`readiness_state.current_lab_wide_gates()` computes from REAL, LIVE
repository state is satisfied.

P2-4 ("`credentials_present` is a trust-me boolean never derived from real
credential state"): `current_lab_wide_gates()`'s
`real_ibm_credentials_present` is derived from
`core.config.get_settings().ibm.is_present` (an environment-variable
presence check, never a hand-typed literal) -- this gate reuses that
derivation rather than accepting a caller-supplied boolean, closing the
exact gap the runbook's own worked example demonstrated (hardcoding
`credentials_present=True` rather than deriving it).

**Any future real-hardware entry point should call
`assert_hardware_day_authorized()` before constructing a real
`QiskitRuntimeService` or calling `hardware.qpu_execution.execute_on_qpu`.**
This module does not itself call or wrap `execute_on_qpu` -- that function
lives in the general-purpose `hardware/` package (outside this session's
`experiments/quantum1b/`/`tests/quantum1b/`/`docs/quantum/` scope boundary,
see `QUANTUM1B_ESTATE_INVENTORY.md`), and a QUANTUM-1B-specific capstone
gate does not belong baked into shared, multi-workstream infrastructure.
Wiring this gate into an actual entry point is the responsibility of
whoever writes that entry point, per the runbook.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.quantum1b.quantum_lab.readiness_state import current_lab_wide_gates

_REQUIRED_GATES: tuple[str, ...] = (
    "real_ibm_credentials_present",
    "capstone_decision_is_hardware_day_ready",
    "independent_review_signed_off",
)


class HardwareDayNotAuthorizedError(Exception):
    """Raised by `assert_hardware_day_authorized()` when any required gate
    is unmet. Never silently returns False -- a caller must handle this
    explicitly; there is no "proceed anyway" parameter."""


@dataclass(frozen=True)
class HardwareDayAuthorization:
    authorized: bool
    unmet_gates: tuple[str, ...]
    all_gates: dict[str, bool]


def check_hardware_day_authorization() -> HardwareDayAuthorization:
    """Non-raising check -- returns the full authorization picture rather
    than just a boolean, so a caller (or a Command Center display) can show
    WHICH gate(s) are unmet, not just that authorization failed."""
    gates = current_lab_wide_gates()
    unmet = tuple(name for name in _REQUIRED_GATES if not gates[name])
    return HardwareDayAuthorization(authorized=not unmet, unmet_gates=unmet, all_gates=gates)


def assert_hardware_day_authorized() -> None:
    """Raises `HardwareDayNotAuthorizedError` unless every required gate is
    satisfied. Call this before any real-hardware submission path -- it
    makes the previously-procedural-only authorization a code-level
    refusal, not merely a checklist item a human might skip under time
    pressure."""
    result = check_hardware_day_authorization()
    if not result.authorized:
        raise HardwareDayNotAuthorizedError(
            f"Hardware Day is not authorized -- unmet gate(s): {', '.join(result.unmet_gates)}. "
            "See docs/quantum/QUANTUM1B_HARDWARE_DAY_RUNBOOK.md and "
            "docs/quantum/PRE_HARDWARE_QUANTUM_CAPSTONE.md before proceeding."
        )


__all__ = [
    "HardwareDayAuthorization",
    "HardwareDayNotAuthorizedError",
    "check_hardware_day_authorization",
    "assert_hardware_day_authorized",
]
