"""Cross-cloud backend fallback chain: if the caller's preferred
backend/cloud is unavailable — missing credentials, an SDK import error, an
unregistered/unknown backend name — fall through a documented, loud chain
(by default: IBM fake backend -> Braket local simulator -> Aer plain
simulator) rather than either raising immediately or, worse, silently
substituting a different backend without saying so.

Every attempt (success or failure) is logged via `logger.warning` in real
time AND returned in `FallbackOutcome.attempts`, and
`FallbackOutcome.fell_back` is explicit — this is operational resilience,
not a silent wrong-answer fallback. See `hardware.provenance.
runtime_diagnostics` for wiring a `FallbackOutcome` into
`core.experiment_registry` so a recorded experiment always shows which
backend/cloud actually ran a given circuit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from core.errors import QuantumSep2026Error
from hardware.backend_selection.capabilities import (
    BackendCapabilities,
    capabilities_from_braket_local_simulator,
    capabilities_from_ibm_backend,
)
from hardware.backends.catalog import BackendCatalog

logger = logging.getLogger(__name__)


class AllBackendsUnavailableError(QuantumSep2026Error):
    """Raised when every step of the fallback chain failed. A loud, typed
    failure — never a silent no-op — carrying every attempt's error."""


@dataclass
class ResolvedTarget:
    """What resolving one fallback-chain step returns: a handle plus its
    normalized capabilities, so callers get both "the thing to run on" and
    "what it can do" together."""

    step_name: str
    cloud: str
    capabilities: BackendCapabilities
    handle: object  # ResolvedBackend (IBM) or braket.devices.LocalSimulator (AWS)


@dataclass
class FallbackAttempt:
    step: str
    ok: bool
    error: str | None = None


@dataclass
class FallbackOutcome:
    requested: str
    actual: ResolvedTarget
    fell_back: bool
    attempts: list[FallbackAttempt] = field(default_factory=list)


def _resolve_ibm(name: str) -> ResolvedTarget:
    resolved = BackendCatalog.get(name)
    return ResolvedTarget(
        step_name=resolved.descriptor.name,
        cloud="ibm_quantum",
        capabilities=capabilities_from_ibm_backend(resolved),
        handle=resolved,
    )


def _resolve_braket_local() -> ResolvedTarget:
    # Import here, not at module load, so an amazon-braket-sdk import error
    # (the documented "Braket unavailable" case) is caught by the fallback
    # logic below rather than surfacing at import time of this module.
    from braket.devices import LocalSimulator

    device = LocalSimulator()
    return ResolvedTarget(
        step_name="braket_local_simulator",
        cloud="aws_braket_local",
        capabilities=capabilities_from_braket_local_simulator(),
        handle=device,
    )


def _resolve_aer_plain() -> ResolvedTarget:
    resolved = BackendCatalog.get("noiseless")
    return ResolvedTarget(
        step_name=resolved.descriptor.name,
        cloud="local",
        capabilities=capabilities_from_ibm_backend(resolved),
        handle=resolved,
    )


# name -> resolver. These three are the documented, fixed operational
# fallback order; `requested` (see `resolve_with_fallback`) is tried first
# and is looked up in this same map, so any catalog/braket_local/noiseless
# name is a valid `requested` value too.
_DEFAULT_STEP_RESOLVERS: dict[str, Callable[[], ResolvedTarget]] = {
    "fake_heron": lambda: _resolve_ibm("fake_heron"),
    "fake_eagle": lambda: _resolve_ibm("fake_eagle"),
    "braket_local": _resolve_braket_local,
    "noiseless": _resolve_aer_plain,
}

DEFAULT_FALLBACK_ORDER: tuple[str, ...] = ("fake_heron", "braket_local", "noiseless")


def resolve_with_fallback(
    requested: str,
    *,
    fallback_order: tuple[str, ...] = DEFAULT_FALLBACK_ORDER,
    resolvers: dict[str, Callable[[], ResolvedTarget]] | None = None,
) -> FallbackOutcome:
    """Resolve `requested`, falling through `fallback_order` on failure.

    `resolvers` lets callers (and tests) inject a resolver map — e.g. to
    simulate "IBM Quantum unavailable: no credentials" or "Braket SDK not
    installed" without actually breaking the real dependency — defaults to
    the real, documented resolvers (`_DEFAULT_STEP_RESOLVERS`).

    Never silently substitutes: every attempt (success or failure) is
    logged via `logger.warning` as it happens and appended to
    `FallbackOutcome.attempts`; `FallbackOutcome.fell_back` is True whenever
    the backend that actually ran is not the one requested.

    Raises `AllBackendsUnavailableError` (never returns a None/degenerate
    result) if every step — requested plus the entire fallback order —
    fails.
    """
    resolvers = resolvers or _DEFAULT_STEP_RESOLVERS
    # De-duplicate while preserving order: requested first, then the chain,
    # skipping the requested step if it also appears later in the chain.
    order = [requested] + [s for s in fallback_order if s != requested]

    attempts: list[FallbackAttempt] = []
    for step in order:
        resolver = resolvers.get(step)
        if resolver is None:
            err = f"no resolver registered for backend step {step!r}"
            attempts.append(FallbackAttempt(step=step, ok=False, error=err))
            logger.warning("backend step %r unavailable (%s); trying next fallback step", step, err)
            continue
        try:
            target = resolver()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # import error / credential error / unavailable-backend error
            # from ANY registered resolver must fall through, not just one
            # specific known exception type.
            attempts.append(FallbackAttempt(step=step, ok=False, error=f"{type(exc).__name__}: {exc}"))
            logger.warning(
                "backend step %r unavailable (%s: %s); trying next fallback step",
                step, type(exc).__name__, exc,
            )
            continue

        attempts.append(FallbackAttempt(step=step, ok=True))
        fell_back = step != requested
        if fell_back:
            logger.warning(
                "requested backend %r unavailable; fell back to %r (cloud=%s)",
                requested, target.step_name, target.cloud,
            )
        return FallbackOutcome(requested=requested, actual=target, fell_back=fell_back, attempts=attempts)

    raise AllBackendsUnavailableError(
        f"Every backend in the fallback chain failed for requested={requested!r}: "
        f"{[(a.step, a.error) for a in attempts]}"
    )


__all__ = [
    "ResolvedTarget",
    "FallbackAttempt",
    "FallbackOutcome",
    "AllBackendsUnavailableError",
    "DEFAULT_FALLBACK_ORDER",
    "resolve_with_fallback",
]
