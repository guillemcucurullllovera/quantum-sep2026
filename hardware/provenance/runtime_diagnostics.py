"""Runtime diagnostics + experiment provenance: capture WHICH
backend/cloud/fallback-path actually ran a given circuit — not just what
was requested — and wire that into `core.experiment_registry` so a run
read back out of the registry months later is self-describing about
whether it silently ran somewhere other than the caller asked for.

Distinct from `hardware.experiment_tracking.hardware_context` (the
static backend/calibration/mitigation description of *how* a run was
configured) and `hardware.provenance.reproducibility_record` (whether a
recorded run has enough of that static description to reproduce) — this
module is specifically about `hardware.backend_selection.fallback`'s
dynamic "what actually happened when we tried to get a backend" outcome.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from core.schemas import ExperimentResult
from hardware.backend_selection.fallback import FallbackOutcome


class RuntimeDiagnostics(BaseModel):
    requested_backend: str
    actual_backend: str
    actual_cloud: str
    fell_back: bool
    attempts: list[dict] = Field(
        default_factory=list,
        description="One entry per fallback-chain step tried, in order: "
        "{'step': str, 'ok': bool, 'error': str | None}.",
    )


def build_runtime_diagnostics(outcome: FallbackOutcome) -> RuntimeDiagnostics:
    """Turn a `hardware.backend_selection.fallback.FallbackOutcome` into
    the serializable `RuntimeDiagnostics` record."""
    return RuntimeDiagnostics(
        requested_backend=outcome.requested,
        actual_backend=outcome.actual.step_name,
        actual_cloud=outcome.actual.cloud,
        fell_back=outcome.fell_back,
        attempts=[{"step": a.step, "ok": a.ok, "error": a.error} for a in outcome.attempts],
    )


def stamp_runtime_diagnostics(result: ExperimentResult, diagnostics: RuntimeDiagnostics) -> ExperimentResult:
    """Merge `diagnostics` into `result.notes`, under a `runtime_diagnostics`
    key, preserving whatever JSON payload (e.g. from
    `hardware.experiment_tracking.hardware_context`) is already there —
    the same `notes`-as-JSON-dict convention
    `hardware.provenance.reproducibility_record.build_reproducibility_record`
    already reads. `ExperimentResult` is a pydantic model treated as
    immutable here: returns a new instance, does not mutate `result`.
    """
    existing: dict = {}
    if result.notes:
        try:
            parsed = json.loads(result.notes)
            if isinstance(parsed, dict):
                existing = parsed
        except Exception:
            existing = {}
    existing["runtime_diagnostics"] = diagnostics.model_dump()
    return result.model_copy(update={"notes": json.dumps(existing)})


def record_with_runtime_diagnostics(
    result: ExperimentResult,
    outcome: FallbackOutcome,
    registry=None,
) -> ExperimentResult:
    """Stamp `outcome` onto `result.notes` (via `stamp_runtime_diagnostics`)
    and persist the stamped result through
    `core.experiment_registry.ExperimentRegistry` — the "wired into
    core.experiment_registry" requirement: after this call,
    `ExperimentRegistry.get(run_id)` returns a result whose `notes` reveal
    exactly which backend/cloud actually ran it (and every fallback step
    tried along the way), not just what was originally requested.

    `registry` defaults to a fresh `ExperimentRegistry()` (the platform's
    default on-disk registry) if not given — pass an explicit instance
    (e.g. pointed at a tmp_path db) in tests to avoid touching the real one.
    """
    from core.experiment_registry import ExperimentRegistry

    diagnostics = build_runtime_diagnostics(outcome)
    stamped = stamp_runtime_diagnostics(result, diagnostics)
    reg = registry or ExperimentRegistry()
    reg.record(stamped)
    return stamped


__all__ = [
    "RuntimeDiagnostics",
    "build_runtime_diagnostics",
    "stamp_runtime_diagnostics",
    "record_with_runtime_diagnostics",
]
