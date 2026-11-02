from __future__ import annotations

from pydantic import BaseModel

from core.provenance import environment_fingerprint
from core.schemas import ExperimentResult


class ReproducibilityRecord(BaseModel):
    run_id: str
    algorithm: str
    backend_name: str | None = None
    backend_kind: str | None = None
    calibration_source: str | None = None
    transpilation_strategy: str | None = None
    mitigation_kind: str | None = None
    seed: int | None = None
    environment: dict
    reproducible: bool
    missing_fields: list[str]


_REQUIRED_HARDWARE_FIELDS = ("backend_name", "calibration_source")


def build_reproducibility_record(result: ExperimentResult) -> ReproducibilityRecord:
    """Inspect an already-recorded `ExperimentResult` (whose `config.backend`
    and `notes`/artifacts may carry a `hardware.experiment_tracking.
    hardware_context()` payload) and report exactly what's needed to
    reproduce it, honestly flagging what's missing rather than assuming
    completeness.
    """
    backend = result.config.backend
    env = environment_fingerprint()

    context: dict = {}
    if result.notes:
        try:
            import json

            parsed = json.loads(result.notes)
            if isinstance(parsed, dict):
                context = parsed
        except Exception:
            context = {}

    values = {
        "backend_name": backend.name if backend else context.get("backend_name"),
        "backend_kind": backend.kind.value if backend else context.get("backend_kind"),
        "calibration_source": backend.calibration_source if backend else context.get("calibration_source"),
        "transpilation_strategy": context.get("transpilation_strategy"),
        "mitigation_kind": result.config.mitigation.kind.value if result.config.mitigation else context.get("mitigation_kind"),
        "seed": result.metadata.seed if result.metadata.seed is not None else result.config.seed,
    }

    missing = [f for f in _REQUIRED_HARDWARE_FIELDS if not values.get(f)]

    return ReproducibilityRecord(
        run_id=result.metadata.run_id,
        algorithm=result.metadata.algorithm,
        environment=env,
        reproducible=len(missing) == 0,
        missing_fields=missing,
        **values,
    )
