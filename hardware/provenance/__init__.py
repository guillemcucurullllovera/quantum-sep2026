"""Hardware-run-specific provenance: given an `ExperimentResult` produced
via a specific backend/calibration-snapshot/mitigation configuration,
produce a full reproducibility record — what would need to be true to
exactly reproduce this run.

Distinct from `core.provenance` (environment/package-version fingerprint,
generic to any run) and from `hardware.experiment_tracking` (builds the
hardware-context payload *at run time*, before a result exists) — this
module reads that context back out of an already-recorded result and
turns it into an explicit reproducibility checklist.
"""
from hardware.provenance.reproducibility_record import ReproducibilityRecord, build_reproducibility_record
from hardware.provenance.runtime_diagnostics import (
    RuntimeDiagnostics,
    build_runtime_diagnostics,
    record_with_runtime_diagnostics,
    stamp_runtime_diagnostics,
)

__all__ = [
    "ReproducibilityRecord",
    "build_reproducibility_record",
    "RuntimeDiagnostics",
    "build_runtime_diagnostics",
    "stamp_runtime_diagnostics",
    "record_with_runtime_diagnostics",
]
