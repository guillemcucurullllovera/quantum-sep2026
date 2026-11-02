"""A real (never network-calling in this codebase) implementation of the
IBM Runtime hardware-evidence verifier whose checklist
`core.verification.IBMRuntimeVerifier`'s docstring specifies and whose
`.verify()` deliberately raises `NotImplementedError` (`core.verification`
is frozen/shared platform infrastructure per
`docs/quantum/QUANTUM1B_PLAN.md` section 0 -- not modified by this pass).

**What this is.** A concrete `core.verification.HardwareEvidenceVerifier`
implementing every item on that checklist, against a caller-INJECTED
`service` object (duck-typed via `RuntimeServiceLike`/`RuntimeJobLike`
below -- the exact subset of `qiskit_ibm_runtime.RuntimeJobV2`'s public API
this needs, verified directly against the installed
`qiskit-ibm-runtime==0.49.0` by introspection before writing this module,
not assumed from documentation alone: `RuntimeJobV2.job_id`/`.backend`/
`.status`/`.metrics`/`.result` are all real public methods on that class,
and `.status()` returns one of `INITIALIZING`/`QUEUED`/`RUNNING`/
`CANCELLED`/`DONE`/`ERROR`, with `DONE` the one terminal-success value;
`.metrics()` returns a dict with a `timestamps` sub-dict including a
`finished` entry).

**The credential/network boundary, deliberate and total.** This module
never imports `qiskit_ibm_runtime.QiskitRuntimeService`, never reads
`core.config.IBMQuantumCredentials`, and never makes a network call
anywhere in its own code. `IBMRuntimeVerifier` is constructed with a
`service` object already in hand -- injecting it is the caller's problem,
not this module's. Every test in
`tests/quantum1b/hardware_readiness/test_ibm_verifier.py` passes a plain
mock/fake object satisfying `RuntimeServiceLike`/`RuntimeJobLike`; none
constructs or references a real `QiskitRuntimeService`.

**What remains for whoever actually runs Hardware Day** (documented, not
built here, per this phase's explicit no-real-hardware-job boundary): at
the point of actually verifying a real claimed job,
`service = QiskitRuntimeService(channel=..., token=IBMQuantumCredentials.require().token)`,
then `IBMRuntimeVerifier(service=service).verify(claimed)`. That one line
of wiring is the ONLY thing left -- every check this class performs
(job-id/backend/status/timestamp agreement, raw-result-plus-calibration
hashing) is implemented and tested below.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from core.errors import EvidenceIntegrityError
from core.schemas import HardwareRunEvidence
from core.verification import HardwareEvidenceVerifier, VerifiedAttestation

TERMINAL_SUCCESS_STATUS = "DONE"
DEFAULT_TIMESTAMP_TOLERANCE = timedelta(minutes=10)


class RuntimeJobLike(Protocol):
    """The exact subset of `qiskit_ibm_runtime.RuntimeJobV2`'s public API
    this verifier calls."""

    def job_id(self) -> str: ...
    def backend(self) -> Any: ...
    def status(self) -> str: ...
    def metrics(self) -> dict[str, Any]: ...
    def result(self) -> Any: ...


class RuntimeServiceLike(Protocol):
    def job(self, job_id: str) -> RuntimeJobLike: ...


@dataclass(frozen=True)
class IBMVerifierConfig:
    timestamp_tolerance: timedelta = DEFAULT_TIMESTAMP_TOLERANCE


class IBMRuntimeVerifier(HardwareEvidenceVerifier):
    """Implements every checklist item from
    `core.verification.IBMRuntimeVerifier`'s docstring against an injected
    `service`. Raises `EvidenceIntegrityError` (never silently downgrades
    or fabricates) at the first check that fails to confirm the claim."""

    provider = "ibm"

    def __init__(
        self, service: RuntimeServiceLike, config: IBMVerifierConfig | None = None
    ) -> None:
        self._service = service
        self._config = config or IBMVerifierConfig()

    def verify(self, claimed: HardwareRunEvidence) -> VerifiedAttestation:
        job = self._service.job(claimed.job_id)

        fetched_job_id = job.job_id()
        if fetched_job_id != claimed.job_id:
            raise EvidenceIntegrityError(
                f"IBMRuntimeVerifier: fetched job id {fetched_job_id!r} does not "
                f"match claimed job id {claimed.job_id!r}."
            )

        backend = job.backend()
        fetched_backend_name = getattr(backend, "name", None) or str(backend)
        if fetched_backend_name != claimed.backend_name:
            raise EvidenceIntegrityError(
                f"IBMRuntimeVerifier: fetched backend {fetched_backend_name!r} does "
                f"not match claimed backend {claimed.backend_name!r} for job "
                f"{claimed.job_id!r}."
            )

        status = job.status()
        if status != TERMINAL_SUCCESS_STATUS:
            raise EvidenceIntegrityError(
                f"IBMRuntimeVerifier: job {claimed.job_id!r} is not in a terminal-"
                f"success state (status={status!r}, required "
                f"{TERMINAL_SUCCESS_STATUS!r}) -- a queued/running/cancelled/errored "
                "job can never back a hardware-evidence claim."
            )

        metrics = job.metrics()
        timestamps = metrics.get("timestamps", {}) if isinstance(metrics, dict) else {}
        finished_raw = timestamps.get("finished")
        if not finished_raw:
            raise EvidenceIntegrityError(
                f"IBMRuntimeVerifier: job {claimed.job_id!r} has no "
                "metrics()['timestamps']['finished'] to verify the claimed "
                "execution_timestamp against."
            )
        finished_dt = (
            finished_raw
            if isinstance(finished_raw, datetime)
            else datetime.fromisoformat(str(finished_raw))
        )
        if finished_dt.tzinfo is None:
            finished_dt = finished_dt.replace(tzinfo=timezone.utc)
        claimed_dt = datetime.fromisoformat(claimed.execution_timestamp)
        drift = abs((finished_dt - claimed_dt).total_seconds())
        if drift > self._config.timestamp_tolerance.total_seconds():
            raise EvidenceIntegrityError(
                f"IBMRuntimeVerifier: job {claimed.job_id!r} actual completion "
                f"{finished_dt.isoformat()!r} does not match claimed "
                f"execution_timestamp {claimed.execution_timestamp!r} within "
                f"{self._config.timestamp_tolerance} (drift={drift:.1f}s)."
            )

        raw_result = job.result()
        raw_result_hash = _hash_result(raw_result, backend, metrics)
        digest_source = (
            f"{claimed.backend_name}:{claimed.job_id}:"
            f"{finished_dt.isoformat()}:{raw_result_hash}"
        )
        attestation_id = hashlib.sha256(digest_source.encode()).hexdigest()[:32]

        return VerifiedAttestation(
            attestation_id=attestation_id,
            provider=self.provider,
            backend_name=claimed.backend_name,
            job_id=claimed.job_id,
            verified_completion_timestamp=finished_dt.isoformat(),
            raw_result_hash=raw_result_hash,
            verifier_name="IBMRuntimeVerifier",
        )


def _hash_result(raw_result: Any, backend: Any, metrics: dict[str, Any]) -> str:
    """Bind the raw job result plus whatever backend/calibration metadata
    is cheaply available into a single hash -- checklist item 5 ('the
    attestation should be checkable independent of this object'). This is
    a binding hash, not a canonical serialization format: falls back to
    `repr()` for anything not JSON-serializable."""

    def _safe(obj: Any) -> str:
        try:
            return json.dumps(obj, sort_keys=True, default=str)
        except TypeError:
            return repr(obj)

    backend_name = getattr(backend, "name", None) or str(backend)
    payload = _safe(raw_result) + "|" + backend_name + "|" + _safe(metrics)
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "IBMRuntimeVerifier",
    "IBMVerifierConfig",
    "RuntimeJobLike",
    "RuntimeServiceLike",
    "TERMINAL_SUCCESS_STATUS",
]
