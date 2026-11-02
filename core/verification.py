"""Hardware-evidence verification boundary — separates CLAIMED metadata
from VERIFIED evidence.

**Why this exists.** An independent adversarial review of QUANTUM-1A
demonstrated that `core.evidence.compute_evidence_level` was fail-closed
against *omission* (a missing `job_id`/`backend_name` raises) but NOT
fail-closed against *fabrication*: a caller could construct
`HardwareRunEvidence(backend_name="ibm_brisbane",
job_id="totally-made-up-job-id", execution_timestamp=<any well-formed
string>)`, attach it to an `ExperimentResult`, and receive evidence level F
with zero cross-checks against any external system of record. This
directly reproduced the exact failure class QUANTUM-0 found in the first
place (an evidence claim that outruns what actually happened) — the old
schema closed the *specific* historical bug (a dangling/wrong citation to a
file that didn't exist) but not the *general* class (a plausible-looking
citation to a job/backend that was never checked against anything).

**The fix.** `HardwareRunEvidence` (in `core.schemas`) remains the CLAIMED
side — self-typed, and `core.schemas`'s own field validators now reject
obviously-malformed claims (blank fields, an unparseable/naive/future
`execution_timestamp`). But claimed fields alone can never reach evidence
level F. Level F additionally requires `HardwareRunEvidence.attestation_id`
to resolve, via an `EvidenceStore`, to a `VerifiedAttestation` that only
`attest()` — by actually running the claim through a
`HardwareEvidenceVerifier` — could have produced. A serialized
`ExperimentResult`, however carefully hand-crafted, cannot manufacture a
matching store entry by itself: the store is a separate runtime object, not
a field inside the result's own JSON.

**What is implemented now vs. deferred to the real-hardware phase.**
Implemented: the verifier interface, `VerifiedAttestation`, an in-memory
reference `EvidenceStore`, a `MockVerifier` for tests, and the `attest()`
wiring `core.evidence` relies on. Deferred, and documented rather than
silently stubbed: `IBMRuntimeVerifier.verify()` raises `NotImplementedError`
— no real IBM Runtime network call is made anywhere in this module. Before
this is used to gate a real hardware-evidence claim, `IBMRuntimeVerifier`
must be finished (see its docstring for the exact checklist), and the
`InMemoryEvidenceStore` used here must be replaced with something
persistent — see this module's own docstring note under
`InMemoryEvidenceStore` and `docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md`.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from core.schemas import HardwareRunEvidence


@dataclass(frozen=True)
class VerifiedAttestation:
    """A verifier's statement that a claimed hardware run really happened
    as claimed. Only ever constructed by a `HardwareEvidenceVerifier`'s
    `verify()` method — never by hand in application code."""

    attestation_id: str
    provider: str
    backend_name: str
    job_id: str
    verified_completion_timestamp: str
    raw_result_hash: str
    verifier_name: str


class HardwareEvidenceVerifier(ABC):
    """Provider-aware boundary between a claim and a fact. A real
    implementation for a given provider must, at minimum: retrieve the job
    by `job_id` from that provider's own service API; verify the fetched
    job's backend identity matches the claim; require a terminal
    successful status (not queued/running/cancelled/errored); verify the
    job's actual completion timestamp against the claim; and bind (e.g.
    hash) the raw provider result and relevant calibration/backend
    metadata into `raw_result_hash` so the attestation is falsifiable
    independent of the `ExperimentResult` object itself.
    """

    provider: str

    @abstractmethod
    def verify(self, claimed: HardwareRunEvidence) -> VerifiedAttestation:
        """Return a VerifiedAttestation, or raise if the claim cannot be
        confirmed against the external system of record. Must never
        return a VerifiedAttestation for a claim it did not actually check."""


class IBMRuntimeVerifier(HardwareEvidenceVerifier):
    """**Not implemented.** No IBM Runtime network call is made by this
    class today — it exists to document the exact shape the real
    implementation must have, so a future contributor doesn't have to
    design this from scratch under hardware-spend time pressure.

    Required before this class gates a real hardware-evidence claim:
    1. `QiskitRuntimeService(...).job(claimed.job_id)` — fetch the real job.
    2. Assert the fetched job's backend name equals `claimed.backend_name`.
    3. Assert the fetched job's status is terminal-success (`DONE`), not
       queued/running/cancelled/errored.
    4. Assert the fetched job's actual completion timestamp matches
       `claimed.execution_timestamp` within a documented tolerance.
    5. Hash the raw job result payload plus the backend's calibration
       snapshot at execution time into `raw_result_hash`, and persist that
       raw payload (or a reference to it) somewhere auditable — the
       attestation should be checkable independent of this object.
    """

    provider = "ibm"

    def verify(self, claimed: HardwareRunEvidence) -> VerifiedAttestation:
        raise NotImplementedError(
            "Real IBM Runtime verification is not implemented. See this "
            "class's docstring for the exact checklist, and "
            "docs/quantum/QUANTUM1A_REVIEW_REMEDIATION.md for the "
            "remaining-work record. No IBM network call is made anywhere "
            "in core.verification."
        )


class MockVerifier(HardwareEvidenceVerifier):
    """Test-only verifier. 'Verifies' by construction — it does not check
    anything against a real external system — so it must never be treated
    as a default or fallback verifier for a real evidence claim. There is
    deliberately no default verifier anywhere in this module; every caller
    of `attest()` must supply one explicitly, so a real hardware-evidence
    claim can never silently pass through a mock."""

    provider = "mock"

    def verify(self, claimed: HardwareRunEvidence) -> VerifiedAttestation:
        raw = f"{claimed.backend_name}:{claimed.job_id}:{claimed.execution_timestamp}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return VerifiedAttestation(
            attestation_id=digest[:24],
            provider=self.provider,
            backend_name=claimed.backend_name,
            job_id=claimed.job_id,
            verified_completion_timestamp=claimed.execution_timestamp,
            raw_result_hash=digest,
            verifier_name="MockVerifier",
        )


class EvidenceStore(Protocol):
    def put(self, attestation: VerifiedAttestation) -> None: ...
    def get(self, attestation_id: str) -> VerifiedAttestation | None: ...


class InMemoryEvidenceStore:
    """Reference `EvidenceStore` implementation. Process-local only: does
    not survive a process restart and is not shared across processes.

    **Remaining limitation, documented not hidden:** a real Phase-3
    deployment needs a persistent, tamper-evident store (e.g. a dedicated,
    write-restricted table alongside `core.experiment_registry`'s SQLite
    database) so an attestation recorded by the process that ran a real
    hardware job can still be resolved by a different later process (a
    report generator, a re-run, code review). Using an in-memory store
    here is a deliberate, narrow scope choice for Phase 0/1A remediation —
    it is sufficient to prove the CLAIMED-vs-VERIFIED separation works, and
    to keep a hand-edited/deserialized `ExperimentResult` from manufacturing
    level F on its own — but it is NOT sufficient for a real cross-process
    hardware-evidence workflow. Replace before relying on this for a real
    QPU run whose result will be read back later or by another process.
    """

    def __init__(self) -> None:
        self._records: dict[str, VerifiedAttestation] = {}

    def put(self, attestation: VerifiedAttestation) -> None:
        self._records[attestation.attestation_id] = attestation

    def get(self, attestation_id: str) -> VerifiedAttestation | None:
        return self._records.get(attestation_id)


def attest(
    claimed: HardwareRunEvidence,
    verifier: HardwareEvidenceVerifier,
    store: EvidenceStore,
) -> str:
    """The only sanctioned way to obtain an `attestation_id`: run `claimed`
    through `verifier` (which may raise — e.g. `IBMRuntimeVerifier` always
    does today), store the resulting `VerifiedAttestation`, and return its
    id for the caller to set on `HardwareRunEvidence.attestation_id`."""
    attestation = verifier.verify(claimed)
    store.put(attestation)
    return attestation.attestation_id


__all__ = [
    "VerifiedAttestation",
    "HardwareEvidenceVerifier",
    "IBMRuntimeVerifier",
    "MockVerifier",
    "EvidenceStore",
    "InMemoryEvidenceStore",
    "attest",
]
