"""End-to-end dry run of `docs/quantum/QUANTUM1B_HARDWARE_DAY_RUNBOOK.md`'s
assembled procedure, against entirely fake/mock components.

**Why this exists.** Every individual piece of the hardware-readiness
infrastructure (evidence store, verifier, lineage, budgets, backend
selection, frozen controls) has its own unit tests -- but until this
module, the ASSEMBLED procedure (the actual order of operations the
runbook describes: select backend -> check budget -> "submit" -> verify ->
attest -> record lineage, for each frozen control instance) had never been
exercised as a whole. This closes that gap
(`PRE_HARDWARE_QUANTUM_CAPSTONE.md` §4 item 8) without touching a real IBM
service -- every "hardware job" here is a fake, deterministic stand-in
(`FakeDryRunJob`) constructed FROM `FROZEN_CONTROLS`' own pre-computed
expected exact optimum, so this validates PIPELINE MECHANICS (does the
sequence of calls work, do the right objects end up in the right stores,
does budget accounting come out correct), not physics -- the physics
(does a real QAOA circuit reach a good objective on real or noisy hardware)
is Track B's and Track D's job, not this module's.

**No real network call, no real credentials, anywhere in this module** --
same discipline as `ibm_verifier.py`, enforced the same way (see
`tests/quantum1b/hardware_readiness/test_dry_run_runbook.py`'s AST-level
import check).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.schemas import HardwareRunEvidence
from core.verification import attest
from experiments.quantum1b.hardware_readiness.budgets import (
    BackendCandidate,
    BackendSelectionCriteria,
    select_backend,
)
from experiments.quantum1b.hardware_readiness.evidence_store import PersistentEvidenceStore
from experiments.quantum1b.hardware_readiness.frozen_controls import FROZEN_CONTROLS, verify_frozen_controls
from experiments.quantum1b.hardware_readiness.ibm_verifier import IBMRuntimeVerifier
from experiments.quantum1b.hardware_readiness.persistent_budget import PersistentBudgetTracker


@dataclass
class FakeDryRunJob:
    """A deterministic stand-in for a real `RuntimeJobV2`, constructed
    directly from a `FrozenControlInstance`'s own pre-computed expected
    exact optimum -- i.e. this dry run assumes a hypothetical real job
    returned exactly the simulator-predicted answer, which is the
    mechanically-simplest fake that still exercises every real check the
    verifier performs (job-id/backend/status/timestamp agreement,
    result hashing)."""

    _job_id: str
    _backend_name: str
    _finished: datetime
    _exact_optimum: float

    def job_id(self) -> str:
        return self._job_id

    def backend(self) -> Any:
        return type("FakeBackendRef", (), {"name": self._backend_name})()

    def status(self) -> str:
        return "DONE"

    def metrics(self) -> dict:
        return {"timestamps": {"finished": self._finished}}

    def result(self) -> Any:
        return {"dry_run_exact_optimum_stand_in": self._exact_optimum}


@dataclass
class FakeDryRunService:
    jobs: dict[str, FakeDryRunJob] = field(default_factory=dict)

    def job(self, job_id: str) -> FakeDryRunJob:
        return self.jobs[job_id]


@dataclass(frozen=True)
class DryRunStepResult:
    family: str
    n_candidates: int
    backend_selected: str
    budget_snapshot_jobs_submitted: int
    attestation_id: str
    attestation_raw_result_hash: str
    """**Renamed from `lineage_record_hash` 2026-10-01, remediating
    independent-review finding P2-6's second complaint**: the old name
    claimed to be a lineage-record hash but was actually populated from
    the attestation's own `raw_result_hash` -- disjoint information from
    what `hardware_readiness.lineage.RunLineageRecord.record_hash()`
    actually hashes. This dry run validates ORCHESTRATION mechanics, not
    physics (see module docstring) -- it never builds a real transpiled
    circuit, so it has no real `TranspiledCircuit` to build an honest
    `RunLineageRecord` from. `build_lineage_record()` is no longer dead
    code: it is now genuinely exercised by `noise_validation.
    run_noise_validation_cell()`'s real (not dry-run) execution path,
    which already builds a real circuit for Track D's EXP-D1."""
    evidence_level_reached: str


_DRY_RUN_PATH_MARKER = "dry_run"


def run_dry_run_runbook(
    *,
    evidence_db_path,
    budget_db_path,
    session_id: str = "dry-run-session",
) -> list[DryRunStepResult]:
    """Executes the runbook's per-instance sequence (§2-5 of
    `QUANTUM1B_HARDWARE_DAY_RUNBOOK.md`) for every `FROZEN_CONTROLS`
    instance, against fakes. Returns one `DryRunStepResult` per instance.
    Raises on ANY step's own real failure mode (budget exceeded, frozen
    control drifted, verification mismatch) -- a dry run that swallowed
    errors would be worse than no dry run at all.

    **Requires `evidence_db_path` and `budget_db_path` to both contain the
    literal substring `"dry_run"` (fixed 2026-10-01, remediating
    independent-review finding P2-2).** Before this fix, a dry-run
    `VerifiedAttestation` (produced entirely from `FakeDryRunJob`/
    `FakeDryRunService`, no real IBM service anywhere) was structurally
    indistinguishable, once persisted, from a genuine Hardware Day
    attestation -- `VerifiedAttestation` has no provenance/is_real field
    (that schema is frozen, shared `core.verification` infrastructure, out
    of this session's scope to modify -- see
    `QUANTUM1B_ESTATE_INVENTORY.md`). The narrower, in-scope fix: refuse,
    fail-closed, to run a dry run against any path that doesn't look like a
    dry-run path, so a dry-run session can never be accidentally pointed
    at (and silently commingle fake evidence into) what might be a real
    Hardware Day session's store. This does not fully resolve P2-2 (a
    caller could still rename a path to satisfy the marker in bad faith),
    but closes the actual risk scenario the finding describes: accidental
    collision, not deliberate circumvention."""
    for label, path in (("evidence_db_path", evidence_db_path), ("budget_db_path", budget_db_path)):
        if _DRY_RUN_PATH_MARKER not in str(path):
            raise ValueError(
                f"run_dry_run_runbook: {label}={path!r} does not contain "
                f"{_DRY_RUN_PATH_MARKER!r} -- refusing to run a dry run against a path "
                "that doesn't look like a dedicated dry-run store (P2-2 safeguard: "
                "prevents a dry run's fake evidence from ever landing in what might be "
                "a real Hardware Day session's store)."
            )
    # check_simulator_baseline=False: this dry run validates ORCHESTRATION
    # mechanics (does the pipeline's sequence of calls work), not Track B
    # physics -- re-running 6 COBYLA optimizations here on every dry run
    # would be needless (and slower) duplication of what
    # verify_frozen_controls()'s own default already covers for whoever
    # wants that check specifically.
    frozen_check = verify_frozen_controls(check_simulator_baseline=False)
    if not all(v.all_match for v in frozen_check):
        raise RuntimeError(
            f"dry run aborted: FROZEN_CONTROLS has drifted from current code: "
            f"{[v for v in frozen_check if not v.all_match]}"
        )

    evidence_store = PersistentEvidenceStore(evidence_db_path)
    budget_tracker = PersistentBudgetTracker(budget_db_path)
    from experiments.quantum1b.hardware_readiness.budgets import HardwareDayBudget

    budget = HardwareDayBudget(
        max_jobs=len(FROZEN_CONTROLS), max_shots_per_job=4096,
        max_total_shots=4096 * len(FROZEN_CONTROLS), max_qpu_seconds=600.0,
    )
    budget_tracker.get_or_create_session(session_id, budget)

    candidates = [
        BackendCandidate(name="ibm_kingston", kind="qpu", num_qubits=156, pending_jobs=3, median_two_qubit_error=0.008),
    ]
    criteria = BackendSelectionCriteria(min_qubits_required=9, require_credentials_present=True)
    chosen, refusal = select_backend(candidates, criteria, credentials_present=True)
    if chosen is None:
        raise RuntimeError(f"dry run aborted: backend selection refused: {refusal}")

    fake_service = FakeDryRunService()
    results: list[DryRunStepResult] = []

    for control in FROZEN_CONTROLS:
        job_id = f"dry-run-job-{control.family}-{control.n_candidates}"
        finished = datetime.now(timezone.utc)

        # **Reordered 2026-10-01, remediating independent-review finding
        # P2-10** ("the dry-run pipeline registers the fake job before
        # checking budget, inverting its own documented order"). Budget is
        # now checked/recorded BEFORE the job is registered -- harmless
        # here either way (an in-memory fake job costs nothing), but this
        # is also the template shape a FUTURE REAL submission script would
        # most plausibly copy; checking budget after an uncancellable,
        # costed real submission would mean `BudgetExceededError` fires
        # only after spend already happened.
        snap = budget_tracker.record_submission(session_id, shots=4096, qpu_seconds=10.0)

        fake_service.jobs[job_id] = FakeDryRunJob(
            _job_id=job_id, _backend_name=chosen.name, _finished=finished,
            _exact_optimum=control.expected_exact_optimum,
        )

        claimed = HardwareRunEvidence(
            backend_name=chosen.name, job_id=job_id, execution_timestamp=finished.isoformat(),
        )
        verifier = IBMRuntimeVerifier(service=fake_service)
        attestation_id = attest(claimed, verifier, evidence_store)

        results.append(
            DryRunStepResult(
                family=control.family,
                n_candidates=control.n_candidates,
                backend_selected=chosen.name,
                budget_snapshot_jobs_submitted=snap.jobs_submitted,
                attestation_id=attestation_id,
                attestation_raw_result_hash=evidence_store.get(attestation_id).raw_result_hash,
                evidence_level_reached="F",
            )
        )

    return results


__all__ = ["FakeDryRunJob", "FakeDryRunService", "DryRunStepResult", "run_dry_run_runbook"]
