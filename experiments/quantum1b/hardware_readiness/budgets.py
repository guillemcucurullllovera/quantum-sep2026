"""Hardware-day budgets, stopping rules, backend-selection criteria, and
retry/refusal policy.

None of this existed anywhere in the estate before this pass (verified by
searching for "shot_budget"/"run_budget"/"BackendSelection" across
`core/`, `algorithms/`, `applications/` -- nothing matched). This is
genuinely net-new hardware-readiness engineering, not a reuse-and-extend of
an existing module, unlike `evidence_store.py`/`ibm_verifier.py` (which
extend `core.verification`'s existing Protocol/ABC).

**Design principle: refuse loudly, never retry blindly.** Every function
here that can refuse to proceed returns a typed `RefusalReason` (or raises
`BudgetExceededError`) rather than silently downgrading scope, and
`RetryPolicy` caps retries at a small fixed count with NO retry at all for
error classes that are definitionally not transient (an
`EvidenceIntegrityError`, a malformed claim, an ineligible backend) --
retrying those would just resubmit the same already-answered question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RefusalReason(str, Enum):
    NO_CREDENTIALS = "no_credentials"
    OVER_JOB_BUDGET = "over_job_budget"
    OVER_SHOT_BUDGET = "over_shot_budget"
    OVER_QPU_SECONDS_BUDGET = "over_qpu_seconds_budget"
    OVER_COST_BUDGET = "over_cost_budget"
    BACKEND_NOT_ELIGIBLE = "backend_not_eligible"
    NO_ELIGIBLE_BACKEND = "no_eligible_backend"
    PREREGISTRATION_MISSING = "preregistration_missing"


class BudgetExceededError(Exception):
    """Raised by `BudgetTracker.record_submission` if called after
    `can_submit` would have returned False -- callers must check
    `can_submit` first; this is the fail-closed backstop if one doesn't."""


# ---------------------------------------------------------------------------
# Budgets / stopping rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareDayBudget:
    """A pre-committed (preregistration-style) ceiling on real hardware
    spend for one Hardware Day session. Every field is a HARD cap -- there
    is deliberately no "soft warn then continue" mode; `BudgetTracker`
    refuses once any cap would be exceeded."""

    max_jobs: int
    max_shots_per_job: int
    max_total_shots: int
    max_qpu_seconds: float
    max_usd_cost: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_jobs", "max_shots_per_job", "max_total_shots", "max_qpu_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"HardwareDayBudget.{name} must be positive, got {getattr(self, name)}")
        if self.max_usd_cost is not None and self.max_usd_cost <= 0:
            raise ValueError(f"HardwareDayBudget.max_usd_cost must be positive if set, got {self.max_usd_cost}")


@dataclass
class BudgetTracker:
    """Mutable, in-process accounting against a `HardwareDayBudget`. Not
    itself persistent -- a real Hardware Day session should pair this with
    `PersistentEvidenceStore`-style durability before relying on it across
    process restarts; tracked here as a known, disclosed gap (see
    `docs/quantum/QUANTUM1B_HARDWARE_READINESS.md`)."""

    budget: HardwareDayBudget
    jobs_submitted: int = 0
    total_shots_submitted: int = 0
    qpu_seconds_used: float = 0.0
    usd_cost_used: float = 0.0

    def can_submit(self, shots: int, *, estimated_qpu_seconds: float = 0.0, estimated_usd_cost: float = 0.0) -> tuple[bool, RefusalReason | None]:
        if self.jobs_submitted + 1 > self.budget.max_jobs:
            return False, RefusalReason.OVER_JOB_BUDGET
        if shots > self.budget.max_shots_per_job:
            return False, RefusalReason.OVER_SHOT_BUDGET
        if self.total_shots_submitted + shots > self.budget.max_total_shots:
            return False, RefusalReason.OVER_SHOT_BUDGET
        if self.qpu_seconds_used + estimated_qpu_seconds > self.budget.max_qpu_seconds:
            return False, RefusalReason.OVER_QPU_SECONDS_BUDGET
        if self.budget.max_usd_cost is not None and self.usd_cost_used + estimated_usd_cost > self.budget.max_usd_cost:
            return False, RefusalReason.OVER_COST_BUDGET
        return True, None

    def record_submission(self, shots: int, *, qpu_seconds: float = 0.0, usd_cost: float = 0.0) -> None:
        """**Validates non-negative inputs (fixed 2026-10-01, remediating
        independent-review finding P2-8).** Before this fix, a negative
        `shots`/`qpu_seconds`/`usd_cost` (e.g. from a cost-estimation sign
        error upstream) was accepted with no error and drove the running
        totals permanently negative -- reproduced directly:
        `record_submission(-5000, qpu_seconds=-500, usd_cost=-500)` left
        `total_shots_submitted=-5000`, after which legitimate submissions
        could keep succeeding indefinitely since the cumulative total never
        climbed back above the cap -- a silent defeat of the documented
        "HARD cap... no soft warn then continue" promise."""
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        if qpu_seconds < 0:
            raise ValueError(f"qpu_seconds must be >= 0, got {qpu_seconds}")
        if usd_cost < 0:
            raise ValueError(f"usd_cost must be >= 0, got {usd_cost}")
        ok, reason = self.can_submit(shots, estimated_qpu_seconds=qpu_seconds, estimated_usd_cost=usd_cost)
        if not ok:
            raise BudgetExceededError(
                f"cannot record submission of {shots} shots: {reason.value if reason else 'unknown'} "
                f"(jobs_submitted={self.jobs_submitted}/{self.budget.max_jobs}, "
                f"total_shots={self.total_shots_submitted}/{self.budget.max_total_shots}, "
                f"qpu_seconds={self.qpu_seconds_used}/{self.budget.max_qpu_seconds})"
            )
        self.jobs_submitted += 1
        self.total_shots_submitted += shots
        self.qpu_seconds_used += qpu_seconds
        self.usd_cost_used += usd_cost

    def remaining_jobs(self) -> int:
        return max(0, self.budget.max_jobs - self.jobs_submitted)

    def remaining_shots(self) -> int:
        return max(0, self.budget.max_total_shots - self.total_shots_submitted)


# ---------------------------------------------------------------------------
# Backend-selection criteria
# ---------------------------------------------------------------------------


_VALID_BACKEND_KINDS = frozenset({"simulator", "fake", "qpu"})


@dataclass(frozen=True)
class BackendCandidate:
    """Minimal, provider-agnostic shape describing one candidate backend --
    deliberately NOT `qiskit_ibm_runtime`'s own richer backend object, so
    this module's selection logic can be unit-tested with plain data and
    never needs a real service connection.

    **`kind` is validated at construction (fixed 2026-10-01, remediating
    independent-review finding P2-1).** `select_backend`'s per-candidate
    QPU check (`c.kind != "qpu"`) is a bare string comparison -- before this
    fix, `BackendCandidate(kind="QPU", ...)` (or any other casing/typo) was
    accepted with zero validation and then silently bypassed that check,
    reproduced directly: `select_backend([BackendCandidate(kind="QPU", ...)],
    BackendSelectionCriteria(..., require_credentials_present=False),
    credentials_present=False)` returned the QPU-cased candidate as the
    chosen backend. (Note on actual exposure: `select_backend`'s own
    top-level `criteria.require_credentials_present and not
    credentials_present` check already refuses with `NO_CREDENTIALS` by
    default before reaching the per-candidate filter at all -- this bug
    only mattered when a caller explicitly set
    `require_credentials_present=False` and relied on the per-candidate
    filter as the sole safety net, which no caller in this codebase does
    today. Fixed here anyway, since "no current caller happens to avoid it"
    is not the same as "cannot happen.")"""

    name: str
    kind: str  # "simulator" | "fake" | "qpu" -- mirrors core.schemas.BackendKind values
    num_qubits: int
    pending_jobs: int = 0
    median_two_qubit_error: float | None = None  # None only legitimate for simulator/fake with no calibration

    def __post_init__(self) -> None:
        if self.kind not in _VALID_BACKEND_KINDS:
            raise ValueError(
                f"BackendCandidate.kind must be one of {sorted(_VALID_BACKEND_KINDS)}, got {self.kind!r}"
            )


@dataclass(frozen=True)
class BackendSelectionCriteria:
    """Preregistered-style, documented backend-selection policy -- applied
    mechanically, not by ad hoc judgment at Hardware Day time."""

    min_qubits_required: int
    max_acceptable_pending_jobs: int = 50
    max_acceptable_two_qubit_error: float = 0.02
    require_credentials_present: bool = True


def select_backend(
    candidates: list[BackendCandidate],
    criteria: BackendSelectionCriteria,
    *,
    credentials_present: bool,
) -> tuple[BackendCandidate | None, RefusalReason | None]:
    """Deterministic backend selection: filter to eligible candidates (per
    `criteria`), then pick the eligible candidate with the FEWEST pending
    jobs (a documented, simple tie-break — shortest queue — not the only
    reasonable policy, but a fixed, disclosed one rather than an
    unspecified "pick the best one" judgment call at Hardware Day time).
    Returns `(None, RefusalReason.NO_ELIGIBLE_BACKEND)` if nothing
    qualifies; never returns a QPU-kind candidate when
    `credentials_present=False`."""
    if criteria.require_credentials_present and not credentials_present:
        return None, RefusalReason.NO_CREDENTIALS

    eligible = [
        c
        for c in candidates
        if c.num_qubits >= criteria.min_qubits_required
        and c.pending_jobs <= criteria.max_acceptable_pending_jobs
        and (
            c.median_two_qubit_error is None
            or c.median_two_qubit_error <= criteria.max_acceptable_two_qubit_error
        )
        and (c.kind != "qpu" or credentials_present)
    ]
    if not eligible:
        return None, RefusalReason.NO_ELIGIBLE_BACKEND
    best = min(eligible, key=lambda c: c.pending_jobs)
    return best, None


# ---------------------------------------------------------------------------
# Retry / refusal policy
# ---------------------------------------------------------------------------


NON_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "EvidenceIntegrityError",  # the claim was checked and is definitively wrong -- retrying resubmits the same wrong claim
        "CredentialError",  # missing credentials don't fix themselves on retry
        "BudgetExceededError",  # budget is exhausted, not transient
        "ValidationError",  # malformed input, not a transient fault
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")

    def is_retryable(self, exc: BaseException) -> bool:
        return type(exc).__name__ not in NON_RETRYABLE_EXCEPTION_NAMES


def run_with_retry(
    fn: Callable[[], "object"],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = lambda _s: None,
) -> "object":
    """Run `fn()`, retrying up to `policy.max_attempts` times ONLY for
    exceptions `policy.is_retryable` accepts; any non-retryable exception
    (see `NON_RETRYABLE_EXCEPTION_NAMES`) propagates immediately on the
    first occurrence, never retried. `sleep` is injectable so tests never
    actually sleep."""
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised below
            last_exc = exc
            if not policy.is_retryable(exc):
                raise
            if attempt < policy.max_attempts:
                sleep(policy.backoff_seconds * attempt)
    assert last_exc is not None
    raise last_exc


__all__ = [
    "RefusalReason",
    "BudgetExceededError",
    "HardwareDayBudget",
    "BudgetTracker",
    "BackendCandidate",
    "BackendSelectionCriteria",
    "select_backend",
    "RetryPolicy",
    "NON_RETRYABLE_EXCEPTION_NAMES",
    "run_with_retry",
]
