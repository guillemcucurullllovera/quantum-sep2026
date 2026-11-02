"""The real-hardware entry point. This module — and only this module plus
`hardware.sessions.PlatformSession` and the QPU branch inside
`hardware.ibm_runtime.primitives` — is allowed to attempt a real IBM
Quantum submission. Every path through here calls
`core.config.get_settings().ibm.require()` before doing anything else, so
a missing credential fails loudly and immediately rather than silently
degrading to a simulator run that the caller might mistake for real
hardware data.

**Budget enforcement (added 2026-10-01, remediating independent review
finding P1-1):** `budget_tracker`/`session_id` are REQUIRED keyword-only
parameters — there is no default, so a caller that omits them gets a
`TypeError` before any network call, not a silent unbudgeted submission.
This was previously the single largest gap in this platform's hardware-
readiness work: an entire concurrency-hardened budget subsystem
(`experiments.quantum1b.hardware_readiness.persistent_budget.
PersistentBudgetTracker`) existed and was tested, but governed nothing on
the one code path that can actually reach real hardware. `budget_tracker`
is typed here as the local `BudgetTrackerLike` Protocol (structural,
duck-typed) rather than importing `PersistentBudgetTracker` directly, so
this general-purpose `hardware/` package does not take a hard dependency
on `experiments.quantum1b` — any object with a matching
`record_submission(session_id, shots, *, qpu_seconds=0.0, usd_cost=0.0)`
method works, including `PersistentBudgetTracker` itself.

**Retry/idempotency (same remediation):** submission and waiting-for-result
are now retried SEPARATELY. Once `submit_qpu_sampler_job` returns a job
handle, a real job may already be billed/queued; a transient failure while
WAITING for that job's result (`job.result()`) now retries by re-fetching
the SAME job, never by calling `sampler.run()` again — eliminating the
"up to 3 real duplicate submissions for one caller intent" risk the
original single-retry-wraps-everything structure had.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.config import get_settings
from core.errors import CredentialError
from core.schemas import BackendDescriptor, BackendKind
from hardware.ibm_runtime.primitives import SamplerRunResult, fetch_qpu_sampler_result, submit_qpu_sampler_job
from hardware.retries import RetryPolicy, call_with_retry


class BudgetTrackerLike(Protocol):
    """Structural type for whatever budget tracker is injected —
    `experiments.quantum1b.hardware_readiness.persistent_budget.
    PersistentBudgetTracker` satisfies this today; any object with a
    matching `record_submission` satisfies it, including a test double."""

    def record_submission(
        self, session_id: str, shots: int, *, qpu_seconds: float = 0.0, usd_cost: float = 0.0
    ) -> object: ...


@dataclass
class QPUExecutionResult:
    sampler_result: SamplerRunResult
    backend_name: str
    was_retried: bool


def execute_on_qpu(
    circuit,
    backend_name: str,
    shots: int = 4096,
    retry_policy: RetryPolicy | None = None,
    *,
    budget_tracker: BudgetTrackerLike,
    session_id: str,
    estimated_qpu_seconds: float = 0.0,
    estimated_usd_cost: float = 0.0,
    service=None,
) -> QPUExecutionResult:
    """Submit `circuit` to a real IBM backend named `backend_name`.

    Raises `core.errors.CredentialError` immediately — before touching the
    network — if `IBM_QUANTUM_TOKEN` is not set. This is the load-bearing
    behavior of the whole hardware/ layer: there is no code path anywhere
    in this platform that silently substitutes a simulator for a requested
    QPU run.

    `budget_tracker`/`session_id` are required (see module docstring) and
    are checked/recorded BEFORE constructing a service or making any
    network call — a refused submission (`BudgetExceededError`) never
    reaches IBM at all. `service`, if supplied, is used instead of
    constructing a real `QiskitRuntimeService` — the injection point tests
    use to exercise this function without ever touching a real account
    (same pattern as `hardware_readiness.ibm_verifier.IBMRuntimeVerifier`).
    """
    get_settings().ibm.require()  # raises CredentialError, loud and early

    # Budget check/record happens before ANY network call, including
    # service construction and backend resolution — fail-closed, and a
    # refused submission leaves no real-hardware trace at all.
    budget_tracker.record_submission(
        session_id, shots, qpu_seconds=estimated_qpu_seconds, usd_cost=estimated_usd_cost
    )

    if service is None:
        from qiskit_ibm_runtime import QiskitRuntimeService

        creds = get_settings().ibm
        service = QiskitRuntimeService(channel=creds.channel, token=creds.token, instance=creds.instance)

    backend = service.backend(backend_name)
    descriptor = BackendDescriptor(
        name=backend_name,
        kind=BackendKind.QPU,
        num_qubits=backend.num_qubits,
        calibration_source=f"live:{backend_name}",
    )

    submit_attempts = {"n": 0}

    def _submit():
        submit_attempts["n"] += 1
        return submit_qpu_sampler_job(circuit, descriptor, backend, shots=shots)

    # Retrying SUBMISSION is safe: if sampler.run() itself raises before
    # returning a job handle, no billable job exists yet to duplicate.
    job = call_with_retry(_submit, policy=retry_policy or RetryPolicy(max_attempts=3))

    wait_attempts = {"n": 0}

    def _wait():
        wait_attempts["n"] += 1
        return fetch_qpu_sampler_result(job, descriptor, shots)

    # Retrying the WAIT re-polls the SAME job — never resubmits, so a
    # transient failure here (e.g. a network blip while blocked in
    # job.result()) cannot create a second real job for this circuit.
    result = call_with_retry(_wait, policy=retry_policy or RetryPolicy(max_attempts=3))

    was_retried = submit_attempts["n"] > 1 or wait_attempts["n"] > 1
    return QPUExecutionResult(sampler_result=result, backend_name=backend_name, was_retried=was_retried)


__all__ = ["execute_on_qpu", "QPUExecutionResult", "BudgetTrackerLike"]
