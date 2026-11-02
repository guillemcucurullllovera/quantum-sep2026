"""Tests for `experiments.quantum1b.hardware_readiness.budgets`."""
from __future__ import annotations

import pytest

from experiments.quantum1b.hardware_readiness.budgets import (
    BackendCandidate,
    BackendSelectionCriteria,
    BudgetExceededError,
    BudgetTracker,
    HardwareDayBudget,
    NON_RETRYABLE_EXCEPTION_NAMES,
    RefusalReason,
    RetryPolicy,
    run_with_retry,
    select_backend,
)


class TestHardwareDayBudgetValidation:
    def test_rejects_non_positive_fields(self):
        with pytest.raises(ValueError):
            HardwareDayBudget(max_jobs=0, max_shots_per_job=100, max_total_shots=100, max_qpu_seconds=10)

    def test_accepts_valid_budget(self):
        b = HardwareDayBudget(max_jobs=5, max_shots_per_job=4096, max_total_shots=20000, max_qpu_seconds=600)
        assert b.max_jobs == 5


class TestBudgetTracker:
    def _budget(self) -> HardwareDayBudget:
        return HardwareDayBudget(max_jobs=3, max_shots_per_job=4096, max_total_shots=10000, max_qpu_seconds=100)

    def test_can_submit_within_budget(self):
        t = BudgetTracker(self._budget())
        ok, reason = t.can_submit(1000)
        assert ok and reason is None

    def test_refuses_over_shots_per_job(self):
        t = BudgetTracker(self._budget())
        ok, reason = t.can_submit(5000)
        assert not ok and reason == RefusalReason.OVER_SHOT_BUDGET

    def test_refuses_over_total_shots(self):
        t = BudgetTracker(self._budget())
        t.record_submission(4000)
        t.record_submission(4000)
        ok, reason = t.can_submit(4000)
        assert not ok and reason == RefusalReason.OVER_SHOT_BUDGET

    def test_refuses_over_job_count(self):
        t = BudgetTracker(self._budget())
        t.record_submission(100)
        t.record_submission(100)
        t.record_submission(100)
        ok, reason = t.can_submit(100)
        assert not ok and reason == RefusalReason.OVER_JOB_BUDGET

    def test_refuses_over_qpu_seconds(self):
        t = BudgetTracker(self._budget())
        ok, reason = t.can_submit(100, estimated_qpu_seconds=200)
        assert not ok and reason == RefusalReason.OVER_QPU_SECONDS_BUDGET

    def test_record_submission_raises_if_over_budget(self):
        t = BudgetTracker(self._budget())
        with pytest.raises(BudgetExceededError):
            t.record_submission(5000)

    def test_record_submission_updates_state(self):
        t = BudgetTracker(self._budget())
        t.record_submission(1000, qpu_seconds=10, usd_cost=5)
        assert t.jobs_submitted == 1
        assert t.total_shots_submitted == 1000
        assert t.qpu_seconds_used == 10
        assert t.usd_cost_used == 5
        assert t.remaining_jobs() == 2
        assert t.remaining_shots() == 9000

    def test_cost_budget_enforced_when_set(self):
        b = HardwareDayBudget(max_jobs=5, max_shots_per_job=4096, max_total_shots=20000, max_qpu_seconds=600, max_usd_cost=10.0)
        t = BudgetTracker(b)
        ok, reason = t.can_submit(100, estimated_usd_cost=11.0)
        assert not ok and reason == RefusalReason.OVER_COST_BUDGET


class TestRecordSubmissionRejectsNegativeInputs:
    """Regression tests for independent-review finding P2-8 ('No
    non-negative validation on budget inputs; demonstrated to silently
    disable the "hard cap"'). Independently reproduced before this fix: a
    negative shots/qpu_seconds/usd_cost was accepted with no error and
    drove the running totals permanently negative, after which legitimate
    submissions could keep succeeding indefinitely since the cumulative
    total never climbed back above the cap."""

    def _budget(self) -> HardwareDayBudget:
        return HardwareDayBudget(max_jobs=5, max_shots_per_job=4096, max_total_shots=10000, max_qpu_seconds=600)

    def test_rejects_negative_shots(self):
        t = BudgetTracker(self._budget())
        with pytest.raises(ValueError):
            t.record_submission(-5000)

    def test_rejects_zero_shots(self):
        t = BudgetTracker(self._budget())
        with pytest.raises(ValueError):
            t.record_submission(0)

    def test_rejects_negative_qpu_seconds(self):
        t = BudgetTracker(self._budget())
        with pytest.raises(ValueError):
            t.record_submission(100, qpu_seconds=-10.0)

    def test_rejects_negative_usd_cost(self):
        t = BudgetTracker(self._budget())
        with pytest.raises(ValueError):
            t.record_submission(100, usd_cost=-5.0)

    def test_rejection_leaves_totals_unchanged(self):
        t = BudgetTracker(self._budget())
        t.record_submission(1000)
        with pytest.raises(ValueError):
            t.record_submission(-5000)
        assert t.total_shots_submitted == 1000  # unaffected by the rejected call
        assert t.jobs_submitted == 1


class TestBackendSelection:
    def _candidates(self):
        return [
            BackendCandidate(name="sim", kind="simulator", num_qubits=32, pending_jobs=0),
            BackendCandidate(name="fake_kingston", kind="fake", num_qubits=27, pending_jobs=0),
            BackendCandidate(name="ibm_kingston", kind="qpu", num_qubits=27, pending_jobs=20, median_two_qubit_error=0.008),
            BackendCandidate(name="ibm_busy", kind="qpu", num_qubits=27, pending_jobs=5, median_two_qubit_error=0.008),
            BackendCandidate(name="ibm_noisy", kind="qpu", num_qubits=27, pending_jobs=1, median_two_qubit_error=0.05),
        ]

    def test_refuses_without_credentials_for_qpu_required(self):
        criteria = BackendSelectionCriteria(min_qubits_required=20, require_credentials_present=True)
        chosen, reason = select_backend(self._candidates(), criteria, credentials_present=False)
        assert chosen is None
        assert reason == RefusalReason.NO_CREDENTIALS

    def test_picks_least_busy_eligible_qpu_when_credentials_present(self):
        criteria = BackendSelectionCriteria(min_qubits_required=20, require_credentials_present=True)
        chosen, reason = select_backend(self._candidates(), criteria, credentials_present=True)
        assert reason is None
        # among eligible QPUs (ibm_kingston pending=20, ibm_busy pending=5;
        # ibm_noisy excluded for error rate; sim/fake also eligible with
        # pending=0) the global minimum pending_jobs wins
        assert chosen is not None
        assert chosen.pending_jobs == 0

    def test_excludes_backend_over_error_threshold(self):
        criteria = BackendSelectionCriteria(
            min_qubits_required=20, max_acceptable_two_qubit_error=0.01, require_credentials_present=False
        )
        candidates = [
            BackendCandidate(name="only_option", kind="qpu", num_qubits=27, pending_jobs=0, median_two_qubit_error=0.05)
        ]
        chosen, reason = select_backend(candidates, criteria, credentials_present=True)
        assert chosen is None
        assert reason == RefusalReason.NO_ELIGIBLE_BACKEND

    def test_excludes_backend_with_too_few_qubits(self):
        # every fixture candidate tops out at 32 qubits (the simulator) --
        # requiring more than that must exclude everything, including the
        # simulator, and refuse.
        criteria = BackendSelectionCriteria(min_qubits_required=40, require_credentials_present=False)
        chosen, reason = select_backend(self._candidates(), criteria, credentials_present=True)
        assert chosen is None
        assert reason == RefusalReason.NO_ELIGIBLE_BACKEND

    def test_never_selects_qpu_without_credentials_even_if_required_false(self):
        criteria = BackendSelectionCriteria(min_qubits_required=20, require_credentials_present=False)
        chosen, reason = select_backend(self._candidates(), criteria, credentials_present=False)
        assert reason is None
        assert chosen is not None
        assert chosen.kind != "qpu"


class TestBackendCandidateKindValidation:
    """Regression tests for independent-review finding P2-1 ('"never
    selects a QPU-kind backend without credentials" guarantee is a
    case-sensitive string check, demonstrated bypassable'). Independently
    reproduced before this fix: `BackendCandidate(kind="QPU", ...)`
    (mis-cased) was accepted with zero validation, and
    `select_backend([that], BackendSelectionCriteria(...,
    require_credentials_present=False), credentials_present=False)`
    selected it -- the per-candidate `c.kind != "qpu"` filter silently
    failed to recognize the mis-cased string. Fixed by validating `kind`
    at construction instead, so the bad candidate can never exist in the
    first place."""

    def test_rejects_miscased_kind_at_construction(self):
        with pytest.raises(ValueError):
            BackendCandidate(name="fake-qpu", kind="QPU", num_qubits=27)

    def test_rejects_unknown_kind_at_construction(self):
        with pytest.raises(ValueError):
            BackendCandidate(name="x", kind="bogus", num_qubits=5)

    def test_accepts_every_valid_kind(self):
        for kind in ("simulator", "fake", "qpu"):
            BackendCandidate(name="x", kind=kind, num_qubits=5)  # must not raise

    def test_a_miscased_candidate_can_no_longer_reach_select_backend_at_all(self):
        """The actual P2-1 reproduction scenario -- now impossible, because
        construction itself refuses the bad candidate before select_backend
        is ever called."""
        with pytest.raises(ValueError):
            bad = BackendCandidate(name="fake-qpu", kind="QPU", num_qubits=27)
            criteria = BackendSelectionCriteria(min_qubits_required=5, require_credentials_present=False)
            select_backend([bad], criteria, credentials_present=False)


class TestRetryPolicy:
    def test_non_retryable_names_frozen(self):
        assert "EvidenceIntegrityError" in NON_RETRYABLE_EXCEPTION_NAMES
        assert "CredentialError" in NON_RETRYABLE_EXCEPTION_NAMES

    def test_rejects_invalid_policy(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_retries_transient_failure_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        sleeps = []
        result = run_with_retry(flaky, RetryPolicy(max_attempts=5, backoff_seconds=1.0), sleep=sleeps.append)
        assert result == "ok"
        assert calls["n"] == 3
        assert len(sleeps) == 2  # slept before attempt 2 and 3, not after success

    def test_gives_up_after_max_attempts(self):
        def always_fails():
            raise ConnectionError("still transient")

        with pytest.raises(ConnectionError):
            run_with_retry(always_fails, RetryPolicy(max_attempts=3), sleep=lambda _s: None)

    def test_non_retryable_exception_raised_immediately_no_retry(self):
        from core.errors import EvidenceIntegrityError

        calls = {"n": 0}

        def fails_non_retryably():
            calls["n"] += 1
            raise EvidenceIntegrityError("claim does not check out")

        with pytest.raises(EvidenceIntegrityError):
            run_with_retry(fails_non_retryably, RetryPolicy(max_attempts=5), sleep=lambda _s: None)
        assert calls["n"] == 1  # never retried
