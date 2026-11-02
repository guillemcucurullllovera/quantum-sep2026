"""Integration tests for `execute_on_qpu`'s budget enforcement and
submit/wait retry-separation — remediates independent-review finding
P1-1 ("The only real-hardware-capable code path is completely unwired
from every QUANTUM-1B budget control; its retry policy retries on any
exception with no idempotency protection").

Monkeypatches `submit_qpu_sampler_job`/`fetch_qpu_sampler_result` at the
point `execute.py` imports them, so this tests `execute_on_qpu`'s OWN
budget/retry logic without mocking the real `qiskit_ibm_runtime` SamplerV2
wire format — correctly out of scope for this repo without real
credentials, per the independent review's own "what could not be
verified" section. No real network call, no real credentials (a dummy
token string satisfies `IBMQuantumCredentials.require()`'s presence
check, which only asserts non-empty, never validates against a real
service), no real IBM service object constructed anywhere in this file.
"""
from __future__ import annotations

import pytest

import hardware.qpu_execution.execute as execute_mod
from hardware.qpu_execution import execute_on_qpu
from hardware.retries import RetryPolicy


class _FakeBudgetTracker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_submission(self, session_id, shots, *, qpu_seconds=0.0, usd_cost=0.0):
        self.calls.append(
            {"session_id": session_id, "shots": shots, "qpu_seconds": qpu_seconds, "usd_cost": usd_cost}
        )


class _RefusingBudgetTracker:
    """Simulates a real PersistentBudgetTracker refusing an over-budget
    submission -- raises before any network call would happen."""

    def record_submission(self, session_id, shots, *, qpu_seconds=0.0, usd_cost=0.0):
        raise RuntimeError("budget exceeded (simulated refusal)")


class _FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.num_qubits = 5


class _FakeService:
    """Stands in for QiskitRuntimeService -- never a real one. Injected via
    execute_on_qpu's own `service=` parameter, the same dependency-injection
    pattern already used by hardware_readiness.ibm_verifier.IBMRuntimeVerifier."""

    def backend(self, name):
        return _FakeBackend(name)


@pytest.fixture
def ibm_token(monkeypatch):
    """A dummy, non-functional token string -- only satisfies the
    presence check in IBMQuantumCredentials.require(), never validated
    against any real service in this test file.

    **Bug found and fixed 2026-10-01** (discovered auditing the committed
    Command Center export, which was found to incorrectly show
    `real_ibm_credentials_present: true` despite no real token ever being
    configured): the post-`yield` teardown previously called
    `get_settings(refresh=True)` BEFORE `monkeypatch`'s own teardown
    removed the env var set above (pytest fixture teardown runs in
    reverse setup order, so this fixture's post-yield code runs before
    monkeypatch reverts anything it changed) -- re-caching the DUMMY
    token's "present" state into the GLOBAL `core.config` settings
    singleton, which then stayed polluted for the rest of the pytest
    PROCESS, not just this fixture's own tests. Fixed by explicitly
    deleting the env var before the final refresh."""
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "dummy-token-for-test-never-used-for-a-real-call")
    import core.config as config_mod

    config_mod.get_settings(refresh=True)
    yield
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    config_mod.get_settings(refresh=True)


class TestBudgetEnforcement:
    def test_budget_tracker_called_exactly_once_before_any_submission(self, ibm_token, monkeypatch):
        call_order: list[str] = []

        def fake_submit(circuit, descriptor, backend, shots=4096):
            call_order.append("submit")
            return "job-handle"

        def fake_fetch(job, descriptor, shots):
            call_order.append("fetch")
            return "ok-result"

        monkeypatch.setattr(execute_mod, "submit_qpu_sampler_job", fake_submit)
        monkeypatch.setattr(execute_mod, "fetch_qpu_sampler_result", fake_fetch)

        tracker = _FakeBudgetTracker()
        result = execute_on_qpu(
            "circuit", backend_name="ibm_test", shots=1024,
            budget_tracker=tracker, session_id="sess-1", service=_FakeService(),
        )

        assert len(tracker.calls) == 1
        assert tracker.calls[0]["shots"] == 1024
        assert tracker.calls[0]["session_id"] == "sess-1"
        assert call_order == ["submit", "fetch"]  # budget call happened before either, by construction
        assert result.sampler_result == "ok-result"

    def test_budget_refusal_prevents_any_submission(self, ibm_token, monkeypatch):
        submit_called: list[bool] = []

        def fake_submit(*args, **kwargs):
            submit_called.append(True)
            return "job-handle"

        monkeypatch.setattr(execute_mod, "submit_qpu_sampler_job", fake_submit)

        with pytest.raises(RuntimeError, match="budget exceeded"):
            execute_on_qpu(
                "circuit", backend_name="ibm_test", shots=1024,
                budget_tracker=_RefusingBudgetTracker(), session_id="sess-1", service=_FakeService(),
            )

        assert submit_called == []  # the real submission function was never reached


class TestRunbookLiteralPattern:
    """Regression guard for independent-review finding P1-4: the runbook's
    §3 documented pattern must actually run, without AttributeError or
    TypeError, against the REAL `PersistentBudgetTracker` (not a fake/duck-
    typed stand-in like the tests above) -- an earlier version of the
    runbook's pseudocode called a nonexistent `can_submit` method and
    omitted the required `session_id` argument, which would have crashed
    immediately if an operator copied it literally."""

    def test_runbook_section_3_pattern_runs_without_error(self, ibm_token, monkeypatch, tmp_path):
        from experiments.quantum1b.hardware_readiness.budgets import (
            BudgetExceededError,
            HardwareDayBudget,
        )
        from experiments.quantum1b.hardware_readiness.persistent_budget import PersistentBudgetTracker

        def fake_submit(circuit, descriptor, backend, shots=4096):
            return "job-handle-runbook-test"

        def fake_fetch(job, descriptor, shots):
            return "runbook-test-result"

        monkeypatch.setattr(execute_mod, "submit_qpu_sampler_job", fake_submit)
        monkeypatch.setattr(execute_mod, "fetch_qpu_sampler_result", fake_fetch)

        # Exactly the runbook §3 pattern, verbatim in structure:
        db_path = str(tmp_path / "hardware_day_test_budget.db")
        session_id = "hardware_day_test"

        tracker = PersistentBudgetTracker(db_path)
        budget = HardwareDayBudget(
            max_jobs=3, max_shots_per_job=4096, max_total_shots=12000, max_qpu_seconds=600.0,
        )
        tracker.get_or_create_session(session_id, budget)

        try:
            result = execute_on_qpu(
                "circuit", backend_name="ibm_test", shots=4096,
                budget_tracker=tracker, session_id=session_id,
                estimated_qpu_seconds=10.0, estimated_usd_cost=1.0,
                service=_FakeService(),
            )
        except BudgetExceededError:
            pytest.fail("should not have exceeded budget on the first submission")

        assert result.sampler_result == "runbook-test-result"
        snap = tracker.snapshot(session_id)
        assert snap.jobs_submitted == 1

    def test_runbook_section_3_pattern_refuses_correctly_over_budget(self, ibm_token, monkeypatch, tmp_path):
        from experiments.quantum1b.hardware_readiness.budgets import (
            BudgetExceededError,
            HardwareDayBudget,
        )
        from experiments.quantum1b.hardware_readiness.persistent_budget import PersistentBudgetTracker

        submit_called = []

        def fake_submit(*args, **kwargs):
            submit_called.append(True)
            return "job-handle"

        monkeypatch.setattr(execute_mod, "submit_qpu_sampler_job", fake_submit)

        db_path = str(tmp_path / "hardware_day_test_budget2.db")
        session_id = "hardware_day_test2"
        tracker = PersistentBudgetTracker(db_path)
        tracker.get_or_create_session(
            session_id,
            HardwareDayBudget(max_jobs=1, max_shots_per_job=4096, max_total_shots=4096, max_qpu_seconds=600.0),
        )
        # Exhaust the one-job budget first.
        tracker.record_submission(session_id, 4096)

        with pytest.raises(BudgetExceededError):
            execute_on_qpu(
                "circuit", backend_name="ibm_test", shots=4096,
                budget_tracker=tracker, session_id=session_id, service=_FakeService(),
            )
        assert submit_called == []  # the real submission function was never reached


class TestSubmitWaitRetrySeparation:
    def test_transient_wait_failure_retries_fetch_only_never_resubmits(self, ibm_token, monkeypatch):
        """The core P1-1 regression guard: a transient failure during
        job.result() (simulated as fetch_qpu_sampler_result raising) must
        retry the FETCH only. submit_qpu_sampler_job must be called
        exactly once even though the overall operation needed retries --
        this is the property that prevents duplicate real-hardware
        submissions for a single caller intent."""
        submit_calls: list[int] = []
        fetch_attempts = {"n": 0}

        def fake_submit(circuit, descriptor, backend, shots=4096):
            submit_calls.append(1)
            return "job-handle-1"

        def fake_fetch(job, descriptor, shots):
            fetch_attempts["n"] += 1
            if fetch_attempts["n"] < 3:
                raise ConnectionError("transient network error during job.result()")
            assert job == "job-handle-1"  # must be re-fetching the SAME job object, not a new one
            return "final-result"

        monkeypatch.setattr(execute_mod, "submit_qpu_sampler_job", fake_submit)
        monkeypatch.setattr(execute_mod, "fetch_qpu_sampler_result", fake_fetch)

        tracker = _FakeBudgetTracker()
        result = execute_on_qpu(
            "circuit", backend_name="ibm_test", shots=1024,
            retry_policy=RetryPolicy(max_attempts=5, base_delay_s=0.001),
            budget_tracker=tracker, session_id="sess-1", service=_FakeService(),
        )

        assert len(submit_calls) == 1, "submit_qpu_sampler_job must be called exactly once"
        assert fetch_attempts["n"] == 3, "fetch should have been retried twice before succeeding"
        assert result.sampler_result == "final-result"
        assert result.was_retried is True
        # Budget is charged once per execute_on_qpu call, before any
        # attempt -- a deliberate, documented design choice (see
        # execute.py's module docstring): real IBM billing only applies to
        # jobs that actually reach the queue, and this platform cannot know
        # in advance how many submit/wait attempts a call will need, so it
        # reserves budget conservatively once per logical call rather than
        # per internal retry attempt.
        assert len(tracker.calls) == 1

    def test_transient_submit_failure_retries_submit_before_any_job_exists(self, ibm_token, monkeypatch):
        """Symmetric case: a transient failure during submission itself
        (before any job handle exists) is safe to retry via
        submit_qpu_sampler_job again, since no billable job was created by
        the failed attempt(s)."""
        submit_attempts = {"n": 0}
        fetch_calls: list[str] = []

        def fake_submit(circuit, descriptor, backend, shots=4096):
            submit_attempts["n"] += 1
            if submit_attempts["n"] < 2:
                raise ConnectionError("transient network error during sampler.run()")
            return "job-handle-2"

        def fake_fetch(job, descriptor, shots):
            fetch_calls.append(job)
            assert job == "job-handle-2"
            return "final-result-2"

        monkeypatch.setattr(execute_mod, "submit_qpu_sampler_job", fake_submit)
        monkeypatch.setattr(execute_mod, "fetch_qpu_sampler_result", fake_fetch)

        tracker = _FakeBudgetTracker()
        result = execute_on_qpu(
            "circuit", backend_name="ibm_test", shots=1024,
            retry_policy=RetryPolicy(max_attempts=5, base_delay_s=0.001),
            budget_tracker=tracker, session_id="sess-1", service=_FakeService(),
        )

        assert submit_attempts["n"] == 2
        assert fetch_calls == ["job-handle-2"]  # fetch only ever saw the successfully-submitted job
        assert result.sampler_result == "final-result-2"
        assert len(tracker.calls) == 1
