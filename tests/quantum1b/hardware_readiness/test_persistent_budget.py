"""Tests for `experiments.quantum1b.hardware_readiness.persistent_budget`."""
from __future__ import annotations

import multiprocessing
import threading

import pytest

from experiments.quantum1b.hardware_readiness.budgets import BudgetExceededError, HardwareDayBudget
from experiments.quantum1b.hardware_readiness.persistent_budget import (
    PersistentBudgetTracker,
    SessionBudgetMismatchError,
)


def _budget() -> HardwareDayBudget:
    return HardwareDayBudget(max_jobs=5, max_shots_per_job=4096, max_total_shots=10000, max_qpu_seconds=600)


class TestBasics:
    def test_get_or_create_starts_at_zero(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        snap = tracker.get_or_create_session("sess1", _budget())
        assert snap.jobs_submitted == 0
        assert snap.total_shots_submitted == 0

    def test_get_or_create_is_idempotent(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        tracker.record_submission("sess1", 1000)
        snap2 = tracker.get_or_create_session("sess1", _budget())
        assert snap2.jobs_submitted == 1  # existing session, not reset

    def test_record_submission_updates_totals(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        snap = tracker.record_submission("sess1", 1000, qpu_seconds=5, usd_cost=2)
        assert snap.jobs_submitted == 1
        assert snap.total_shots_submitted == 1000
        assert snap.qpu_seconds_used == 5
        assert snap.usd_cost_used == 2

    def test_refuses_over_budget_without_partial_update(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        with pytest.raises(BudgetExceededError):
            tracker.record_submission("sess1", 999999)
        snap = tracker.snapshot("sess1")
        assert snap.jobs_submitted == 0  # refused submission left totals untouched
        assert snap.total_shots_submitted == 0

    def test_unknown_session_raises_key_error(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        with pytest.raises(KeyError):
            tracker.record_submission("never-created", 100)


class TestRecordSubmissionRejectsNegativeInputs:
    """Regression tests for independent-review finding P2-8, same defect as
    `budgets.BudgetTracker.record_submission` -- independently reproduced
    against this class too before the fix: a negative shots/qpu_seconds/
    usd_cost was accepted with no error and drove the stored running totals
    permanently negative."""

    def test_rejects_negative_shots(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        with pytest.raises(ValueError):
            tracker.record_submission("sess1", -5000)

    def test_rejects_negative_qpu_seconds(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        with pytest.raises(ValueError):
            tracker.record_submission("sess1", 100, qpu_seconds=-10.0)

    def test_rejects_negative_usd_cost(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        with pytest.raises(ValueError):
            tracker.record_submission("sess1", 100, usd_cost=-5.0)

    def test_rejection_leaves_totals_unchanged(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        tracker.record_submission("sess1", 1000)
        with pytest.raises(ValueError):
            tracker.record_submission("sess1", -5000)
        snap = tracker.snapshot("sess1")
        assert snap.total_shots_submitted == 1000


class TestGetOrCreateSessionRejectsMismatchedBudget:
    """Regression tests for independent-review finding P3-9
    ('get_or_create_session silently ignores a mismatched budget argument
    for an already-existing session'). Before this fix, calling
    get_or_create_session a second time with a DIFFERENT budget for the
    same session_id silently kept the original budget, with no error or
    log -- a caller could believe it changed the budget and be wrong."""

    def test_same_budget_twice_is_fine(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        b = _budget()
        tracker.get_or_create_session("sess1", b)
        # A separately-constructed but field-equal budget must also be fine.
        tracker.get_or_create_session("sess1", _budget())  # must not raise

    def test_different_budget_for_existing_session_raises(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        tracker.get_or_create_session("sess1", _budget())
        different = HardwareDayBudget(
            max_jobs=99, max_shots_per_job=4096, max_total_shots=10000, max_qpu_seconds=600
        )
        with pytest.raises(SessionBudgetMismatchError):
            tracker.get_or_create_session("sess1", different)

    def test_rejected_call_does_not_change_the_stored_budget(self, tmp_path):
        tracker = PersistentBudgetTracker(tmp_path / "budget.db")
        original = _budget()
        tracker.get_or_create_session("sess1", original)
        different = HardwareDayBudget(
            max_jobs=99, max_shots_per_job=4096, max_total_shots=10000, max_qpu_seconds=600
        )
        with pytest.raises(SessionBudgetMismatchError):
            tracker.get_or_create_session("sess1", different)
        snap = tracker.snapshot("sess1")
        assert snap.budget == original  # unchanged, not silently swapped to `different`


class TestPersistsAcrossReopens:
    def test_survives_reopening_the_tracker(self, tmp_path):
        db_path = tmp_path / "budget.db"
        PersistentBudgetTracker(db_path).get_or_create_session("sess1", _budget())
        PersistentBudgetTracker(db_path).record_submission("sess1", 500)

        reopened = PersistentBudgetTracker(db_path)
        snap = reopened.snapshot("sess1")
        assert snap is not None
        assert snap.total_shots_submitted == 500


class TestConcurrency:
    def test_many_threads_never_jointly_overspend_the_job_cap(self, tmp_path):
        """20 threads each try to record one submission against a budget
        with max_jobs=5 -- exactly 5 must succeed, the rest must be
        refused, and the final state must show exactly 5 jobs_submitted.
        This is the actual property PersistentBudgetTracker exists for:
        an in-process-only BudgetTracker could not make this guarantee
        across multiple processes; this test uses threads (which still
        drive genuinely concurrent sqlite3 connections against the same
        file, exercising the same BEGIN IMMEDIATE locking a multi-process
        Hardware Day session would rely on) to keep the test fast and
        within this session's resource discipline."""
        db_path = tmp_path / "budget.db"
        tracker = PersistentBudgetTracker(db_path)
        budget = HardwareDayBudget(max_jobs=5, max_shots_per_job=100, max_total_shots=100000, max_qpu_seconds=10000)
        tracker.get_or_create_session("race", budget)

        n_threads = 20
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            t = PersistentBudgetTracker(db_path)  # separate connection per thread, like separate processes would have
            try:
                t.record_submission("race", 10)
                with lock:
                    outcomes.append("ok")
            except BudgetExceededError:
                with lock:
                    outcomes.append("refused")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30)

        assert outcomes.count("ok") == 5
        assert outcomes.count("refused") == 15
        final = tracker.snapshot("race")
        assert final.jobs_submitted == 5
        assert final.total_shots_submitted == 50


def _subprocess_submit(db_path: str, session_id: str, results_queue) -> None:
    """Top-level (picklable) target for a real OS-process worker."""
    from experiments.quantum1b.hardware_readiness.budgets import BudgetExceededError as _BEE
    from experiments.quantum1b.hardware_readiness.persistent_budget import PersistentBudgetTracker as _PBT

    tracker = _PBT(db_path)
    try:
        tracker.record_submission(session_id, 10)
        results_queue.put("ok")
    except _BEE:
        results_queue.put("refused")


class TestRealMultiProcessConcurrency:
    def test_real_separate_processes_never_jointly_overspend(self, tmp_path):
        """The property TestConcurrency proves with threads, reproduced
        with GENUINE separate OS processes (bounded to a small, fixed
        count -- this session's resource discipline: no uncontrolled
        worker pools) -- this is the actual deployment shape
        PersistentBudgetTracker is for."""
        db_path = str(tmp_path / "budget.db")
        budget = HardwareDayBudget(max_jobs=3, max_shots_per_job=100, max_total_shots=100000, max_qpu_seconds=10000)
        PersistentBudgetTracker(db_path).get_or_create_session("mp-race", budget)

        ctx = multiprocessing.get_context("spawn")
        results_queue = ctx.Queue()
        n_procs = 6  # small, fixed, bounded -- not an uncontrolled pool
        procs = [
            ctx.Process(target=_subprocess_submit, args=(db_path, "mp-race", results_queue))
            for _ in range(n_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)

        outcomes = [results_queue.get(timeout=5) for _ in range(n_procs)]
        assert outcomes.count("ok") == 3
        assert outcomes.count("refused") == 3
        final = PersistentBudgetTracker(db_path).snapshot("mp-race")
        assert final.jobs_submitted == 3


def _subprocess_get_or_create(db_path: str, session_id: str, results_queue) -> None:
    """Top-level (picklable) target for a real OS-process worker racing to
    CREATE (not just use) the same new session -- the exact P1-3
    reproduction scenario."""
    from experiments.quantum1b.hardware_readiness.budgets import HardwareDayBudget as _HDB
    from experiments.quantum1b.hardware_readiness.persistent_budget import PersistentBudgetTracker as _PBT

    tracker = _PBT(db_path)
    budget = _HDB(max_jobs=5, max_shots_per_job=4096, max_total_shots=10000, max_qpu_seconds=600)
    try:
        snap = tracker.get_or_create_session(session_id, budget)
        results_queue.put(("ok", snap.jobs_submitted))
    except Exception as exc:  # noqa: BLE001 -- must capture ANY exception to prove none occurs
        results_queue.put(("error", f"{type(exc).__name__}: {exc}"))


class TestGetOrCreateSessionConcurrency:
    """Regression test for independent-review finding P1-3
    ('PersistentBudgetTracker.get_or_create_session() is not protected
    against concurrent first-time session creation'). Exact reproduction
    shape the review specified: >=8 real OS processes racing to create the
    SAME NEW session_id against a fresh db path. Before the fix, this
    reliably raised sqlite3.IntegrityError in multiple processes; after
    the fix, every process must return a consistent BudgetSnapshot with
    jobs_submitted=0 and zero exceptions."""

    def test_many_processes_racing_to_create_the_same_new_session(self, tmp_path):
        db_path = str(tmp_path / "budget.db")
        # Deliberately do NOT call get_or_create_session or
        # PersistentBudgetTracker(db_path) first in this process beyond
        # constructing the schema file -- the race is specifically about
        # FIRST-TIME creation, so the db file must not already contain the
        # session row before the race starts. Constructing a tracker here
        # only creates the empty schema (budget_sessions table, no rows),
        # which is required for any process to connect at all.
        PersistentBudgetTracker(db_path)

        ctx = multiprocessing.get_context("spawn")
        results_queue = ctx.Queue()
        n_procs = 12  # matches the review's own reproduction scale
        procs = [
            ctx.Process(target=_subprocess_get_or_create, args=(db_path, "new-session-race", results_queue))
            for _ in range(n_procs)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)

        outcomes = [results_queue.get(timeout=5) for _ in range(n_procs)]
        errors = [o for o in outcomes if o[0] == "error"]
        assert not errors, f"get_or_create_session raised in {len(errors)}/{n_procs} processes: {errors}"
        assert all(o == ("ok", 0) for o in outcomes), outcomes

        # Exactly one row must exist for this session, with no corruption.
        final = PersistentBudgetTracker(db_path).snapshot("new-session-race")
        assert final is not None
        assert final.jobs_submitted == 0
        assert final.budget.max_jobs == 5

    def test_many_threads_racing_to_create_the_same_new_session(self, tmp_path):
        """Thread variant of the same race -- faster, complements the
        real-process test above (both techniques were used independently
        by different review passes)."""
        db_path = tmp_path / "budget_threads.db"
        PersistentBudgetTracker(db_path)
        budget = _budget()

        n_threads = 20
        outcomes: list[tuple] = []
        lock = threading.Lock()

        def worker() -> None:
            tracker = PersistentBudgetTracker(db_path)
            try:
                snap = tracker.get_or_create_session("thread-race-session", budget)
                with lock:
                    outcomes.append(("ok", snap.jobs_submitted))
            except Exception as exc:  # noqa: BLE001
                with lock:
                    outcomes.append(("error", f"{type(exc).__name__}: {exc}"))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        errors = [o for o in outcomes if o[0] == "error"]
        assert not errors, f"get_or_create_session raised in {len(errors)}/{n_threads} threads: {errors}"
        assert all(o == ("ok", 0) for o in outcomes)

        final = PersistentBudgetTracker(db_path).snapshot("thread-race-session")
        assert final is not None
        assert final.jobs_submitted == 0
