"""Persistent (SQLite-backed) `HardwareDayBudget` accounting.

Closes the disclosed gap in `budgets.py`'s own docstring/capstone record:
`BudgetTracker` is in-process-only, so a session spanning a process
restart (or, more importantly, a Hardware Day run coordinated by more than
one process/script) would lose its accounting, or worse, two independent
in-process trackers could each think they have the full budget available
and jointly overspend it.

**Design: atomic check-and-increment via `BEGIN IMMEDIATE`.** SQLite's
`BEGIN IMMEDIATE` acquires a write lock at the START of the transaction
(not lazily at the first write), so two processes racing to
`record_submission()` are genuinely serialized -- the second one's
`BEGIN IMMEDIATE` blocks until the first commits, then re-reads the
now-updated totals before deciding whether it still fits the budget. This
is the same atomicity property a real multi-process Hardware Day session
needs and `budgets.BudgetTracker` (correctly, for its narrower in-process
scope) does not provide.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from experiments.quantum1b.hardware_readiness.budgets import (
    BudgetExceededError,
    HardwareDayBudget,
    RefusalReason,
)


class SessionBudgetMismatchError(Exception):
    """Raised by `get_or_create_session()` when called for an EXISTING
    session_id with a DIFFERENT `HardwareDayBudget` than the one already on
    record. **Added 2026-10-01, remediating independent-review finding
    P3-9** ("get_or_create_session silently ignores a mismatched budget
    argument for an already-existing session") -- before this fix, no
    comparison was made at all: a caller could believe it changed a
    session's budget by calling `get_or_create_session` with a new value,
    and it would silently do nothing, the session keeping its ORIGINAL
    budget. This is the same "a stored record silently became something
    different without an error" problem `evidence_store.py`'s
    `AttestationTamperedError` exists to catch for attestations -- now
    consistently applied here too."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_sessions (
    session_id TEXT PRIMARY KEY,
    max_jobs INTEGER NOT NULL,
    max_shots_per_job INTEGER NOT NULL,
    max_total_shots INTEGER NOT NULL,
    max_qpu_seconds REAL NOT NULL,
    max_usd_cost REAL,
    jobs_submitted INTEGER NOT NULL DEFAULT 0,
    total_shots_submitted INTEGER NOT NULL DEFAULT 0,
    qpu_seconds_used REAL NOT NULL DEFAULT 0.0,
    usd_cost_used REAL NOT NULL DEFAULT 0.0
);
"""


@dataclass(frozen=True)
class BudgetSnapshot:
    session_id: str
    budget: HardwareDayBudget
    jobs_submitted: int
    total_shots_submitted: int
    qpu_seconds_used: float
    usd_cost_used: float

    def remaining_jobs(self) -> int:
        return max(0, self.budget.max_jobs - self.jobs_submitted)

    def remaining_shots(self) -> int:
        return max(0, self.budget.max_total_shots - self.total_shots_submitted)


class PersistentBudgetTracker:
    """SQLite-backed, multi-process-safe `HardwareDayBudget` accounting
    for one named session. A session is created once (first
    `get_or_create_session`) and every subsequent process pointed at the
    same db path + session_id shares the same running totals."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_or_create_session(self, session_id: str, budget: HardwareDayBudget) -> BudgetSnapshot:
        """Get the existing session, or atomically create it if this is the
        first caller to see it.

        **Fixed 2026-10-01, remediating independent-review finding P1-3**
        ("not protected against concurrent first-time session creation,
        contradicting its own 'multi-process-safe' claim"). The original
        version used a bare `SELECT` then, if no row existed, a bare
        `INSERT` with no transaction isolation — two processes racing to
        create the same new session could both observe "no row yet" and
        both attempt the INSERT, reliably producing an uncaught
        `sqlite3.IntegrityError` (independently reproduced by three review
        passes: 12-process and 20-thread races, plus direct code
        inspection). Fixed with the same `BEGIN IMMEDIATE` + reconcile-on-
        `IntegrityError` pattern already used by `record_submission()`
        below and by `evidence_store.py::PersistentEvidenceStore.put()` in
        this same package — a losing process re-fetches and returns the
        winner's row instead of crashing.

        **Raises `SessionBudgetMismatchError` if called for an EXISTING
        session with a DIFFERENT budget (fixed 2026-10-01, remediating
        P3-9)** -- a caller passing a different `HardwareDayBudget` for an
        already-created session almost certainly has a bug (two different
        parts of a session disagreeing about the budget), and silently
        keeping the original is exactly the kind of "nothing visibly
        happened" failure mode this platform's error-handling discipline
        otherwise refuses to allow."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT max_jobs, max_shots_per_job, max_total_shots, max_qpu_seconds, "
                "max_usd_cost, jobs_submitted, total_shots_submitted, qpu_seconds_used, "
                "usd_cost_used FROM budget_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                conn.rollback()
                existing_snap = self._row_to_snapshot(session_id, row)
                if existing_snap.budget != budget:
                    raise SessionBudgetMismatchError(
                        f"session {session_id!r} already exists with a DIFFERENT budget.\n"
                        f"  existing={existing_snap.budget!r}\n  requested={budget!r}\n"
                        "get_or_create_session() never silently changes an existing session's "
                        "budget -- construct a new session_id if a different budget is genuinely intended."
                    )
                return existing_snap
            try:
                conn.execute(
                    "INSERT INTO budget_sessions (session_id, max_jobs, max_shots_per_job, "
                    "max_total_shots, max_qpu_seconds, max_usd_cost) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        budget.max_jobs,
                        budget.max_shots_per_job,
                        budget.max_total_shots,
                        budget.max_qpu_seconds,
                        budget.max_usd_cost,
                    ),
                )
                conn.commit()
                return BudgetSnapshot(session_id, budget, 0, 0, 0.0, 0.0)
            except sqlite3.IntegrityError:
                # Lost a race: another process committed this session_id
                # between our SELECT and our INSERT (can happen even under
                # BEGIN IMMEDIATE if that other process's own transaction
                # was already in flight). Re-fetch and return the winner's
                # row rather than propagating the crash.
                conn.rollback()
                reread = conn.execute(
                    "SELECT max_jobs, max_shots_per_job, max_total_shots, max_qpu_seconds, "
                    "max_usd_cost, jobs_submitted, total_shots_submitted, qpu_seconds_used, "
                    "usd_cost_used FROM budget_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if reread is None:
                    raise  # genuinely unexpected: INTEGRITY error but no row exists -- do not swallow
                winner_snap = self._row_to_snapshot(session_id, reread)
                if winner_snap.budget != budget:
                    raise SessionBudgetMismatchError(
                        f"session {session_id!r}: lost a creation race, and the winner's budget "
                        f"differs from ours.\n  winner={winner_snap.budget!r}\n  ours={budget!r}\n"
                        "Two processes raced to create the same session_id with DIFFERENT budgets "
                        "-- this is a real configuration disagreement, not a benign race."
                    )
                return winner_snap
        finally:
            conn.close()

    @staticmethod
    def _row_to_snapshot(session_id: str, row) -> BudgetSnapshot:
        budget = HardwareDayBudget(
            max_jobs=row[0], max_shots_per_job=row[1], max_total_shots=row[2],
            max_qpu_seconds=row[3], max_usd_cost=row[4],
        )
        return BudgetSnapshot(
            session_id=session_id, budget=budget, jobs_submitted=row[5],
            total_shots_submitted=row[6], qpu_seconds_used=row[7], usd_cost_used=row[8],
        )

    def snapshot(self, session_id: str) -> BudgetSnapshot | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT max_jobs, max_shots_per_job, max_total_shots, max_qpu_seconds, "
                "max_usd_cost, jobs_submitted, total_shots_submitted, qpu_seconds_used, "
                "usd_cost_used FROM budget_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return None if row is None else self._row_to_snapshot(session_id, row)

    def record_submission(
        self, session_id: str, shots: int, *, qpu_seconds: float = 0.0, usd_cost: float = 0.0
    ) -> BudgetSnapshot:
        """Atomically check-and-increment, across processes. Raises
        `BudgetExceededError` (never silently truncates or partially
        applies) if the submission would exceed any cap -- the session's
        stored totals are left completely unchanged on refusal.

        **Validates non-negative inputs (fixed 2026-10-01, remediating
        independent-review finding P2-8 -- same defect, same fix, as
        `budgets.BudgetTracker.record_submission`).** Before this fix, a
        negative `shots`/`qpu_seconds`/`usd_cost` was accepted with no
        error and drove the stored running totals permanently negative,
        reproduced directly against this exact class."""
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        if qpu_seconds < 0:
            raise ValueError(f"qpu_seconds must be >= 0, got {qpu_seconds}")
        if usd_cost < 0:
            raise ValueError(f"usd_cost must be >= 0, got {usd_cost}")
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT max_jobs, max_shots_per_job, max_total_shots, max_qpu_seconds, "
                "max_usd_cost, jobs_submitted, total_shots_submitted, qpu_seconds_used, "
                "usd_cost_used FROM budget_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise KeyError(
                    f"no budget session {session_id!r} -- call get_or_create_session() first"
                )
            snap = self._row_to_snapshot(session_id, row)
            reason = self._refusal_reason(snap, shots, qpu_seconds, usd_cost)
            if reason is not None:
                conn.rollback()
                raise BudgetExceededError(
                    f"session {session_id!r}: cannot record submission of {shots} shots: "
                    f"{reason.value} (jobs_submitted={snap.jobs_submitted}/{snap.budget.max_jobs}, "
                    f"total_shots={snap.total_shots_submitted}/{snap.budget.max_total_shots})"
                )
            conn.execute(
                "UPDATE budget_sessions SET jobs_submitted = jobs_submitted + 1, "
                "total_shots_submitted = total_shots_submitted + ?, "
                "qpu_seconds_used = qpu_seconds_used + ?, "
                "usd_cost_used = usd_cost_used + ? WHERE session_id = ?",
                (shots, qpu_seconds, usd_cost, session_id),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        result = self.snapshot(session_id)
        assert result is not None
        return result

    @staticmethod
    def _refusal_reason(
        snap: BudgetSnapshot, shots: int, qpu_seconds: float, usd_cost: float
    ) -> RefusalReason | None:
        b = snap.budget
        if snap.jobs_submitted + 1 > b.max_jobs:
            return RefusalReason.OVER_JOB_BUDGET
        if shots > b.max_shots_per_job:
            return RefusalReason.OVER_SHOT_BUDGET
        if snap.total_shots_submitted + shots > b.max_total_shots:
            return RefusalReason.OVER_SHOT_BUDGET
        if snap.qpu_seconds_used + qpu_seconds > b.max_qpu_seconds:
            return RefusalReason.OVER_QPU_SECONDS_BUDGET
        if b.max_usd_cost is not None and snap.usd_cost_used + usd_cost > b.max_usd_cost:
            return RefusalReason.OVER_COST_BUDGET
        return None


__all__ = ["PersistentBudgetTracker", "BudgetSnapshot", "SessionBudgetMismatchError"]
