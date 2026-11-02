"""Persistent, append-only (immutable) `EvidenceStore` -- fills the exact
gap `core.verification.InMemoryEvidenceStore`'s own docstring flags:
"Replace before relying on this for a real QPU run whose result will be
read back later or by another process."

SQLite-backed, sibling in spirit to `core.experiment_registry.ExperimentRegistry`
but deliberately different in one critical way:

- `ExperimentRegistry.record()` uses `INSERT OR REPLACE` -- correct for
  general experiment bookkeeping (a result can legitimately be re-recorded),
  but WRONG for a hardware-evidence audit trail, where "an attestation
  quietly became a different attestation" is exactly the kind of tamper
  this whole evidence-level system exists to make detectable.
- `PersistentEvidenceStore` is append-only: writing a NEW `attestation_id`
  succeeds; writing the SAME `attestation_id` with identical fields is a
  no-op (idempotent retry-safe); writing the SAME `attestation_id` with
  DIFFERENT fields raises `AttestationTamperedError` rather than silently
  overwriting. An attestation, once recorded through THIS STORE'S OWN API,
  is a historical fact.

**Scope of the tamper-resistance guarantee, corrected 2026-10-01 (remediating
independent-review finding P2-5).** The guarantee above holds only for
writes that go through `put()`. A caller with raw filesystem/SQL access to
the underlying `.db` file can mutate or delete a row directly, bypassing
this class entirely -- independently reproduced: `sqlite3.connect(path)` +
a raw `UPDATE` against an already-`put()` row succeeds silently, and a
subsequent `store.get()` returns the mutated row with no error or warning.
This is an inherent limitation of any application-level API wrapping a
plain SQLite file, not specific to this implementation, but an earlier
version of this docstring did not disclose it.

**What is new in this fix: `verify_integrity()`, a hash chain detecting
exactly this class of out-of-band mutation.** Every row's `record_hash` is
`sha256(prev_row's record_hash + this row's own fields)`, computed and
written atomically at `put()` time (see `put()`'s own docstring for the
concurrency-safety argument). `verify_integrity()` re-walks the chain in
insertion order and reports the first position (if any) where a stored
hash no longer matches what the chain implies -- which a raw `UPDATE`/
`DELETE`/row-reordering will always produce, since it does not (and cannot,
without knowing every subsequent row's content) recompute every hash after
the mutated point. This DETECTS tampering after the fact; it does not
PREVENT it -- true prevention would need external measures (a hash chain
mirrored/signed outside this file, post-session read-only file
permissions, a genuinely append-only/WORM log) that are out of scope for
this pass. For real Hardware Day use, `verify_integrity()` should be run
as a routine check, not assumed to pass silently forever.

Implements the exact minimal `core.verification.EvidenceStore` Protocol
(`.put(attestation)`, `.get(attestation_id)`), so it is a drop-in
replacement for `InMemoryEvidenceStore` anywhere `core.evidence.
compute_evidence_level` or `core.verification.attest()` is called.
`core.verification` itself is not modified -- it is frozen/shared platform
infrastructure per `docs/quantum/QUANTUM1B_PLAN.md` section 0, just
remediated in QUANTUM-1A -- this module only extends it via the Protocol
it already declares.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.verification import VerifiedAttestation

_GENESIS_HASH = "0" * 64


class AttestationAlreadyExistsError(Exception):
    """Raised by a lower-level integrity check if the database's own
    PRIMARY KEY constraint is hit unexpectedly (should not happen in
    practice, since `put()` checks for an existing record first -- kept as
    a defense-in-depth signal, never silently swallowed)."""


class AttestationTamperedError(Exception):
    """Raised when `put()` is called for an EXISTING `attestation_id` with
    different field values than what is already on record. Should never
    happen in normal operation (attestation_ids are content-derived
    hashes -- see `core.verification.MockVerifier`/
    `hardware_readiness.ibm_verifier.IBMRuntimeVerifier`), but checked
    explicitly rather than assumed."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS attestations (
    attestation_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    backend_name TEXT NOT NULL,
    job_id TEXT NOT NULL,
    verified_completion_timestamp TEXT NOT NULL,
    raw_result_hash TEXT NOT NULL,
    verifier_name TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    recorded_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_attestations_job_id ON attestations(job_id);
"""

_COLUMNS = (
    "attestation_id, provider, backend_name, job_id, "
    "verified_completion_timestamp, raw_result_hash, verifier_name"
)


def _compute_record_hash(prev_hash: str, attestation: VerifiedAttestation) -> str:
    payload = {
        "prev_hash": prev_hash,
        "attestation_id": attestation.attestation_id,
        "provider": attestation.provider,
        "backend_name": attestation.backend_name,
        "job_id": attestation.job_id,
        "verified_completion_timestamp": attestation.verified_completion_timestamp,
        "raw_result_hash": attestation.raw_result_hash,
        "verifier_name": attestation.verifier_name,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class IntegrityVerificationResult:
    all_match: bool
    n_records_checked: int
    first_broken_attestation_id: str | None
    """The attestation_id of the FIRST row (in insertion order) whose
    stored `record_hash` does not match what the chain implies -- `None` if
    `all_match` is True. Only the first break is reported, not every row
    after it (every subsequent row's expected hash depends on the broken
    one, so reporting them all would just restate the same single
    tampering event many times)."""


class PersistentEvidenceStore:
    """SQLite-backed, append-only `core.verification.EvidenceStore`
    implementation. Survives process restarts and is readable by a
    different later process (a report generator, a re-run, code review) --
    the exact cross-process durability `InMemoryEvidenceStore` cannot
    provide, per that class's own documented limitation."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            # **Added 2026-10-01, remediating independent-review finding
            # P3-7** ("never explicitly sets PRAGMA synchronous"): WAL mode
            # was already set, but durability previously relied on
            # whatever this environment's COMPILED-IN default happened to
            # be (FULL, incidentally safe) rather than an explicit code
            # guarantee. Evidence-store durability should not depend on an
            # environment default.
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        # busy_timeout (30s) so a writer blocked behind another writer's
        # transaction retries internally instead of raising
        # `sqlite3.OperationalError: database is locked` under genuine
        # concurrent access (see
        # tests/quantum1b/hardware_readiness/test_evidence_store_concurrency.py,
        # which specifically exercises multiple threads writing at once).
        # WAL journal mode (set once, in __init__ above) additionally lets
        # readers proceed without blocking on a concurrent writer.
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def put(self, attestation: VerifiedAttestation) -> None:
        """Insert `attestation`, or no-op if it is already recorded with
        byte-identical fields.

        **Serialized via `BEGIN IMMEDIATE` for genuinely new inserts
        (changed 2026-10-01, alongside adding the `record_hash` chain --
        remediating P2-5).** Computing each row's `record_hash` requires
        knowing the TRUE previous row in insertion order at write time, not
        merely "whatever `get()` observed a moment earlier" -- two
        different new attestations racing to be inserted next could
        otherwise both read the same stale "latest hash" and chain off it,
        producing a `record_hash` that `verify_integrity()` would then
        (wrongly) report as broken, even though nothing was actually
        tampered with. `BEGIN IMMEDIATE` acquires the write lock at the
        START of the transaction (the same pattern already used by
        `persistent_budget.py::PersistentBudgetTracker`, fixed there for
        P1-3), so the previous-hash read and this row's insert are
        atomic relative to every other writer.

        Still safe under CONCURRENT callers `put()`-ing the SAME
        (content-identical) attestation at once (see
        `tests/quantum1b/hardware_readiness/test_evidence_store_concurrency.py::
        TestConcurrentIdenticalWrites`): the initial `get()` check below
        short-circuits the common case; for the genuine race window, the
        losing transaction's own re-check (inside its `BEGIN IMMEDIATE`)
        sees the winner's already-committed row and reconciles via
        `_check_matches_or_raise` rather than raising a spurious
        `IntegrityError`."""
        existing = self.get(attestation.attestation_id)
        if existing is not None:
            self._check_matches_or_raise(existing, attestation)
            return  # idempotent re-put of the byte-identical record: fine.

        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM attestations WHERE attestation_id = ?",
                (attestation.attestation_id,),
            ).fetchone()
            if row is not None:
                conn.rollback()
                self._check_matches_or_raise(VerifiedAttestation(*row), attestation)
                return
            prev_row = conn.execute(
                "SELECT record_hash FROM attestations ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            prev_hash = prev_row[0] if prev_row is not None else _GENESIS_HASH
            record_hash = _compute_record_hash(prev_hash, attestation)
            try:
                conn.execute(
                    f"INSERT INTO attestations ({_COLUMNS}, record_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        attestation.attestation_id,
                        attestation.provider,
                        attestation.backend_name,
                        attestation.job_id,
                        attestation.verified_completion_timestamp,
                        attestation.raw_result_hash,
                        attestation.verifier_name,
                        record_hash,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # Should be unreachable under BEGIN IMMEDIATE (the write
                # lock means no other writer's INSERT can land between our
                # SELECT above and this INSERT) -- kept as a
                # defense-in-depth signal, never silently swallowed.
                conn.rollback()
                raced_existing = self.get(attestation.attestation_id)
                if raced_existing is None:
                    raise AttestationAlreadyExistsError(
                        f"attestation_id={attestation.attestation_id!r}: INSERT hit "
                        "a PRIMARY KEY conflict but the row is not visible on "
                        "re-fetch -- unexpected store-level inconsistency."
                    )
                self._check_matches_or_raise(raced_existing, attestation)
        finally:
            conn.close()

    @staticmethod
    def _check_matches_or_raise(existing: VerifiedAttestation, new: VerifiedAttestation) -> None:
        if existing != new:
            raise AttestationTamperedError(
                f"attestation_id={new.attestation_id!r} is already recorded "
                f"with different fields.\n  existing={existing!r}\n  new={new!r}\n"
                "An attestation is immutable once written -- this store never "
                "silently overwrites one."
            )

    def get(self, attestation_id: str) -> VerifiedAttestation | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM attestations WHERE attestation_id = ?",
                (attestation_id,),
            ).fetchone()
        if row is None:
            return None
        return VerifiedAttestation(*row)

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM attestations").fetchone()[0]

    def list_all(self) -> list[VerifiedAttestation]:
        """Ordered by insertion order (`rowid`), which IS the hash chain's
        own canonical order (see `verify_integrity()`). **Fixed 2026-10-01,
        remediating independent-review finding P3-6** ("recorded_at has
        whole-second resolution with no tiebreaker, so order among
        same-second writes -- the normal case under genuine concurrent
        writers -- is undefined"): previously ordered by `recorded_at ASC`
        alone; `rowid` is monotonically increasing and assigned at insert
        time, giving a deterministic order even when multiple rows share
        the same whole-second `recorded_at` value."""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM attestations ORDER BY rowid ASC").fetchall()
        return [VerifiedAttestation(*row) for row in rows]

    def verify_integrity(self) -> IntegrityVerificationResult:
        """Re-walks the hash chain in insertion order (`rowid`) and
        confirms every stored `record_hash` matches what the chain implies.
        Detects (does not prevent -- see module docstring) out-of-band
        mutation via raw filesystem/SQL access, which bypasses `put()`'s
        own tamper checks entirely. Returns the FIRST broken position, if
        any -- see `IntegrityVerificationResult`'s own docstring for why
        only the first."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS}, record_hash FROM attestations ORDER BY rowid ASC"
            ).fetchall()

        prev_hash = _GENESIS_HASH
        for row in rows:
            *fields, stored_hash = row
            attestation = VerifiedAttestation(*fields)
            expected_hash = _compute_record_hash(prev_hash, attestation)
            if expected_hash != stored_hash:
                return IntegrityVerificationResult(
                    all_match=False,
                    n_records_checked=len(rows),
                    first_broken_attestation_id=attestation.attestation_id,
                )
            prev_hash = stored_hash
        return IntegrityVerificationResult(
            all_match=True, n_records_checked=len(rows), first_broken_attestation_id=None
        )


__all__ = [
    "PersistentEvidenceStore",
    "AttestationAlreadyExistsError",
    "AttestationTamperedError",
    "IntegrityVerificationResult",
]
