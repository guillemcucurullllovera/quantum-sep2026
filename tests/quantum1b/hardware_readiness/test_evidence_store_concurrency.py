"""Concurrent-writer test for `PersistentEvidenceStore` -- closes one of
the disclosed gaps in `docs/quantum/QUANTUM1B_HARDWARE_READINESS.md`
("untested under concurrent writers"). Uses real OS threads (not
multiprocessing -- SQLite's file-level locking is what's actually being
exercised, and threads are sufficient to drive genuinely concurrent
`sqlite3.connect()` calls against the same file; this also respects the
overnight session's "no uncontrolled multiprocessing" resource discipline).
"""
from __future__ import annotations

import threading

from core.verification import MockVerifier
from core.schemas import HardwareRunEvidence
from experiments.quantum1b.hardware_readiness.evidence_store import (
    AttestationAlreadyExistsError,
    AttestationTamperedError,
    PersistentEvidenceStore,
)


def _attestation_for(job_id: str):
    claimed = HardwareRunEvidence(
        backend_name="ibm_kingston",
        job_id=job_id,
        execution_timestamp="2026-09-30T12:00:00+00:00",
    )
    return MockVerifier().verify(claimed)


class TestConcurrentDistinctWrites:
    def test_many_threads_writing_distinct_attestations_all_land(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        n_threads = 16
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                store.put(_attestation_for(f"job-{i}"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, errors
        assert store.count() == n_threads
        # every distinct job's attestation must be independently resolvable
        for i in range(n_threads):
            att = _attestation_for(f"job-{i}")
            assert store.get(att.attestation_id) == att


class TestConcurrentIdenticalWrites:
    def test_many_threads_writing_the_same_attestation_is_safe(self, tmp_path):
        """Many threads racing to `put()` the SAME (content-identical)
        attestation must all succeed (idempotent) and the store must end
        up with exactly one record -- no crash, no duplicate-row error
        surfacing as an uncaught exception, no corruption."""
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        attestation = _attestation_for("job-shared")
        n_threads = 16
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                store.put(attestation)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, errors
        assert store.count() == 1
        assert store.get(attestation.attestation_id) == attestation


class TestConcurrentReadWhileWrite:
    def test_reads_during_writes_never_see_a_torn_record(self, tmp_path):
        """While one thread writes N distinct attestations, another reads
        in a loop -- every read must return either None or a complete,
        valid, byte-identical-to-what-was-written record; never a partial
        or corrupted one. This is the practical concurrent-access property
        a real Hardware Day session (one process writing attestations,
        another process/report generator reading them) actually needs."""
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        n = 30
        read_errors: list[BaseException] = []
        stop = threading.Event()

        def writer() -> None:
            for i in range(n):
                store.put(_attestation_for(f"job-rw-{i}"))

        def reader() -> None:
            while not stop.is_set():
                try:
                    for i in range(n):
                        rec = store.get(_attestation_for(f"job-rw-{i}").attestation_id)
                        if rec is not None:
                            assert rec == _attestation_for(f"job-rw-{i}")
                except BaseException as exc:  # noqa: BLE001
                    read_errors.append(exc)
                    return

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        r.start()
        w.start()
        w.join(timeout=30)
        stop.set()
        r.join(timeout=30)

        assert not read_errors, read_errors
        assert store.count() == n


class TestConcurrentTamperDetectionStillWorks:
    def test_conflicting_concurrent_writes_to_same_id_raise_for_the_loser(self, tmp_path):
        """If two threads race to `put()` two DIFFERENT attestations that
        happen to share an attestation_id (simulating a hash collision or
        a deliberately malformed record), the store must never silently
        keep one over the other -- exactly one write wins, and the other
        raises `AttestationTamperedError` (never silently swallowed)."""
        from core.verification import VerifiedAttestation

        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        shared_id = "collision-id-0123456789abcdef"
        att_a = VerifiedAttestation(
            attestation_id=shared_id, provider="mock", backend_name="a",
            job_id="job-a", verified_completion_timestamp="t", raw_result_hash="hash-a",
            verifier_name="MockVerifier",
        )
        att_b = VerifiedAttestation(
            attestation_id=shared_id, provider="mock", backend_name="b",
            job_id="job-b", verified_completion_timestamp="t", raw_result_hash="hash-b",
            verifier_name="MockVerifier",
        )

        outcomes: list[str] = []
        lock = threading.Lock()

        def try_put(att) -> None:
            try:
                store.put(att)
                with lock:
                    outcomes.append("ok")
            except (AttestationTamperedError, AttestationAlreadyExistsError):
                # Both are correct-behavior outcomes for the losing thread
                # of a genuine race: if its own get()-then-insert window
                # loses to the other thread's INSERT first, SQLite's own
                # PRIMARY KEY constraint raises IntegrityError -> this
                # store surfaces that as AttestationAlreadyExistsError
                # (see evidence_store.py); if instead its get() call
                # happens to run AFTER the other thread's commit, put()
                # sees an existing, DIFFERENT record and raises
                # AttestationTamperedError itself. Which one occurs is a
                # nondeterministic function of thread scheduling -- what
                # must be true regardless is that exactly one write wins
                # and the other is rejected, never silently dropped or
                # silently overwritten.
                with lock:
                    outcomes.append("rejected")

        t_a = threading.Thread(target=try_put, args=(att_a,))
        t_b = threading.Thread(target=try_put, args=(att_b,))
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)

        # exactly one write must have won (recorded), the other must have
        # been rejected -- never both "ok", never both silently accepted
        # with different content under the same id.
        assert sorted(outcomes) == ["ok", "rejected"]
        final = store.get(shared_id)
        assert final in (att_a, att_b)
