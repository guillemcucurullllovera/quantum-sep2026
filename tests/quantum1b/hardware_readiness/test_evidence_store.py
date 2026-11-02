"""Tests for `experiments.quantum1b.hardware_readiness.evidence_store`."""
from __future__ import annotations

import sqlite3

import pytest

from core.verification import MockVerifier, VerifiedAttestation
from core.schemas import HardwareRunEvidence
from experiments.quantum1b.hardware_readiness.evidence_store import (
    AttestationTamperedError,
    IntegrityVerificationResult,
    PersistentEvidenceStore,
)


def _make_attestation(job_id: str = "job-abc", backend: str = "ibm_kingston") -> VerifiedAttestation:
    claimed = HardwareRunEvidence(
        backend_name=backend,
        job_id=job_id,
        execution_timestamp="2026-09-30T12:00:00+00:00",
    )
    return MockVerifier().verify(claimed)


class TestPersistentEvidenceStoreBasics:
    def test_put_then_get_roundtrips(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        att = _make_attestation()
        store.put(att)
        fetched = store.get(att.attestation_id)
        assert fetched == att

    def test_get_missing_returns_none(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        assert store.get("does-not-exist") is None

    def test_count_and_list_all(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        a1 = _make_attestation(job_id="job-1")
        a2 = _make_attestation(job_id="job-2")
        store.put(a1)
        store.put(a2)
        assert store.count() == 2
        ids = {a.attestation_id for a in store.list_all()}
        assert ids == {a1.attestation_id, a2.attestation_id}


class TestPersistentEvidenceStoreImmutability:
    def test_identical_reput_is_idempotent(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        att = _make_attestation()
        store.put(att)
        store.put(att)  # same object, byte-identical fields -- must not raise
        assert store.count() == 1

    def test_reput_with_different_fields_raises(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        att = _make_attestation(job_id="job-x")
        store.put(att)

        tampered = VerifiedAttestation(
            attestation_id=att.attestation_id,  # same id
            provider=att.provider,
            backend_name="a_different_backend",  # different field
            job_id=att.job_id,
            verified_completion_timestamp=att.verified_completion_timestamp,
            raw_result_hash=att.raw_result_hash,
            verifier_name=att.verifier_name,
        )
        with pytest.raises(AttestationTamperedError):
            store.put(tampered)
        # original record must be unchanged after the rejected attempt
        assert store.get(att.attestation_id) == att


class TestPersistentEvidenceStorePersistsAcrossProcesses:
    def test_survives_reopening_the_store(self, tmp_path):
        db_path = tmp_path / "evidence.db"
        att = _make_attestation()
        PersistentEvidenceStore(db_path).put(att)

        # simulate "a different later process" by constructing a brand new
        # PersistentEvidenceStore instance pointed at the same db file --
        # the exact cross-process durability InMemoryEvidenceStore cannot
        # provide.
        reopened = PersistentEvidenceStore(db_path)
        assert reopened.get(att.attestation_id) == att
        assert reopened.count() == 1

    def test_raw_sqlite_file_is_actually_on_disk(self, tmp_path):
        db_path = tmp_path / "evidence.db"
        PersistentEvidenceStore(db_path).put(_make_attestation())
        assert db_path.exists()
        # verify it is genuinely readable as sqlite by a totally independent
        # connection, not just by this class's own code
        conn = sqlite3.connect(str(db_path))
        try:
            n = conn.execute("SELECT COUNT(*) FROM attestations").fetchone()[0]
        finally:
            conn.close()
        assert n == 1


class TestVerifyIntegrity:
    """Regression tests for independent-review finding P2-5 ('evidence_store
    .py's "impossible to tamper" claim is an API-surface guarantee only;
    raw filesystem/SQL access bypasses it completely and undetectably').
    Independently reproduced before this fix: a raw `sqlite3.connect()` +
    `UPDATE` against an already-`put()` row succeeded silently, and
    `store.get()` returned the mutated row with no error. This fix cannot
    PREVENT that (an inherent limitation of any app-level API over a plain
    SQLite file, disclosed in the module docstring) -- it adds
    `verify_integrity()`, a hash chain that DETECTS it after the fact."""

    def test_empty_store_is_trivially_valid(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        result = store.verify_integrity()
        assert result.all_match is True
        assert result.n_records_checked == 0

    def test_untampered_chain_of_several_records_is_valid(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        for i in range(5):
            store.put(_make_attestation(job_id=f"job-{i}"))
        result = store.verify_integrity()
        assert result.all_match is True
        assert result.n_records_checked == 5
        assert result.first_broken_attestation_id is None

    def test_detects_a_raw_sql_mutation_bypassing_put(self, tmp_path):
        """The actual P2-5 reproduction scenario: a raw UPDATE via a
        completely independent sqlite3 connection, bypassing put() and
        its AttestationTamperedError check entirely."""
        db_path = tmp_path / "evidence.db"
        store = PersistentEvidenceStore(db_path)
        att = _make_attestation(job_id="job-tamper-target")
        store.put(att)

        # Confirm the tamper actually lands and is invisible to get() --
        # i.e. confirm the vulnerability this fix detects genuinely exists.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE attestations SET backend_name = 'tampered_backend' WHERE attestation_id = ?",
            (att.attestation_id,),
        )
        conn.commit()
        conn.close()
        assert store.get(att.attestation_id).backend_name == "tampered_backend"  # get() doesn't notice

        result = store.verify_integrity()
        assert result.all_match is False
        assert result.first_broken_attestation_id == att.attestation_id

    def test_detects_tampering_of_an_earlier_record_in_a_longer_chain(self, tmp_path):
        db_path = tmp_path / "evidence.db"
        store = PersistentEvidenceStore(db_path)
        atts = [_make_attestation(job_id=f"job-{i}") for i in range(4)]
        for a in atts:
            store.put(a)

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE attestations SET raw_result_hash = 'forged' WHERE attestation_id = ?",
            (atts[1].attestation_id,),  # tamper with the SECOND record, not the last
        )
        conn.commit()
        conn.close()

        result = store.verify_integrity()
        assert result.all_match is False
        assert result.first_broken_attestation_id == atts[1].attestation_id

    def test_deleting_a_record_breaks_the_chain(self, tmp_path):
        db_path = tmp_path / "evidence.db"
        store = PersistentEvidenceStore(db_path)
        atts = [_make_attestation(job_id=f"job-{i}") for i in range(3)]
        for a in atts:
            store.put(a)

        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM attestations WHERE attestation_id = ?", (atts[0].attestation_id,))
        conn.commit()
        conn.close()

        result = store.verify_integrity()
        assert result.all_match is False  # the remaining chain no longer starts from genesis correctly

    def test_returns_a_frozen_dataclass(self, tmp_path):
        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        result = store.verify_integrity()
        assert isinstance(result, IntegrityVerificationResult)


class TestEvidenceStoreProtocolCompatibility:
    def test_works_as_compute_evidence_level_store(self, tmp_path):
        """End-to-end: a PersistentEvidenceStore is a drop-in replacement
        for InMemoryEvidenceStore wherever core.evidence.compute_evidence_level
        needs an evidence_store -- this is the actual integration point
        this whole module exists to satisfy."""
        from core.evidence import EvidenceLevel, compute_evidence_level
        from core.schemas import (
            BackendDescriptor,
            BackendKind,
            ExperimentConfig,
            ExperimentResult,
            RunMetadata,
        )
        from core.verification import attest

        store = PersistentEvidenceStore(tmp_path / "evidence.db")
        claimed = HardwareRunEvidence(
            backend_name="ibm_kingston",
            job_id="job-e2e",
            execution_timestamp="2026-09-30T12:00:00+00:00",
        )
        attestation_id = attest(claimed, MockVerifier(), store)
        claimed_with_id = claimed.model_copy(update={"attestation_id": attestation_id})

        result = ExperimentResult(
            metadata=RunMetadata(algorithm="test"),
            config=ExperimentConfig(
                name="test",
                algorithm="test",
                backend=BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=5),
            ),
            hardware_evidence=claimed_with_id,
        )
        level = compute_evidence_level(result, evidence_store=store)
        assert level == EvidenceLevel.F
