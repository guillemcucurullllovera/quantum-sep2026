"""Tests for `experiments.quantum1b.hardware_readiness.ibm_verifier`.

Every test here uses a plain fake `service`/`job` object -- none imports or
constructs a real `qiskit_ibm_runtime.QiskitRuntimeService`, and no test
makes a network call. This is deliberate: this module proves the
verification LOGIC is correct without ever touching a real IBM account.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from core.errors import EvidenceIntegrityError
from core.schemas import HardwareRunEvidence
from experiments.quantum1b.hardware_readiness.ibm_verifier import (
    IBMRuntimeVerifier,
    IBMVerifierConfig,
)


@dataclass
class FakeBackend:
    name: str


@dataclass
class FakeJob:
    _job_id: str
    _backend: FakeBackend
    _status: str = "DONE"
    _finished: datetime | None = None
    _result: Any = field(default_factory=lambda: {"counts": {"00": 512, "11": 512}})

    def job_id(self) -> str:
        return self._job_id

    def backend(self) -> FakeBackend:
        return self._backend

    def status(self) -> str:
        return self._status

    def metrics(self) -> dict[str, Any]:
        return {"timestamps": {"finished": self._finished}} if self._finished else {"timestamps": {}}

    def result(self) -> Any:
        return self._result


@dataclass
class FakeService:
    _jobs: dict[str, FakeJob]

    def job(self, job_id: str) -> FakeJob:
        return self._jobs[job_id]


NOW = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)


def _claim(job_id="job-1", backend="ibm_kingston", ts=NOW) -> HardwareRunEvidence:
    return HardwareRunEvidence(
        backend_name=backend, job_id=job_id, execution_timestamp=ts.isoformat()
    )


class TestIBMRuntimeVerifierHappyPath:
    def test_verifies_matching_claim(self):
        job = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", NOW)
        service = FakeService({"job-1": job})
        verifier = IBMRuntimeVerifier(service=service)

        attestation = verifier.verify(_claim())

        assert attestation.backend_name == "ibm_kingston"
        assert attestation.job_id == "job-1"
        assert attestation.provider == "ibm"
        assert attestation.verifier_name == "IBMRuntimeVerifier"
        assert len(attestation.raw_result_hash) == 64  # sha256 hex digest

    def test_same_inputs_produce_same_attestation_id(self):
        job = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", NOW)
        service = FakeService({"job-1": job})
        verifier = IBMRuntimeVerifier(service=service)

        a1 = verifier.verify(_claim())
        a2 = verifier.verify(_claim())
        assert a1.attestation_id == a2.attestation_id

    def test_different_result_produces_different_attestation_id(self):
        job_a = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", NOW, {"counts": {"00": 1000}})
        job_b = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", NOW, {"counts": {"11": 1000}})
        v = IBMRuntimeVerifier(service=FakeService({"job-1": job_a}))
        a1 = v.verify(_claim())
        v2 = IBMRuntimeVerifier(service=FakeService({"job-1": job_b}))
        a2 = v2.verify(_claim())
        assert a1.attestation_id != a2.attestation_id
        assert a1.raw_result_hash != a2.raw_result_hash


class TestIBMRuntimeVerifierRejectsMismatch:
    def test_wrong_backend_raises(self):
        job = FakeJob("job-1", FakeBackend("ibm_brisbane"), "DONE", NOW)
        verifier = IBMRuntimeVerifier(service=FakeService({"job-1": job}))
        with pytest.raises(EvidenceIntegrityError, match="backend"):
            verifier.verify(_claim(backend="ibm_kingston"))

    def test_non_terminal_status_raises(self):
        for bad_status in ("QUEUED", "RUNNING", "CANCELLED", "ERROR", "INITIALIZING"):
            job = FakeJob("job-1", FakeBackend("ibm_kingston"), bad_status, NOW)
            verifier = IBMRuntimeVerifier(service=FakeService({"job-1": job}))
            with pytest.raises(EvidenceIntegrityError, match="terminal"):
                verifier.verify(_claim())

    def test_missing_finished_timestamp_raises(self):
        job = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", None)
        verifier = IBMRuntimeVerifier(service=FakeService({"job-1": job}))
        with pytest.raises(EvidenceIntegrityError, match="timestamps"):
            verifier.verify(_claim())

    def test_timestamp_drift_beyond_tolerance_raises(self):
        actual_finish = NOW + timedelta(hours=2)
        job = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", actual_finish)
        verifier = IBMRuntimeVerifier(service=FakeService({"job-1": job}))
        with pytest.raises(EvidenceIntegrityError, match="drift"):
            verifier.verify(_claim(ts=NOW))

    def test_timestamp_within_tolerance_passes(self):
        actual_finish = NOW + timedelta(minutes=2)
        job = FakeJob("job-1", FakeBackend("ibm_kingston"), "DONE", actual_finish)
        verifier = IBMRuntimeVerifier(
            service=FakeService({"job-1": job}),
            config=IBMVerifierConfig(timestamp_tolerance=timedelta(minutes=10)),
        )
        attestation = verifier.verify(_claim(ts=NOW))
        assert attestation.verified_completion_timestamp == actual_finish.isoformat()

    def test_wrong_job_id_in_fetched_job_raises(self):
        job = FakeJob("some-other-id", FakeBackend("ibm_kingston"), "DONE", NOW)
        verifier = IBMRuntimeVerifier(service=FakeService({"job-1": job}))
        with pytest.raises(EvidenceIntegrityError, match="job id"):
            verifier.verify(_claim(job_id="job-1"))


class TestIBMRuntimeVerifierNeverTouchesRealNetwork:
    def test_module_does_not_import_qiskit_runtime_service(self):
        """Checks actual import statements, not prose -- the module's own
        docstring mentions `QiskitRuntimeService(...)` and
        `IBMQuantumCredentials` as documentation of what a FUTURE caller
        must wire up, which is expected and fine; what must never appear
        is this module itself importing or constructing either one."""
        import ast

        import experiments.quantum1b.hardware_readiness.ibm_verifier as mod

        tree = ast.parse(open(mod.__file__).read())
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
                imported_names.update(f"{node.module}.{a.name}" for a in node.names)

        assert not any("qiskit_ibm_runtime" in name for name in imported_names), imported_names
        assert not any("IBMQuantumCredentials" in name for name in imported_names), imported_names
        assert "IBMQuantumCredentials" not in [
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        ]
