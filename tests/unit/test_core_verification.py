from datetime import datetime, timezone

import pytest

from core.schemas import HardwareRunEvidence
from core.verification import (
    IBMRuntimeVerifier,
    InMemoryEvidenceStore,
    MockVerifier,
    attest,
)


def _claim(**overrides) -> HardwareRunEvidence:
    defaults = dict(
        backend_name="ibm_kingston",
        job_id="job-abc",
        execution_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return HardwareRunEvidence(**defaults)


def test_attest_round_trips_through_store():
    store = InMemoryEvidenceStore()
    claim = _claim()
    attestation_id = attest(claim, MockVerifier(), store)
    resolved = store.get(attestation_id)
    assert resolved is not None
    assert resolved.backend_name == claim.backend_name
    assert resolved.job_id == claim.job_id


def test_store_returns_none_for_unknown_id():
    store = InMemoryEvidenceStore()
    assert store.get("never-attested") is None


def test_mock_verifier_is_deterministic():
    claim = _claim()
    a = MockVerifier().verify(claim)
    b = MockVerifier().verify(claim)
    assert a.attestation_id == b.attestation_id


def test_mock_verifier_differs_for_different_claims():
    a = MockVerifier().verify(_claim(job_id="job-1"))
    b = MockVerifier().verify(_claim(job_id="job-2"))
    assert a.attestation_id != b.attestation_id


def test_ibm_runtime_verifier_not_implemented_documents_required_checklist():
    with pytest.raises(NotImplementedError) as exc_info:
        IBMRuntimeVerifier().verify(_claim())
    # Not just "raises" — the message must point somewhere that actually
    # documents what's missing, so this isn't a silent dead end.
    message = str(exc_info.value).lower()
    assert "not implemented" in message
    assert "docstring" in message or "checklist" in message
    # And the checklist itself (on the class, not just the exception
    # message) must actually mention the required verification steps.
    doc = (IBMRuntimeVerifier.__doc__ or "").lower()
    for required in ("backend", "status", "timestamp", "hash"):
        assert required in doc
