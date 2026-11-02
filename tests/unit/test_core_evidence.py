import pytest

from core.errors import EvidenceIntegrityError
from core.evidence import EvidenceLevel, compute_evidence_level
from core.result_store import ResultStore
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    HardwareRunEvidence,
    NoiseModelDescriptor,
    RunMetadata,
)
from core.verification import InMemoryEvidenceStore, MockVerifier, attest


def _config(backend: BackendDescriptor, **kwargs) -> ExperimentConfig:
    return ExperimentConfig(name="t", algorithm="qae_iqae", backend=backend, **kwargs)


def _result(config: ExperimentConfig, **kwargs) -> ExperimentResult:
    return ExperimentResult(metadata=RunMetadata(algorithm="qae_iqae"), config=config, **kwargs)


# --- A-E: unaffected by the verification hardening (backward compatibility) ---


def test_ideal_simulator_no_shots_is_level_b():
    backend = BackendDescriptor(name="aer_statevector", kind=BackendKind.SIMULATOR, num_qubits=4)
    result = _result(_config(backend, shots=None))
    assert compute_evidence_level(result) == EvidenceLevel.B


def test_finite_shot_simulator_is_level_c():
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    result = _result(_config(backend, shots=4096))
    assert compute_evidence_level(result) == EvidenceLevel.C


def test_synthetic_noise_simulator_is_level_d():
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    noise = NoiseModelDescriptor(backend_name="synthetic_depolarizing", source="synthetic")
    result = _result(_config(backend, shots=4096, noise_model=noise))
    assert compute_evidence_level(result) == EvidenceLevel.D


def test_fake_backend_is_level_e():
    backend = BackendDescriptor(
        name="FakeKingston", kind=BackendKind.FAKE, num_qubits=156,
        calibration_source="fake_backend:FakeKingston",
    )
    result = _result(_config(backend, shots=4096))
    assert compute_evidence_level(result) == EvidenceLevel.E


# --- Level F: legitimate path now requires a real verified attestation ---


def _valid_hw_evidence(**overrides) -> HardwareRunEvidence:
    from datetime import datetime, timezone

    defaults = dict(
        backend_name="ibm_kingston",
        job_id="d2x0123456789",
        execution_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return HardwareRunEvidence(**defaults)


def test_qpu_with_verified_attestation_is_level_f():
    backend = BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=156)
    claimed = _valid_hw_evidence()
    store = InMemoryEvidenceStore()
    attestation_id = attest(claimed, MockVerifier(), store)
    ev = claimed.model_copy(update={"attestation_id": attestation_id})
    result = _result(_config(backend, shots=4096), hardware_evidence=ev)
    assert compute_evidence_level(result, evidence_store=store) == EvidenceLevel.F


def test_qpu_without_hardware_evidence_fails_closed():
    backend = BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=156)
    result = _result(_config(backend, shots=4096))
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result)


def test_qpu_with_mismatched_backend_name_fails_closed():
    backend = BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=156)
    ev = _valid_hw_evidence(backend_name="ibm_boston")
    result = _result(_config(backend, shots=4096), hardware_evidence=ev)
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result)


# --- Adversarial reproductions: the reviewer's exact forged-F examples ---


def test_forged_f_reviewer_exact_repro_empty_timestamp_now_rejected_at_construction():
    """The reviewer's exact reproduction:
    HardwareRunEvidence(backend_name="ibm_brisbane",
    job_id="totally-made-up-job-id", execution_timestamp="")
    previously sailed through to level F with no exception anywhere. It
    must now fail at construction time (schema-level validation), before
    evidence computation is ever reached."""
    with pytest.raises(Exception):
        HardwareRunEvidence(
            backend_name="ibm_brisbane",
            job_id="totally-made-up-job-id",
            execution_timestamp="",
        )


def test_forged_f_well_formed_claim_with_no_attestation_id_fails_closed():
    """A claim can be well-formed (passes the schema validators) and still
    be pure fabrication — job_id/backend_name/timestamp can all be
    plausible-looking and entirely made up. Without an attestation_id
    resolving against a real EvidenceStore, this must not reach level F."""
    backend = BackendDescriptor(name="ibm_brisbane", kind=BackendKind.QPU, num_qubits=127)
    claimed = _valid_hw_evidence(backend_name="ibm_brisbane", job_id="totally-made-up-job-id")
    assert claimed.attestation_id is None
    result = _result(_config(backend, shots=4096), hardware_evidence=claimed)
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result, evidence_store=InMemoryEvidenceStore())


def test_forged_f_hand_typed_attestation_id_does_not_resolve():
    """Even if a caller guesses/types a plausible-looking attestation_id
    by hand (rather than obtaining one from attest()), it must not resolve
    against a real store that never recorded it."""
    backend = BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=156)
    claimed = _valid_hw_evidence().model_copy(
        update={"attestation_id": "hand-typed-not-a-real-attestation"}
    )
    result = _result(_config(backend, shots=4096), hardware_evidence=claimed)
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result, evidence_store=InMemoryEvidenceStore())


def test_forged_f_no_evidence_store_supplied_fails_closed():
    """A serialized ExperimentResult ALONE (no evidence_store passed at
    all) must never be sufficient to establish level F, even with a
    well-formed attestation_id set."""
    backend = BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=156)
    claimed = _valid_hw_evidence()
    store = InMemoryEvidenceStore()
    attestation_id = attest(claimed, MockVerifier(), store)
    ev = claimed.model_copy(update={"attestation_id": attestation_id})
    result = _result(_config(backend, shots=4096), hardware_evidence=ev)
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result)  # no evidence_store kwarg at all


def test_forged_f_tampered_evidence_after_attestation_fails_closed():
    """The claimed fields must still match the attestation's fields —
    catches tampering with hardware_evidence after a real attestation was
    recorded for a different job."""
    backend = BackendDescriptor(name="ibm_kingston", kind=BackendKind.QPU, num_qubits=156)
    claimed = _valid_hw_evidence(job_id="real-job-1")
    store = InMemoryEvidenceStore()
    attestation_id = attest(claimed, MockVerifier(), store)
    tampered = claimed.model_copy(
        update={"attestation_id": attestation_id, "job_id": "swapped-in-different-job"}
    )
    result = _result(_config(backend, shots=4096), hardware_evidence=tampered)
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result, evidence_store=store)


def test_ibm_runtime_verifier_is_not_implemented():
    """Documents, mechanically, that no real IBM call exists yet — this
    test must start failing (be updated) the day IBMRuntimeVerifier is
    actually implemented, not silently keep passing against dead code."""
    from core.verification import IBMRuntimeVerifier

    with pytest.raises(NotImplementedError):
        IBMRuntimeVerifier().verify(_valid_hw_evidence())


# --- Level G: legitimate path now requires a real resolved artifact ---


def test_financial_edge_claim_with_real_artifact_and_baseline_is_level_g(tmp_path):
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    store = ResultStore(root=tmp_path)
    run_id = "erosion_audit_run_1"
    store.save_json(run_id, "backtest", {"sharpe": 0.9})
    result = _result(
        _config(backend),
        financial_edge_claimed=True,
        financial_backtest_artifact=f"{run_id}/backtest",
        classical_baseline_metrics={"sharpe": 0.895},
    )
    assert compute_evidence_level(result, result_store=store) == EvidenceLevel.G


def test_financial_edge_claim_without_backtest_fails_closed():
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    result = _result(_config(backend), financial_edge_claimed=True)
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result)


# --- Adversarial reproductions: the reviewer's exact forged-G example ---


def test_forged_g_reviewer_exact_repro_garbage_artifact_string():
    """The reviewer's exact reproduction: financial_edge_claimed=True,
    financial_backtest_artifact="not_a_real_path_asdf" previously returned
    level G with no exception. It must now fail closed."""
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    result = _result(
        _config(backend),
        financial_edge_claimed=True,
        financial_backtest_artifact="not_a_real_path_asdf",
        classical_baseline_metrics={"sharpe": 0.9},
    )
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result)


def test_forged_g_plausible_looking_ref_that_does_not_resolve(tmp_path):
    """A "run_id/artifact_name"-shaped string that merely looks legitimate
    but was never actually saved through ResultStore must not resolve."""
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    store = ResultStore(root=tmp_path)
    result = _result(
        _config(backend),
        financial_edge_claimed=True,
        financial_backtest_artifact="erosion_audit_run_1/backtest",
        classical_baseline_metrics={"sharpe": 0.9},
    )
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result, result_store=store)


def test_forged_g_no_result_store_supplied_fails_closed(tmp_path):
    """Even with a real artifact saved and a store available elsewhere,
    omitting result_store from the call must fail closed — a serialized
    ExperimentResult alone is never sufficient for level G."""
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    store = ResultStore(root=tmp_path)
    store.save_json("run1", "backtest", {"sharpe": 0.9})
    result = _result(
        _config(backend),
        financial_edge_claimed=True,
        financial_backtest_artifact="run1/backtest",
        classical_baseline_metrics={"sharpe": 0.9},
    )
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result)  # no result_store kwarg at all


def test_forged_g_empty_classical_baseline_metrics_fails_closed(tmp_path):
    """Related, smaller gap the review also flagged: financial_backtest_
    artifact resolving to a real file is not enough on its own — the claim
    must also be run against a real classical comparator."""
    backend = BackendDescriptor(name="aer_qasm", kind=BackendKind.SIMULATOR, num_qubits=4)
    store = ResultStore(root=tmp_path)
    store.save_json("run1", "backtest", {"sharpe": 0.9})
    result = _result(
        _config(backend),
        financial_edge_claimed=True,
        financial_backtest_artifact="run1/backtest",
        classical_baseline_metrics=None,
    )
    with pytest.raises(EvidenceIntegrityError):
        compute_evidence_level(result, result_store=store)
