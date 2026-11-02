"""Tests for hardware.braket — circuit translation, local-simulator
execution, and AWS credential gating."""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from core.errors import CredentialError
from hardware.braket.backend_adapter import (
    UnsupportedGateError,
    qiskit_to_braket,
    run_local,
)
from hardware.braket.config import AWSCredentials, get_aws_credentials


def test_translation_preserves_qubit_count():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.4, 2)
    result = qiskit_to_braket(qc)
    assert result.n_qubits == 3


def test_translation_drops_measure_and_barrier_without_raising():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    result = qiskit_to_braket(qc)
    assert "measure" in result.dropped_ops
    assert "barrier" in result.dropped_ops


def test_unsupported_gate_raises_not_silently_wrong():
    qc = QuantumCircuit(2)
    qc.append(QuantumCircuit(2).to_gate(label="custom_unitary"), [0, 1])
    with pytest.raises(UnsupportedGateError):
        qiskit_to_braket(qc)


def test_local_simulator_statevector_matches_qiskit_exactly():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 0)
    qc.ry(0.7, 1)

    braket_result = run_local(qc, shots=0)
    qiskit_sv = np.asarray(Statevector.from_instruction(qc))

    np.testing.assert_allclose(braket_result["state_vector"], qiskit_sv, atol=1e-9)


def test_local_simulator_statevector_matches_qiskit_for_asymmetric_3q_circuit():
    """Regression test for the qubit-ordering bug found during development:
    a circuit where each qubit gets a distinct rotation, so any bit-order
    mismatch between Braket and Qiskit conventions is detectable (unlike
    the original 2-qubit test case, which happened to be symmetric)."""
    qc = QuantumCircuit(3)
    qc.rx(0.11, 0)
    qc.ry(0.53, 1)
    qc.rz(0.97, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(0.21, 0)

    braket_result = run_local(qc, shots=0)
    qiskit_sv = np.asarray(Statevector.from_instruction(qc))
    np.testing.assert_allclose(braket_result["state_vector"], qiskit_sv, atol=1e-9)


def test_local_simulator_sampling_produces_valid_counts():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    result = run_local(qc, shots=500)
    counts = result["measurement_counts"]
    assert sum(counts.values()) == 500
    # Bell state: only 00/11 should appear (up to simulator numerical noise)
    assert set(counts.keys()) <= {"00", "11"}
    assert counts.get("00", 0) > 100  # roughly half, not a degenerate distribution
    assert counts.get("11", 0) > 100


def test_local_simulator_needs_no_aws_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    qc = QuantumCircuit(1)
    qc.h(0)
    result = run_local(qc, shots=100)  # must not raise CredentialError
    assert sum(result["measurement_counts"].values()) == 100


def test_aws_credentials_absent_by_default(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    creds = AWSCredentials.from_env()
    assert creds.is_present is False
    with pytest.raises(CredentialError):
        creds.require()


def test_aws_credentials_present_when_env_set(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-test-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-test-secret")
    creds = AWSCredentials.from_env()
    assert creds.is_present is True
    assert creds.require().access_key_id == "fake-test-key-id"


def test_run_remote_raises_credential_error_before_any_network_touch(monkeypatch):
    """The single most important test: run_remote must fail loudly and
    immediately when no AWS credentials are present, never silently no-op
    and never attempt a network call first."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    import hardware.braket.config as config_mod

    config_mod._aws_credentials = None  # force re-read of the patched env
    from hardware.braket.backend_adapter import run_remote

    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(CredentialError):
        run_remote(qc, device_arn="arn:aws:braket:us-east-1::device/qpu/fake/NotReal")


def test_get_aws_credentials_singleton_refresh(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s1")
    c1 = get_aws_credentials(refresh=True)
    assert c1.access_key_id == "k1"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k2")
    c2 = get_aws_credentials(refresh=True)
    assert c2.access_key_id == "k2"
