"""Real execution-path tests for applications.portfolio.hardware_exec: every
non-credential-gated target must actually run the circuit (real counts
summing to `shots`, from a real backend name), and the two credential-gated
targets must raise CredentialError before any network call when the
corresponding credential is absent -- following the same pattern
tests/applications/test_options_execution.py already established for this
platform's other credential-gated execution paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.hardware_exec import HardwareTarget, build_portable_qaoa_circuit, execute_on_target
from core.errors import CredentialError, ValidationError


def _toy_circuit():
    n = 4
    J = np.zeros((n, n))
    J[0, 1] = 0.4
    J[1, 2] = -0.3
    h = np.array([0.1, -0.2, 0.05, 0.15])
    return build_portable_qaoa_circuit(J, h, p=1, gamma=[0.3], beta=[0.2])


def test_portable_circuit_uses_only_the_braket_translatable_gate_vocabulary():
    qc = _toy_circuit()
    used = {instr.operation.name for instr in qc.data}
    assert used <= {"h", "rz", "cx", "rx", "measure", "barrier"}


def test_aer_noiseless_execution_runs_and_returns_real_counts():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.AER_NOISELESS, shots=512)
    assert sum(result.counts.values()) == 512
    assert result.backend_name == "aer_simulator_noiseless"


def test_aer_fake_noisy_execution_uses_a_real_device_calibrated_backend():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.AER_FAKE_NOISY, shots=256, fake_backend="fake_eagle")
    assert sum(result.counts.values()) == 256
    assert result.backend_name.startswith("Fake")


def test_aer_synthetic_noisy_execution_reports_a_real_noise_descriptor():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.AER_SYNTHETIC_NOISY, shots=256)
    assert sum(result.counts.values()) == 256
    assert result.noise_descriptor is not None
    assert result.noise_descriptor["source"] == "synthetic"


def test_braket_local_execution_runs_with_no_aws_credentials():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.BRAKET_LOCAL, shots=200)
    assert result.backend_name == "braket_local_simulator"
    assert sum(result.counts.values()) == 200


def test_ibm_qpu_raises_credential_error_with_no_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings
    get_settings(refresh=True)

    qc = _toy_circuit()
    with pytest.raises(CredentialError, match="IBM_QUANTUM_TOKEN"):
        execute_on_target(qc, HardwareTarget.IBM_QPU)


def test_braket_qpu_raises_credential_error_with_no_aws_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    import hardware.braket.config as config_mod
    config_mod._aws_credentials = None

    qc = _toy_circuit()
    with pytest.raises(CredentialError):
        execute_on_target(qc, HardwareTarget.BRAKET_QPU, device_arn="arn:aws:braket:us-east-1::device/qpu/fake/Fake")


def test_braket_qpu_without_device_arn_raises_validation_error():
    """device_arn is checked before the credential gate (matching
    applications.options.execution's existing ordering for the same
    target), so this is a real ValidationError regardless of whether AWS
    credentials happen to be configured in the environment."""
    qc = _toy_circuit()
    with pytest.raises(ValidationError, match="device_arn"):
        execute_on_target(qc, HardwareTarget.BRAKET_QPU)
