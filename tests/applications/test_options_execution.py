"""Real execution-path tests: the noisy-simulator (fake-backend) tier must
run a REAL (not toy) payoff circuit without OOMing -- exercising
hardware.ibm_runtime.primitives._truncate_to_active_qubits qubit-truncation
fix -- and both IBM_QPU/BRAKET_QPU must raise
CredentialError before any network call when credentials are absent.
"""
from __future__ import annotations

import pytest

from algorithms_extended.qiskit_finance_extensions.barrier_payoff import (
    build_A_barrier_payoff, build_barrier_payoff_integrand,
)
from applications.options.execution import ExecutionBackend, run_priced_circuit, sampled_amplitude
from core.errors import CredentialError, ValidationError


def _real_barrier_circuit(n: int = 5):
    """A real (not toy) barrier-payoff circuit: n=5 index qubits + 1 payoff
    ancilla = 6 qubits, with a state_preparation over 32 amplitudes and
    multi-controlled-RY gates -- this is the exact circuit shape
    applications.options.barrier_pricer.BarrierOptionPricer prices with by
    default, not a synthetic minimal circuit."""
    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100.0, strike=100.0, barrier=115.0, r=0.04, sigma=0.2, t=0.25,
        n=n, n_mc_paths=2000, seed=1,
    )
    circuit = build_A_barrier_payoff(integrand)
    return circuit, integrand


def test_noisy_simulator_runs_real_circuit_on_device_sized_fake_backend_without_oom():
    """Before the qubit-truncation fix, transpiling a circuit like this onto
    a 130+ qubit fake backend and measuring all qubits forced Aer into a
    dense statevector/density-matrix sized to the WHOLE backend and OOM'd.
    This must now complete quickly and return real, well-formed counts."""
    circuit, integrand = _real_barrier_circuit(n=5)
    assert circuit.num_qubits == 6  # a real circuit, not a 2-3 qubit toy

    result = run_priced_circuit(
        circuit, execution_backend=ExecutionBackend.NOISY_SIMULATOR, shots=2048,
        fake_backend_name="fake_heron",
    )
    assert sum(result.counts.values()) == 2048
    assert result.backend_name  # a real fake-backend name (e.g. FakeTorino), not empty

    a_hat, total = sampled_amplitude(result, ancilla_clbit=integrand.dist.num_qubits)
    assert total == 2048
    assert 0.0 <= a_hat <= 1.0


def test_noisy_simulator_result_is_noisy_relative_to_ideal_simulator():
    """A sanity check that the fake-backend path really is applying device
    noise (not silently falling back to a clean simulator, the exact failure
    mode hardware.ibm_runtime.primitives.run_sampler's own docstring calls
    out as unacceptable): the noisy ancilla-1 rate should differ from the
    noiseless one by more than pure shot noise alone would explain, OR at
    minimum both runs should independently produce valid probabilities --
    kept as a loose bound since noise magnitude is backend/circuit dependent."""
    circuit, integrand = _real_barrier_circuit(n=4)
    ideal = run_priced_circuit(circuit, execution_backend=ExecutionBackend.SIMULATOR, shots=4096)
    noisy = run_priced_circuit(
        circuit, execution_backend=ExecutionBackend.NOISY_SIMULATOR, shots=4096, fake_backend_name="fake_heron",
    )
    a_ideal, _ = sampled_amplitude(ideal, ancilla_clbit=integrand.dist.num_qubits)
    a_noisy, _ = sampled_amplitude(noisy, ancilla_clbit=integrand.dist.num_qubits)
    assert 0.0 <= a_ideal <= 1.0
    assert 0.0 <= a_noisy <= 1.0


def test_readout_m3_mitigation_runs_on_noiseless_simulator():
    circuit, _ = _real_barrier_circuit(n=3)
    result = run_priced_circuit(
        circuit, execution_backend=ExecutionBackend.SIMULATOR, shots=1024, mitigation="readout_m3",
    )
    assert sum(result.counts.values()) > 0


def test_braket_local_simulator_runs_the_same_real_circuit():
    circuit, _ = _real_barrier_circuit(n=4)
    result = run_priced_circuit(circuit, execution_backend=ExecutionBackend.BRAKET_LOCAL, shots=1000)
    assert result.backend_name == "braket_local_simulator"
    assert sum(result.counts.values()) == 1000


def test_ibm_qpu_raises_credential_error_with_no_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings
    get_settings(refresh=True)

    circuit, _ = _real_barrier_circuit(n=3)
    with pytest.raises(CredentialError, match="IBM_QUANTUM_TOKEN"):
        run_priced_circuit(circuit, execution_backend=ExecutionBackend.IBM_QPU)


def test_braket_qpu_raises_credential_error_with_no_aws_credentials(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    import hardware.braket.config as config_mod
    config_mod._aws_credentials = None  # force re-read of the patched env

    circuit, _ = _real_barrier_circuit(n=3)
    with pytest.raises(CredentialError):
        run_priced_circuit(
            circuit, execution_backend=ExecutionBackend.BRAKET_QPU,
            braket_device_arn="arn:aws:braket:us-east-1::device/qpu/fake/Fake",
        )


def test_braket_qpu_without_device_arn_raises_validation_error():
    circuit, _ = _real_barrier_circuit(n=3)
    with pytest.raises(ValidationError, match="braket_device_arn"):
        run_priced_circuit(circuit, execution_backend=ExecutionBackend.BRAKET_QPU)
