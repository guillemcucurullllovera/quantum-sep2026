"""Coverage for applications/portfolio/hardware_exec.py (real Aer/Braket
execution paths, credential-gated IBM/Braket QPU paths) and
applications/portfolio/benchmark_suite.py (shared benchmarks.runner
integration + experiment-registry logging) added recently.
"""
from __future__ import annotations

import numpy as np
import pytest

from applications.portfolio.benchmark_suite import benchmark_pair, run_full_generation_benchmark
from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.hardware_exec import HardwareTarget, build_portable_qaoa_circuit, execute_on_target
from core.errors import CredentialError
from core.experiment_registry import ExperimentRegistry


def _toy_spec(n=6, k=3, seed=0) -> PortfolioSpec:
    rng = np.random.default_rng(seed)
    names = [f"A{i}" for i in range(n)]
    mu = rng.uniform(0.02, 0.15, size=n)
    L = rng.normal(scale=0.05, size=(n, n))
    cov = (L @ L.T + np.eye(n) * 0.01).tolist()
    return PortfolioSpec(
        assets=[AssetSpec(name=nm, expected_return=float(m)) for nm, m in zip(names, mu)],
        cov_matrix=cov,
        constraints=[ConstraintSpec(kind="cardinality", params={"k": k})],
    )


def _toy_circuit():
    J = np.zeros((3, 3))
    J[0, 1], J[1, 2] = 0.3, -0.2
    h = np.array([0.1, -0.1, 0.05])
    return build_portable_qaoa_circuit(J, h, p=1, gamma=np.array([0.4]), beta=np.array([0.3]))


def test_portable_circuit_uses_only_braket_translatable_gates():
    qc = _toy_circuit()
    used = {instr.operation.name for instr in qc.data}
    assert used <= {"h", "rz", "cx", "rx", "measure"}


def test_aer_noiseless_execution_returns_real_counts():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.AER_NOISELESS, shots=256)
    assert sum(result.counts.values()) == 256
    assert result.backend_name == "aer_simulator_noiseless"


def test_aer_synthetic_noisy_execution_returns_noise_descriptor():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.AER_SYNTHETIC_NOISY, shots=256)
    assert sum(result.counts.values()) == 256
    assert result.noise_descriptor is not None
    assert result.noise_descriptor["source"] == "synthetic"


def test_aer_fake_noisy_execution_uses_device_calibrated_noise():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.AER_FAKE_NOISY, shots=256, fake_backend="fake_eagle")
    assert sum(result.counts.values()) == 256
    assert result.backend_name.startswith("Fake")


def test_braket_local_execution_needs_no_aws_credentials():
    qc = _toy_circuit()
    result = execute_on_target(qc, HardwareTarget.BRAKET_LOCAL, shots=128)
    assert sum(result.counts.values()) == 128
    assert result.backend_name == "braket_local_simulator"


def test_ibm_qpu_raises_credential_error_cleanly_with_no_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings

    get_settings(refresh=True)
    qc = _toy_circuit()
    with pytest.raises(CredentialError):
        execute_on_target(qc, HardwareTarget.IBM_QPU, shots=10)


def test_braket_qpu_raises_credential_error_cleanly_with_no_aws_creds(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    from hardware.braket.config import get_aws_credentials

    get_aws_credentials(refresh=True)
    qc = _toy_circuit()
    with pytest.raises(CredentialError):
        execute_on_target(qc, HardwareTarget.BRAKET_QPU, shots=10, device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1")


def test_braket_qpu_requires_device_arn_even_with_credentials():
    from core.errors import ValidationError

    qc = _toy_circuit()
    with pytest.raises((CredentialError, ValidationError)):
        execute_on_target(qc, HardwareTarget.BRAKET_QPU, shots=10, device_arn=None)


def test_benchmark_pair_uses_shared_benchmarks_runner():
    spec = _toy_spec(n=6, k=3, seed=1)
    result = benchmark_pair(spec, SolverChoice.QOKIT_FAST, SolverChoice.CLASSICAL_BASELINE, seed=2)
    assert result.method_a == "qokit_fast"
    assert result.method_b == "classical_baseline"
    assert result.metric_name == "portfolio_qubo_cost"
    assert result.wall_time_a_s is not None and result.wall_time_b_s is not None


def test_run_full_generation_benchmark_logs_every_solver_to_experiment_registry(tmp_path):
    spec = _toy_spec(n=6, k=3, seed=2)
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    results, failures = run_full_generation_benchmark(
        spec, choices=[SolverChoice.QOKIT_FAST, SolverChoice.CLASSICAL_BASELINE, SolverChoice.QAOA_DIRECT],
        seed=3, registry=registry,
    )
    assert failures == {}
    assert len(results) == 3
    assert registry.count() == 3
    for r in results:
        fetched = registry.get(r.metadata.run_id)
        assert fetched.config.application == "portfolio"


def test_run_full_generation_benchmark_records_real_failures_not_silent_skips(tmp_path):
    # n=20 exceeds XY_MIXER_QAOA's documented ~14-qubit statevector ceiling,
    # so it must land in `failures` with a real message, not silently vanish.
    spec = _toy_spec(n=20, k=5, seed=4)
    registry = ExperimentRegistry(db_path=tmp_path / "reg2.sqlite3")
    results, failures = run_full_generation_benchmark(
        spec, choices=[SolverChoice.XY_MIXER_QAOA, SolverChoice.CLASSICAL_BASELINE], seed=5, registry=registry,
    )
    assert "xy_mixer_qaoa" in failures
    assert len(results) == 1 and results[0].metadata.algorithm == "classical_baseline"
