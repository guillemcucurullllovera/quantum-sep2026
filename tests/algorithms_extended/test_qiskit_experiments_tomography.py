"""tests: the vendored qiskit-experiments tomography module
(`algorithms_extended.qiskit_experiments_asymmetrica.library.tomography`)
actually running real state tomography -- against fake/noisy backends,
using the default `linear_inversion` fitter (cvxpy is not installed in this
environment) -- and the platform
integration layer (`tomography_integration.py`) that wires it into
`core.experiment_registry` and `hardware.provenance`.
"""
from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from core.experiment_registry import ExperimentRegistry
from hardware.backends.catalog import BackendCatalog

from algorithms_extended.qiskit_experiments_asymmetrica.library.tomography import (
    StateTomography,
    ProcessTomography,
    StateTomographyAnalysis,
    ProcessTomographyAnalysis,
)
from algorithms_extended.qiskit_experiments_asymmetrica.library.tomography.fitters import (
    linear_inversion,
)
from algorithms_extended.qiskit_experiments_asymmetrica.tomography_integration import (
    run_state_tomography_diagnostic,
    run_qaoa_circuit_fidelity_diagnostic,
    circuit_fidelity_vs_noise_level,
)


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture(scope="module")
def fake_backend():
    return BackendCatalog().get("fake_eagle").backend


def test_vendored_tomography_imports_cleanly():
    assert StateTomography is not None
    assert ProcessTomography is not None
    assert StateTomographyAnalysis is not None
    assert ProcessTomographyAnalysis is not None
    assert linear_inversion is not None


def test_state_tomography_recovers_bell_state_high_fidelity_noiseless():
    """Real correctness check: linear-inversion state tomography on a Bell
    state, noiseless simulator, must recover fidelity > 0.9 -- this is the
    load-bearing check that the vendored reconstruction machinery actually
    works, not an import-only smoke test."""
    from qiskit.quantum_info import Statevector, state_fidelity

    qc = _bell_circuit()
    ideal = Statevector(qc)
    sim = AerSimulator()
    exp = StateTomography(qc, backend=sim)
    data = exp.run(shots=4000, seed_simulation=123).block_for_results()
    rho = data.analysis_results("state", dataframe=False).value
    fidelity = state_fidelity(rho, ideal)
    assert fidelity > 0.9


def test_state_tomography_recovers_bell_state_on_fake_backend(fake_backend):
    """Same check but against this platform's own real fake-backend noise
    model (light IBM Eagle-family noise), not an idealized simulator."""
    from qiskit_aer import AerSimulator as _Aer
    from qiskit.quantum_info import Statevector, state_fidelity

    qc = _bell_circuit()
    ideal = Statevector(qc)
    sim = _Aer.from_backend(fake_backend)
    exp = StateTomography(qc, physical_qubits=(0, 1), backend=sim)
    data = exp.run(shots=4000, seed_simulation=321).block_for_results()
    rho = data.analysis_results("state", dataframe=False).value
    fidelity = state_fidelity(rho, ideal)
    assert fidelity > 0.9


def test_fidelity_vs_noise_level_is_monotonic():
    """Controlled correctness check: fidelity must decrease monotonically as
    depolarizing noise increases, across at least two distinct noise
    levels, plus a zero-noise control point."""
    qc = _bell_circuit()
    points = circuit_fidelity_vs_noise_level(
        qc, [0.0, 0.03, 0.10], shots=4000, seed_simulation=99
    )
    fidelities = [p.fidelity for p in points]
    assert len(fidelities) == 3
    # each point is itself a real, bounded fidelity
    for f in fidelities:
        assert 0.0 <= f <= 1.0 + 1e-9
    # monotonically non-increasing as noise increases
    assert fidelities[0] > fidelities[1] > fidelities[2]
    # noiseless control point should itself reconstruct with high fidelity
    assert fidelities[0] > 0.9


def test_run_state_tomography_diagnostic_records_result_and_provenance(fake_backend, tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    qc = _bell_circuit()
    result, prov = run_state_tomography_diagnostic(
        qc, fake_backend, physical_qubits=(0, 1), registry=registry,
        algorithm_name="bell_state_diagnostic_test",
    )
    assert 0.0 <= result.metrics["reconstructed_fidelity"] <= 1.0
    assert result.metrics["reconstructed_fidelity"] > 0.85
    assert result.metadata.adapted_from == "qiskit-experiments (Apache-2.0)"

    # round-trips through core.experiment_registry
    run_ids = registry.list_by_algorithm("bell_state_diagnostic_test")
    assert len(run_ids) >= 1
    fetched = registry.get(run_ids[0])
    assert fetched.metrics["reconstructed_fidelity"] == result.metrics["reconstructed_fidelity"]

    # hardware.provenance integration point
    assert prov.algorithm == "bell_state_diagnostic_test"
    assert prov.backend_name == getattr(fake_backend, "name", None)
    assert prov.reproducible is True
    assert prov.missing_fields == []


def test_qaoa_circuit_fidelity_diagnostic_runs_on_real_platform_circuit(fake_backend, tmp_path):
    """Materially-integrated diagnostic: real state tomography on the output
    of this platform's own Qiskit QAOA circuit
    (algorithms.qaoa_qubo.solvers._build_qiskit_qaoa_circuit), executed
    against a noisy simulator built from a real fake backend's calibration
    data."""
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result, prov = run_qaoa_circuit_fidelity_diagnostic(
        fake_backend, n_assets=3, p_layers=1, registry=registry,
    )
    fidelity = result.metrics["reconstructed_fidelity"]
    assert 0.0 <= fidelity <= 1.0
    # a 3-qubit, p=1 QAOA circuit under light Eagle-family noise should still
    # preserve substantial state fidelity
    assert fidelity > 0.5
    assert result.metrics["num_qubits"] == 3.0
    assert prov.reproducible is True
