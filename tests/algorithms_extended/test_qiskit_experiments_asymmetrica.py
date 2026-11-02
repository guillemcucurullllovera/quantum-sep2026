"""Real tests for the qiskit-experiments full-base adaptation: the vendored
framework/characterization/RB/curve_analysis code actually running against
this platform's fake-backend catalog, not import-only smoke tests."""
from __future__ import annotations

import pytest

from core.experiment_registry import ExperimentRegistry
from hardware.backends.catalog import BackendCatalog
from hardware.calibration.history import CalibrationHistory

from algorithms_extended.qiskit_experiments_asymmetrica.integration import (
    run_t1_experiment,
    run_t2_experiment,
    run_standard_rb,
)


@pytest.fixture(scope="module")
def fake_backend():
    return BackendCatalog().get("fake_eagle").backend


def test_vendored_framework_imports_cleanly():
    from algorithms_extended.qiskit_experiments_asymmetrica.framework import (
        BaseExperiment,
        BaseAnalysis,
        ExperimentData,
    )

    assert BaseExperiment is not None
    assert BaseAnalysis is not None
    assert ExperimentData is not None


def test_t1_measured_value_matches_declared_within_tolerance(fake_backend, tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result = run_t1_experiment(fake_backend, physical_qubit=0, registry=registry)
    assert result.metrics["measured_t1_us"] > 0
    # a real fitted T1 from noisy simulation should land within ~15% of the
    # backend's own declared value -- this is the load-bearing correctness
    # check that the vendored curve-fitting machinery actually works
    assert result.metrics["relative_error"] < 0.15


def test_t2_measured_value_matches_declared_within_tolerance(fake_backend, tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result = run_t2_experiment(fake_backend, physical_qubit=0, registry=registry)
    assert result.metrics["measured_t2_us"] > 0
    assert result.metrics["relative_error"] < 0.15


def test_standard_rb_produces_bounded_fidelity(fake_backend, tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result = run_standard_rb(
        fake_backend, physical_qubits=(0,), lengths=[1, 10, 30, 50], num_samples=3, registry=registry
    )
    epc = result.metrics["error_per_clifford"]
    fidelity = result.metrics["fidelity_per_clifford"]
    assert 0.0 <= epc <= 1.0
    assert 0.0 <= fidelity <= 1.0
    assert abs((1.0 - epc) - fidelity) < 1e-9
    # a real (not fabricated) low-noise fake backend should show high per-Clifford fidelity
    assert fidelity > 0.9


def test_t1_result_feeds_calibration_history(fake_backend, tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    run_t1_experiment(fake_backend, physical_qubit=0, registry=registry)
    ch = CalibrationHistory(registry=registry)
    snaps = ch.list_snapshots(fake_backend.name)
    assert len(snaps) >= 1
    assert snaps[-1].t1_mean_us is not None
    assert snaps[-1].t1_mean_us > 0


def test_experiment_result_recorded_and_retrievable(fake_backend, tmp_path):
    registry = ExperimentRegistry(db_path=tmp_path / "reg.sqlite3")
    result = run_t1_experiment(fake_backend, physical_qubit=1, registry=registry)
    run_ids = registry.list_by_algorithm("qiskit_experiments_t1")
    assert len(run_ids) >= 1
    fetched = registry.get(run_ids[0])
    assert fetched.metadata.algorithm == "qiskit_experiments_t1"
    assert fetched.metadata.adapted_from == "qiskit-experiments (Apache-2.0)"
