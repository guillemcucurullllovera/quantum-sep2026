import tempfile
from pathlib import Path

import pytest

from hardware.backends.catalog import BackendCatalog
from hardware.calibration import CalibrationHistory, extract_calibration


@pytest.fixture()
def fake_backend():
    return BackendCatalog.get("fake_eagle").backend


def test_extract_calibration_returns_real_nonzero_numbers(fake_backend):
    snap = extract_calibration(fake_backend)
    assert snap.backend_name == fake_backend.name
    assert snap.n_qubits == fake_backend.num_qubits
    assert len(snap.t1_us) > 0
    assert all(v > 0 for v in snap.t1_us.values())
    assert snap.t1_mean_us is not None and snap.t1_mean_us > 0
    assert snap.two_qubit_error_samples > 0
    assert 0 < snap.two_qubit_error_mean < 1


def test_calibration_history_records_and_lists(fake_backend):
    with tempfile.TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "reg.sqlite3"
        from core.experiment_registry import ExperimentRegistry

        history = CalibrationHistory(registry=ExperimentRegistry(db_path=registry_path))
        snap1 = history.record(fake_backend)
        snap2 = history.record(fake_backend)
        snaps = history.list_snapshots(fake_backend.name)
        assert len(snaps) == 2
        assert snaps[0].captured_at <= snaps[1].captured_at


def test_drift_detects_real_change_between_snapshots(fake_backend):
    with tempfile.TemporaryDirectory() as tmp:
        registry_path = Path(tmp) / "reg.sqlite3"
        from core.experiment_registry import ExperimentRegistry

        history = CalibrationHistory(registry=ExperimentRegistry(db_path=registry_path))
        drift_empty = history.drift(fake_backend.name)
        assert drift_empty["drift_detectable"] is False

        history.record(fake_backend)
        history.record(fake_backend)
        drift = history.drift(fake_backend.name)
        assert drift["drift_detectable"] is True
        assert "t1_mean_us_delta" in drift
        # same backend snapshotted twice with no time passing -> delta should be exactly 0
        assert drift["t1_mean_us_delta"] == pytest.approx(0.0)
