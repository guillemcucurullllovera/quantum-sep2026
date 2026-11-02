from core.schemas import BackendKind
from hardware.backends import BackendCatalog


def test_lists_expected_backends():
    assert set(BackendCatalog.list_backends()) == {"noiseless", "fake_eagle", "fake_heron"}


def test_fake_eagle_is_real_127_qubit_device():
    r = BackendCatalog.get("fake_eagle")
    assert r.descriptor.kind == BackendKind.FAKE
    assert r.descriptor.num_qubits == 127
    assert r.descriptor.coupling_map  # non-empty real topology


def test_fake_heron_is_real_133_qubit_device():
    r = BackendCatalog.get("fake_heron")
    assert r.descriptor.num_qubits == 133


def test_noiseless_has_no_calibration_source():
    r = BackendCatalog.get("noiseless")
    assert r.descriptor.kind == BackendKind.SIMULATOR
    assert r.descriptor.calibration_source is None


def test_unknown_backend_raises():
    from core.errors import BackendUnavailableError
    import pytest

    with pytest.raises(BackendUnavailableError):
        BackendCatalog.get("does_not_exist")
