"""
Tests for hardware.runtime_versions.gen2_native.backend_factory.

The credential-security tests expect `core.errors.CredentialError` (not a
bare `RuntimeError`) because token handling is rewired through core.config
per the platform-wide security rule (see backend_factory.py's module docstring).
CredentialError is a QuantumSep2026Error, not a RuntimeError subclass, so
the original `pytest.raises(RuntimeError, ...)` would no longer catch it —
updating the test is the correct fix here, not the code.

Qiskit-dependent tests are skipped if qiskit is not installed (it is,
in this repo's venv, so they run for real).
"""

import pytest

from core.errors import CredentialError


class TestBackendFactory:
    @pytest.fixture(autouse=True)
    def skip_if_no_qiskit(self):
        pytest.importorskip("qiskit", reason="qiskit not installed")

    def test_statevector_backend_created(self):
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        backend = get_backend("statevector")
        assert backend is not None

    def test_default_mode_is_statevector(self, monkeypatch):
        monkeypatch.delenv("QUANTUM_BACKEND", raising=False)
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        backend = get_backend()
        assert backend is not None

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("QUANTUM_BACKEND", "statevector")
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        backend = get_backend()
        assert backend is not None

    def test_fake_heron_backend(self):
        pytest.importorskip("qiskit_ibm_runtime", reason="qiskit-ibm-runtime not installed")
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        backend = get_backend("fake_heron")
        assert backend is not None

    def test_backend_info_returns_dict(self):
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend, backend_info
        backend = get_backend("statevector")
        info = backend_info(backend)
        assert isinstance(info, dict)
        assert "name" in info
        assert "n_qubits" in info

    def test_invalid_mode_raises(self):
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        with pytest.raises((ValueError, CredentialError)):
            get_backend("nonexistent_mode_xyz")


class TestIBMTokenSecurity:
    """
    SECURITY: IBM_QUANTUM_TOKEN must NEVER be hardcoded.
    Attempting to use ibm_boston without the env var must raise
    core.errors.CredentialError (not a silent no-op, not a generic crash).
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_qiskit(self):
        pytest.importorskip("qiskit", reason="qiskit not installed")

    def test_ibm_boston_raises_without_token(self, monkeypatch):
        monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
        from core.config import get_settings
        get_settings(refresh=True)
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        with pytest.raises(CredentialError, match="IBM_QUANTUM_TOKEN"):
            get_backend("ibm_boston")

    def test_ibm_boston_raises_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("IBM_QUANTUM_TOKEN", "")
        from core.config import get_settings
        get_settings(refresh=True)
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        with pytest.raises(CredentialError, match="IBM_QUANTUM_TOKEN"):
            get_backend("ibm_boston")


class TestBackendEnvVar:
    """QUANTUM_BACKEND env var must control backend selection without code changes."""

    @pytest.fixture(autouse=True)
    def skip_if_no_qiskit(self):
        pytest.importorskip("qiskit", reason="qiskit not installed")

    def test_statevector_via_env(self, monkeypatch):
        monkeypatch.setenv("QUANTUM_BACKEND", "statevector")
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        backend = get_backend()
        assert backend is not None

    def test_explicit_arg_overrides_env(self, monkeypatch):
        """Explicit mode arg takes precedence over env var."""
        monkeypatch.setenv("QUANTUM_BACKEND", "fake_heron")
        from hardware.runtime_versions.gen2_native.backend_factory import get_backend
        backend = get_backend("statevector")
        assert backend is not None
