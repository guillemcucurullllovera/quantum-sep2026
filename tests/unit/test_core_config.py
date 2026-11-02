import os

import pytest

from core.config import Settings
from core.errors import CredentialError


def test_settings_from_env_no_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    s = Settings.from_env()
    assert s.ibm.is_present is False
    with pytest.raises(CredentialError):
        s.ibm.require()


def test_settings_from_env_with_token(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "fake-token-for-test-only")
    s = Settings.from_env()
    assert s.ibm.is_present is True
    assert s.ibm.require().token == "fake-token-for-test-only"


def test_no_hardcoded_credential_fallback():
    """Regression guard: config.py source must not contain a plausible
    hardcoded secret literal as a fallback value."""
    import inspect

    import core.config as config_mod

    src = inspect.getsource(config_mod)
    # the only occurrences of "token" should be as parameter/attribute names,
    # never assigned a non-empty string literal fallback
    assert 'os.environ.get("IBM_QUANTUM_TOKEN")' in src
    assert 'os.environ.get("IBM_QUANTUM_TOKEN", "' not in src
