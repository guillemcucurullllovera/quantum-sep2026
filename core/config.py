"""Environment-only configuration.

ABSOLUTE RULE: credentials are read from `os.environ` at call time. There is
no hardcoded fallback value for any credential anywhere in this module, and
none may be added. See .env.example for the full list of recognized
variables and README.md "Credentials and security" section for the
project-wide policy this enforces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core.errors import CredentialError


@dataclass(frozen=True)
class IBMQuantumCredentials:
    token: str | None
    channel: str
    instance: str | None

    @property
    def is_present(self) -> bool:
        return bool(self.token)

    def require(self) -> "IBMQuantumCredentials":
        """Return self if credentials are present, else raise CredentialError.

        Call this at the point of actually submitting hardware work — never
        earlier, so that simulator-only workflows never need a token.
        """
        if not self.is_present:
            raise CredentialError(
                "IBM_QUANTUM_TOKEN is not set. Hardware/Runtime execution "
                "requires it to be exported in the environment (see "
                ".env.example). Simulator and fake-backend workflows do not "
                "need it — if you only meant to run a simulation, use a "
                "BackendKind.SIMULATOR or BackendKind.FAKE target instead."
            )
        return self


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    results_dir: Path
    registry_db: Path
    log_level: str
    ibm: IBMQuantumCredentials

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("QUANTUM_SEP2026_DATA_DIR", "./data")),
            results_dir=Path(os.environ.get("QUANTUM_SEP2026_RESULTS_DIR", "./results")),
            registry_db=Path(
                os.environ.get("QUANTUM_SEP2026_REGISTRY_DB", "./core/.registry.sqlite3")
            ),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            ibm=IBMQuantumCredentials(
                token=os.environ.get("IBM_QUANTUM_TOKEN") or None,
                channel=os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"),
                instance=os.environ.get("IBM_QUANTUM_INSTANCE") or None,
            ),
        )


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide lazily-initialized settings singleton.

    `refresh=True` re-reads the environment (mainly useful in tests that
    monkeypatch os.environ).
    """
    global _settings
    if _settings is None or refresh:
        _settings = Settings.from_env()
    return _settings
