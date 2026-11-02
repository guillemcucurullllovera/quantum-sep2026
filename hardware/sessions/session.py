"""Unified session context manager: a simulator session is a no-op context
(nothing to open/close), a real Runtime session wraps
`qiskit_ibm_runtime.Session` — same call site either way.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import BackendDescriptor, BackendKind


@dataclass
class PlatformSession:
    backend_descriptor: BackendDescriptor
    _runtime_session: object = None

    def __enter__(self) -> "PlatformSession":
        if self.backend_descriptor.kind == BackendKind.QPU:
            from core.config import get_settings
            from qiskit_ibm_runtime import QiskitRuntimeService, Session

            creds = get_settings().ibm.require()
            service = QiskitRuntimeService(channel=creds.channel, token=creds.token, instance=creds.instance)
            backend = service.backend(self.backend_descriptor.name)
            self._runtime_session = Session(backend=backend)
            self._runtime_session.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._runtime_session is not None:
            self._runtime_session.__exit__(exc_type, exc, tb)

    @property
    def is_live(self) -> bool:
        return self._runtime_session is not None


__all__ = ["PlatformSession"]
