"""Discoverable, single-purpose entry point for fake (device-calibrated-noise)
backends specifically — a narrower door than `hardware.backends.BackendCatalog`
for callers (especially ported historical/prototype code) that only ever want
"give me a fake backend with at least N qubits," without reaching into the
catalog's general simulator+fake registry.

Thin wrapper, not a duplicate: `list_fake_backends` and `resolve` both delegate
to `hardware.backends.catalog.BackendCatalog` — no backend-loading logic is
reimplemented here.
"""
from __future__ import annotations

from core.errors import BackendUnavailableError
from core.schemas import BackendDescriptor, BackendKind
from hardware.backends.catalog import BackendCatalog, ResolvedBackend

_FAKE_KEYS: tuple[str, ...] = ("fake_eagle", "fake_heron")


def list_fake_backends(min_qubits: int | None = None) -> list[BackendDescriptor]:
    """Resolve every registered fake backend and return its descriptor,
    optionally filtered to backends with at least `min_qubits` qubits.

    Resolves each candidate (not just names) so `min_qubits` filtering is
    against real qubit counts, not guessed from a name.
    """
    descriptors: list[BackendDescriptor] = []
    for key in _FAKE_KEYS:
        try:
            resolved = BackendCatalog.get(key)
        except BackendUnavailableError:
            continue
        if min_qubits is None or resolved.descriptor.num_qubits >= min_qubits:
            descriptors.append(resolved.descriptor)
    return descriptors


def resolve(family: str = "fake_heron") -> ResolvedBackend:
    """`family` is 'fake_eagle' or 'fake_heron'. Raises BackendUnavailableError
    (never returns a silently-different backend) if the requested family
    isn't installed in this qiskit-ibm-runtime version."""
    if family not in _FAKE_KEYS:
        raise BackendUnavailableError(f"Unknown fake-backend family {family!r}. Known: {_FAKE_KEYS}")
    return BackendCatalog.get(family)


__all__ = ["list_fake_backends", "resolve"]
