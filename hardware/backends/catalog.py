"""Backend catalog: a single place that knows about every simulator and
fake-backend the platform can target, producing typed
`core.schemas.BackendDescriptor` records rather than raw Qiskit objects
leaking through every call site.

Ported and extended from Generation 1
(the Gen1 research repository) — the
prioritized fake-backend-name-fallback pattern is preserved because
qiskit-ibm-runtime's fake-backend class names do drift across releases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.errors import BackendUnavailableError
from core.schemas import BackendDescriptor, BackendKind

_EAGLE_CANDIDATES: tuple[str, ...] = ("FakeSherbrooke", "FakeKyoto", "FakeOsaka", "FakeBrisbane")
_HERON_CANDIDATES: tuple[str, ...] = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez")


@dataclass(frozen=True)
class ResolvedBackend:
    descriptor: BackendDescriptor
    backend: object  # the underlying qiskit BackendV2 instance


def _load_fake(candidates: tuple[str, ...], family: str) -> ResolvedBackend:
    try:
        import qiskit_ibm_runtime.fake_provider as fp
    except ImportError as exc:
        raise BackendUnavailableError(
            "qiskit-ibm-runtime is required for fake backends."
        ) from exc

    tried = []
    for name in candidates:
        cls = getattr(fp, name, None)
        if cls is not None:
            backend = cls()
            coupling = None
            try:
                coupling = [tuple(e) for e in backend.coupling_map.get_edges()]
            except Exception:
                pass
            descriptor = BackendDescriptor(
                name=name,
                kind=BackendKind.FAKE,
                num_qubits=backend.num_qubits,
                provider="ibm",
                coupling_map=coupling,
                basis_gates=sorted(backend.target.operation_names),
                calibration_source=f"fake_backend:{name} ({family})",
            )
            return ResolvedBackend(descriptor=descriptor, backend=backend)
        tried.append(name)

    available = sorted(n for n in dir(fp) if n.startswith("Fake"))
    raise BackendUnavailableError(
        f"None of the candidate {family} fake backends {tried} are present in "
        f"this qiskit-ibm-runtime version. Available: {available}"
    )


def _load_noiseless() -> ResolvedBackend:
    from qiskit_aer import AerSimulator

    backend = AerSimulator()
    descriptor = BackendDescriptor(
        name="aer_simulator_noiseless",
        kind=BackendKind.SIMULATOR,
        num_qubits=32,  # practical statevector-simulation ceiling, not a hard limit
        provider="local",
        calibration_source=None,
    )
    return ResolvedBackend(descriptor=descriptor, backend=backend)


_REGISTRY: dict[str, Callable[[], ResolvedBackend]] = {
    "noiseless": _load_noiseless,
    "fake_eagle": lambda: _load_fake(_EAGLE_CANDIDATES, "Eagle"),
    "fake_heron": lambda: _load_fake(_HERON_CANDIDATES, "Heron"),
}


class BackendCatalog:
    """Entry point every other hardware/ submodule and algorithm should use
    to obtain a backend — never `qiskit_ibm_runtime.fake_provider.FakeX()`
    directly, so backend selection stays centralized and typed."""

    @staticmethod
    def list_backends() -> list[str]:
        return sorted(_REGISTRY)

    @staticmethod
    def get(name: str) -> ResolvedBackend:
        key = name.strip().lower()
        if key not in _REGISTRY:
            raise BackendUnavailableError(
                f"Unknown backend {name!r}. Registered: {BackendCatalog.list_backends()}"
            )
        return _REGISTRY[key]()


__all__ = ["BackendCatalog", "ResolvedBackend"]
