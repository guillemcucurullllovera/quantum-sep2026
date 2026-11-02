from __future__ import annotations

from dataclasses import dataclass, field

from core.config import get_settings
from core.schemas import BackendDescriptor, BackendKind
from hardware.backends.catalog import BackendCatalog


@dataclass
class DiscoveryResult:
    live: bool  # True only for a real QiskitRuntimeService query that succeeded
    backends: list[BackendDescriptor] = field(default_factory=list)
    note: str = ""


def _fake_catalog_fallback(note: str) -> DiscoveryResult:
    names = [n for n in BackendCatalog.list_backends()]
    return DiscoveryResult(
        live=False,
        backends=[BackendCatalog.get(n).descriptor for n in names],
        note=note,
    )


def discover_backends() -> DiscoveryResult:
    """Enumerate backends reachable right now.

    With no `IBM_QUANTUM_TOKEN` set, or if the live query fails for any
    reason, falls back to the fake-backend catalog rather than raising —
    discovery is meant to be a safe default entry point, unlike
    `hardware.qpu_execution`, which must fail loudly when asked to actually
    *run* something without credentials.
    """
    settings = get_settings()
    if not settings.ibm.is_present:
        return _fake_catalog_fallback(
            "No IBM_QUANTUM_TOKEN set — returning the fake-backend catalog, "
            "not a live discovery result."
        )
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        return _fake_catalog_fallback(
            "qiskit-ibm-runtime not installed — falling back to fake-backend catalog."
        )
    try:
        service = QiskitRuntimeService(
            channel=settings.ibm.channel, token=settings.ibm.token, instance=settings.ibm.instance
        )
        live_backends = service.backends()
    except Exception as exc:  # network/auth/service errors of any shape
        return _fake_catalog_fallback(
            f"Live discovery failed ({exc.__class__.__name__}) — falling back to fake-backend catalog."
        )

    descriptors = []
    for b in live_backends:
        try:
            coupling = [tuple(e) for e in b.coupling_map.get_edges()]
        except Exception:
            coupling = None
        descriptors.append(
            BackendDescriptor(
                name=b.name,
                kind=BackendKind.QPU,
                num_qubits=b.num_qubits,
                provider="ibm",
                coupling_map=coupling,
                basis_gates=sorted(b.target.operation_names),
                calibration_source=f"live:{b.name}",
            )
        )
    return DiscoveryResult(live=True, backends=descriptors, note=f"Live discovery returned {len(descriptors)} backend(s).")
