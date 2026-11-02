"""Backend capability normalization: one schema (`BackendCapabilities`)
describing what a backend can run, computed uniformly whether the backend
is an IBM fake backend/simulator (`qiskit.providers.BackendV2`, reached via
`hardware.backends.catalog.BackendCatalog`) or Braket's local simulator
(reached via `hardware.braket.backend_adapter`) — so application code can
ask "can this circuit run here" and "how many shots / how many qubits can
I ask for" without branching on which cloud/SDK actually produced the
backend object.

This is deliberately a read-only, side-effect-free normalization layer: it
never imports credentials and never talks to a network. Real-QPU capability
queries would need a live `backend.target`/AWS device-capabilities call,
which is out of scope here (this platform's existing IBM QPU path always
requires `core.config.get_settings().ibm.require()` first — see
`hardware.ibm_runtime.primitives.run_sampler`).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import BackendDescriptor, BackendKind
from hardware.backends.catalog import ResolvedBackend
from hardware.braket.backend_adapter import SUPPORTED_QISKIT_GATE_NAMES

# Braket's `LocalSimulator` runs a dense statevector simulation with no
# device-imposed qubit ceiling; this is the same kind of honest "practical
# ceiling, not a hard limit" documented for the noiseless Aer entry in
# `hardware.backends.catalog._load_noiseless` (32 qubits there) — Braket's
# default simulator is somewhat more memory-hungry per qubit in practice, so
# a slightly lower practical ceiling is used here.
BRAKET_LOCAL_SIMULATOR_PRACTICAL_QUBIT_CEILING = 25

# Braket's LocalSimulator has no platform-imposed shot ceiling (it's a local
# process, not a queued cloud job); this is a generous, purely practical
# default so `max_shots` is still a meaningful, finite number to compare
# against IBM's runtime ceiling rather than `None` (which would silently
# defeat any caller that does `shots <= capabilities.max_shots`).
BRAKET_LOCAL_SIMULATOR_PRACTICAL_MAX_SHOTS = 1_000_000

# IBM Quantum Runtime's documented per-job shot ceiling for Sampler/Estimator
# primitives (both real QPU and local fake-backend execution routed through
# this platform's `hardware.ibm_runtime.primitives`).
IBM_RUNTIME_MAX_SHOTS = 100_000


@dataclass(frozen=True)
class BackendCapabilities:
    """Cloud-agnostic description of what a backend can run.

    `native_gates=None` means "no fixed vocabulary restriction — the
    execution path transpiles arbitrary gates onto this backend's basis"
    (true for every IBM path in this platform, which always transpiles
    before running). `native_gates` being a concrete frozenset means the
    execution path is a fixed translation table that REJECTS gates outside
    it rather than transpiling around them (true for
    `hardware.braket.backend_adapter`, whose vocabulary is exactly
    `SUPPORTED_QISKIT_GATE_NAMES`).
    """

    name: str
    provider: str  # "ibm" | "aws" | "local"
    cloud: str  # "ibm_quantum" | "aws_braket_local" | "local"
    num_qubits: int
    native_gates: frozenset[str] | None
    coupling_map: tuple[tuple[int, int], ...] | None  # None = no topology restriction reported
    supports_mid_circuit_measurement: bool
    max_shots: int | None  # None = no platform-imposed ceiling known
    is_simulator: bool
    is_noisy: bool
    calibration_source: str | None = None


def capabilities_from_ibm_backend(resolved: ResolvedBackend) -> BackendCapabilities:
    """Normalize a `hardware.backends.catalog.ResolvedBackend` (fake backend
    or the noiseless Aer simulator) into `BackendCapabilities`.

    `native_gates=None` here reflects reality: every IBM-path execution
    routine in this platform (`hardware.ibm_runtime.primitives.run_sampler`/
    `run_estimator`) calls `qiskit.transpile(circuit, backend=backend, ...)`
    before running, so an arbitrary input gate set is not a rejection
    condition the way it is for the fixed Braket translation table —
    `basis_gates` is still reported (for callers that want it) but is not
    used as a hard compatibility gate the way Braket's vocabulary is.
    """
    descriptor: BackendDescriptor = resolved.descriptor
    coupling = tuple(descriptor.coupling_map) if descriptor.coupling_map else None
    return BackendCapabilities(
        name=descriptor.name,
        provider="ibm",
        cloud="ibm_quantum",
        num_qubits=descriptor.num_qubits,
        native_gates=None,
        coupling_map=coupling,
        # Real IBM backends and Qiskit's fake-backend/Aer simulation stack
        # both support mid-circuit measurement / dynamic circuits (IfElseOp,
        # measure-then-reset, etc.) — this is a genuine execution-path
        # capability, not merely a translation-table gap.
        supports_mid_circuit_measurement=True,
        max_shots=IBM_RUNTIME_MAX_SHOTS,
        is_simulator=descriptor.kind != BackendKind.QPU,
        is_noisy=descriptor.kind == BackendKind.FAKE,
        calibration_source=descriptor.calibration_source,
    )


def capabilities_from_braket_local_simulator() -> BackendCapabilities:
    """Normalize Braket's `LocalSimulator` (as reached through
    `hardware.braket.backend_adapter.run_local`) into `BackendCapabilities`.

    Unlike the IBM path, `native_gates` here is a real, finite restriction:
    `qiskit_to_braket` raises `UnsupportedGateError` for anything outside
    `SUPPORTED_QISKIT_GATE_NAMES` rather than transpiling around it — so
    this is genuinely narrower than the IBM path's "anything goes" answer,
    and `supports_mid_circuit_measurement=False` because
    `qiskit_to_braket` DROPS measurement instructions entirely (see its
    docstring) rather than deferring or rerouting them — there is no
    mid-circuit conditioning possible through this translation layer at all.
    """
    return BackendCapabilities(
        name="braket_local_simulator",
        provider="aws",
        cloud="aws_braket_local",
        num_qubits=BRAKET_LOCAL_SIMULATOR_PRACTICAL_QUBIT_CEILING,
        native_gates=SUPPORTED_QISKIT_GATE_NAMES,
        coupling_map=None,  # all-to-all: a local statevector simulator has no topology
        supports_mid_circuit_measurement=False,
        max_shots=BRAKET_LOCAL_SIMULATOR_PRACTICAL_MAX_SHOTS,
        is_simulator=True,
        is_noisy=False,
    )


__all__ = [
    "BackendCapabilities",
    "capabilities_from_ibm_backend",
    "capabilities_from_braket_local_simulator",
    "IBM_RUNTIME_MAX_SHOTS",
    "BRAKET_LOCAL_SIMULATOR_PRACTICAL_QUBIT_CEILING",
    "BRAKET_LOCAL_SIMULATOR_PRACTICAL_MAX_SHOTS",
]
