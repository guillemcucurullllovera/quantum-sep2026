"""Qiskit <-> Braket circuit translation and execution.

Honest scope: this covers the common single- and two-qubit gate vocabulary
that this platform's algorithms actually use (H, X, Y, Z, RX, RY, RZ, CNOT/
CX, CZ, SWAP, S, T, and their inverses, plus the two-qubit-parametrized RZZ
and controlled-phase (CP) gates — see below) — the intersection of what
`qiskit.QuantumCircuit` and `braket.circuits.Circuit` both represent
natively. It does NOT attempt full bidirectional coverage of every gate
either SDK supports (e.g. Qiskit's `UGate`/custom unitaries, or Braket's
pulse-level/analog-Hamiltonian-simulation (`ahs`) capabilities) — that would
be a much larger undertaking. Circuits using gates outside this vocabulary
raise `UnsupportedGateError` rather than silently producing a wrong circuit.

addition — RZZ / CP: a grep of this platform's actual algorithm
circuit-builders (`algorithms/qaoa_qubo/solvers.py::_build_qiskit_qaoa_circuit`,
`algorithms/qaoa_qubo/solvers.py` QUBO-cost path in general, and
`algorithms/pqk/feature_maps.py`'s ZZ-entangling feature maps) found real,
load-bearing use of `QuantumCircuit.rzz` (the Ising ZZ-interaction gate QAOA
cost layers are built from) and `QuantumCircuit.cp` (the controlled-phase
gate PQK's entangling feature maps use) — neither of which the original
single-qubit-or-plain-two-qubit `_GATE_MAP` shape could represent. Braket's
`Circuit.zz(q0, q1, angle)` and `Circuit.cphaseshift(control, target, angle)`
are the exact same unitaries (verified numerically against
`qiskit.quantum_info.Statevector` to machine precision — see
`test_rzz_and_cp_translation_matches_qiskit_exactly` in this module's test
file), so both are now translated rather than rejected. Gates not actually
emitted anywhere in this platform's algorithms (e.g. Braket's native `xx`/
`yy`/`ecr`/`ms` family) are intentionally still out of scope — adding
translations nothing calls would just be unverified surface area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from core.errors import QuantumSep2026Error

from .config import get_aws_credentials

try:
    # Import the real, pip-installed `braket` package (amazon-braket-sdk +
    # amazon-braket-default-simulator + amazon-braket-schemas — all in
    # pyproject.toml) rather than sys.path-injecting the vendored copy under
    # algorithms_extended/amazon_braket_asymmetrica/. The vendored copy has an
    # `__init__.py` per subpackage, so it resolves as a regular (non-namespace)
    # package the moment Python finds it — which prevents PEP 420 namespace-
    # package merging with the separately-installed default_simulator/schemas
    # packages and breaks `braket.default_simulator` imports. The vendored
    # tree remains in place as the licensed, diffable, attributed source
    # copy (see THIRD_PARTY_NOTICES.md); this runtime path just doesn't
    # import through it.
    from braket.circuits import Circuit as BraketCircuit
    from braket.devices import LocalSimulator
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "amazon-braket-sdk / amazon-braket-default-simulator / "
        "amazon-braket-schemas are required for hardware.braket — install "
        "via the platform's pyproject.toml dependencies."
    ) from e


class UnsupportedGateError(QuantumSep2026Error):
    """Raised when a circuit uses a gate outside the translated vocabulary."""


# Qiskit instruction name -> (Braket Circuit method name, qubit arity, has_angle_param)
_GATE_MAP: dict[str, tuple[str | None, int, bool]] = {
    "h": ("h", 1, False),
    "x": ("x", 1, False),
    "y": ("y", 1, False),
    "z": ("z", 1, False),
    "s": ("s", 1, False),
    "sdg": ("si", 1, False),
    "t": ("t", 1, False),
    "tdg": ("ti", 1, False),
    "rx": ("rx", 1, True),
    "ry": ("ry", 1, True),
    "rz": ("rz", 1, True),
    "cx": ("cnot", 2, False),
    "cz": ("cz", 2, False),
    "swap": ("swap", 2, False),
    "rzz": ("zz", 2, True),           # Ising ZZ interaction — QAOA cost layers
    "cp": ("cphaseshift", 2, True),   # controlled-phase — PQK entangling feature maps
    "id": (None, 1, False),  # no-op, dropped
    "barrier": (None, 1, False),  # no-op, dropped
}

# Every Qiskit instruction name this adapter can either translate or safely
# drop (measurement is handled separately in `qiskit_to_braket`, not via
# `_GATE_MAP`, but is part of the platform-wide "compatible" vocabulary).
# `hardware.circuit_analysis.compatibility` imports this directly rather
# than re-deriving/duplicating the gate list, so the two stay in sync.
SUPPORTED_QISKIT_GATE_NAMES: frozenset[str] = frozenset(_GATE_MAP) | {"measure"}


@dataclass(frozen=True)
class TranslationResult:
    braket_circuit: "BraketCircuit"
    n_qubits: int
    dropped_ops: list[str]


def qiskit_to_braket(qc: QuantumCircuit) -> TranslationResult:
    """Translate a Qiskit circuit into the equivalent Braket circuit, for
    the supported gate vocabulary (see module docstring). Measurement
    instructions are dropped (Braket's local simulator returns full
    statevector/probability results by default, not qasm-style classical
    bits) — callers reading out results should use `run_local`'s return
    value, not assume classical-register semantics survived translation.
    """
    bc = BraketCircuit()
    dropped: list[str] = []
    for instr in qc.data:
        name = instr.operation.name
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        if name == "measure":
            dropped.append(name)
            continue
        if name not in _GATE_MAP:
            raise UnsupportedGateError(
                f"Gate '{name}' has no Braket translation in this platform's "
                f"vocabulary (supported: {sorted(k for k in _GATE_MAP if _GATE_MAP[k][0])})."
            )
        method_name, arity, has_param = _GATE_MAP[name]
        if method_name is None:
            dropped.append(name)
            continue
        method = getattr(bc, method_name)
        params = list(instr.operation.params)
        if arity == 2 and has_param:
            method(qubits[0], qubits[1], float(params[0]))
        elif arity == 2:
            method(qubits[0], qubits[1])
        elif has_param:
            method(qubits[0], float(params[0]))
        else:
            method(qubits[0])
    return TranslationResult(braket_circuit=bc, n_qubits=qc.num_qubits, dropped_ops=dropped)


def run_local(qc: QuantumCircuit, shots: int = 0) -> dict:
    """Execute a Qiskit circuit on Braket's local statevector simulator.
    No AWS credentials needed — this is the offline validation path.

    shots=0 requests exact statevector/probability results (Braket's local
    simulator supports analytic mode); shots>0 requests sampled measurement
    counts instead.
    """
    translation = qiskit_to_braket(qc)
    bc = translation.braket_circuit
    if shots == 0:
        bc = bc.state_vector()  # analytic mode requires an explicit result type
    device = LocalSimulator()
    task = device.run(bc, shots=shots)
    result = task.result()
    out = {
        "n_qubits": translation.n_qubits,
        "dropped_ops": translation.dropped_ops,
        "shots": shots,
    }
    if shots == 0:
        out["state_vector"] = _to_qiskit_qubit_order(
            np.asarray(result.values[0]), translation.n_qubits
        )
    else:
        out["measurement_counts"] = dict(result.measurement_counts)
        # Which original-circuit qubit index each `measurement_counts`
        # bitstring position corresponds to (ascending order, ints) --
        # NOT necessarily `range(n_qubits)`: a qubit with no gate applied
        # to it never enters the translated Braket circuit at all (Braket
        # only tracks qubits actually referenced by an instruction), so a
        # circuit where some declared qubits are idle produces a
        # *shorter* bitstring than `n_qubits`, not a zero-padded one.
        # `hardware.result_processing.braket_normalize` uses this to
        # rebuild a full-width, idle-qubits-are-'0' bitstring in this
        # platform's standard convention -- see that module's docstring.
        out["measured_qubits"] = [int(q) for q in result.measured_qubits]
    return out


def _to_qiskit_qubit_order(statevector: np.ndarray, n_qubits: int) -> np.ndarray:
    """Braket orders basis-state indices with qubit 0 as the MOST
    significant bit; Qiskit orders with qubit 0 as the LEAST significant
    bit. Reverse the per-qubit axis order so results returned by this
    module are directly comparable to `qiskit.quantum_info.Statevector`
    without callers needing to know which SDK produced which convention.

    Caught by this module's own test (`test_local_simulator_statevector_
    matches_qiskit_exactly`): a naive first version returned Braket's raw
    ordering and silently mismatched Qiskit on any circuit whose amplitudes
    weren't symmetric under qubit relabeling (single-qubit circuits and the
    specific 2-qubit case first tested happened to be symmetric and masked
    the bug).
    """
    reshaped = statevector.reshape((2,) * n_qubits)
    reversed_axes = reshaped.transpose(list(reversed(range(n_qubits))))
    return reversed_axes.reshape(-1)


def run_remote(qc: QuantumCircuit, device_arn: str, shots: int = 100):
    """Execute on a real AWS Braket QPU or managed simulator. Requires AWS
    credentials — raises CredentialError before any network call if absent.
    """
    get_aws_credentials().require()
    # Import here (not at module load) so the credential-gate test doesn't
    # need boto3-network-capable AWS SDK state to exercise the gate itself.
    from braket.aws import AwsDevice

    translation = qiskit_to_braket(qc)
    device = AwsDevice(device_arn)
    task = device.run(translation.braket_circuit, shots=shots)
    result = task.result()
    return {
        "n_qubits": translation.n_qubits,
        "dropped_ops": translation.dropped_ops,
        "shots": shots,
        "measurement_counts": dict(result.measurement_counts),
    }
