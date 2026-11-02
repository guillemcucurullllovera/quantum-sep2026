from __future__ import annotations

from enum import Enum


class SimulatorMethod(str, Enum):
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    MATRIX_PRODUCT_STATE = "matrix_product_state"
    STABILIZER = "stabilizer"


_CLIFFORD_GATES = {"x", "y", "z", "h", "s", "sdg", "cx", "cy", "cz", "swap", "id", "measure", "barrier", "reset"}


def _is_clifford_only(circuit) -> bool:
    return all(instr.operation.name in _CLIFFORD_GATES for instr in circuit.data)


def recommend_method(
    circuit,
    *,
    has_noise_model: bool = False,
    max_statevector_qubits: int = 24,
) -> SimulatorMethod:
    """Pick a simulation method based on real circuit/noise characteristics,
    not a hardcoded default. This is a heuristic, documented as such — not
    a guarantee of optimality, but a principled default better than always
    using "automatic" (which Aer resolves opaquely).
    """
    n = circuit.num_qubits
    if _is_clifford_only(circuit):
        return SimulatorMethod.STABILIZER
    if has_noise_model and n > max_statevector_qubits:
        # exact density-matrix noise sim is 2x the qubit exponent of
        # statevector — beyond this size it's usually worse than
        # matrix_product_state for the entanglement typical of variational
        # circuits, so prefer MPS once we're already past the point where
        # density_matrix would be comfortable.
        return SimulatorMethod.MATRIX_PRODUCT_STATE
    if has_noise_model:
        return SimulatorMethod.DENSITY_MATRIX
    if n > max_statevector_qubits:
        return SimulatorMethod.MATRIX_PRODUCT_STATE
    return SimulatorMethod.STATEVECTOR


def build_simulator(method: SimulatorMethod, noise_model=None):
    """Construct an AerSimulator configured for the given method — the
    single place this platform should call `AerSimulator(...)` with an
    explicit method, rather than relying on Aer's opaque "automatic" choice.
    """
    from qiskit_aer import AerSimulator

    kwargs = {"method": method.value}
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    return AerSimulator(**kwargs)
