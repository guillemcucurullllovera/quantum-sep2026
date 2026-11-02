"""
Minimal 8-Qubit VQC — Phase 1A Smoke Test
==========================================
Pure-numpy statevector simulator for the first variational quantum circuit.

This module proves the data contract works end-to-end through quantum encoding:
  QuantumInputVector.to_reduced_8q() → angle encoding → variational ansatz → measurement

Architecture (8 qubits):
  1. ENCODING LAYER: RY(feature_i × π) on qubit i  (angle encoding)
  2. ENTANGLEMENT: Linear CNOT chain (0→1, 1→2, ..., 6→7)
  3. VARIATIONAL LAYER: RY(θ_i) on each qubit (8 trainable params)
  4. MEASUREMENT: ⟨Z⟩ expectation on qubit 0

Why numpy-only:
  - 8 qubits = 256 complex amplitudes — trivial for classical simulation
  - Zero dependency beyond numpy (no qiskit/pennylane install needed)
  - Exact statevector (no shot noise)
  - Easily upgradeable to qiskit/pennylane later

Contract compliance:
  - Consumes exactly the 8 floats from to_reduced_8q()
  - Indices [0, 2, 3, 5, 7, 8, 10, 13] of the frozen 14-feature vector
  - Encoding: RY(feature × π) maps [−1, 1] → [−π, π] rotation
  - No changes to QuantumInputVector v1.0

Usage:
    from quantum_signal.circuit import VQCSmokeTest

    vqc = VQCSmokeTest()
    features = vector.to_reduced_8q()   # 8 floats from frozen contract
    result = vqc.forward(features)       # returns expectation in [-1, 1]
    grad = vqc.gradient(features)        # parameter-shift gradients
"""

import math
import json
from typing import List, Tuple, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "VQC smoke test requires numpy. Install with: pip install numpy"
    )

N_QUBITS = 8
DIM = 2 ** N_QUBITS  # 256


# ── Gate primitives ─────────────────────────────────────────────────

def _ry(theta: float) -> np.ndarray:
    """Single-qubit RY rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a single-qubit gate to a specific qubit in the statevector."""
    n = N_QUBITS
    # Reshape state into tensor with 2 dimensions per qubit
    shape = [2] * n
    tensor = state.reshape(shape)

    # Apply gate by tensordot on the target qubit axis
    tensor = np.tensordot(gate, tensor, axes=([1], [qubit]))
    # tensordot moves the result axis to position 0; roll it back
    tensor = np.moveaxis(tensor, 0, qubit)

    return tensor.reshape(DIM)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply CNOT gate (control → target) to the statevector."""
    n = N_QUBITS
    shape = [2] * n
    new_state = state.copy()

    # CNOT: flip target when control = |1⟩
    # Iterate over all basis states where control qubit = 1
    for idx in range(DIM):
        # Check if control qubit is 1 in this basis state
        if (idx >> (n - 1 - control)) & 1:
            # Flip the target qubit
            flipped = idx ^ (1 << (n - 1 - target))
            new_state[flipped] = state[idx]
            new_state[idx] = state[flipped]

    # The above double-swaps, so we need to be more careful.
    # Proper approach: for each pair, swap once.
    new_state = state.copy()
    visited = set()
    for idx in range(DIM):
        if idx in visited:
            continue
        if (idx >> (n - 1 - control)) & 1:
            flipped = idx ^ (1 << (n - 1 - target))
            if flipped != idx and flipped not in visited:
                new_state[idx] = state[flipped]
                new_state[flipped] = state[idx]
                visited.add(idx)
                visited.add(flipped)

    return new_state


def _expectation_z(state: np.ndarray, qubit: int) -> float:
    """Compute ⟨Z⟩ expectation value on a specific qubit."""
    n = N_QUBITS
    exp = 0.0
    for idx in range(DIM):
        prob = abs(state[idx]) ** 2
        # Z eigenvalue: +1 if qubit is |0⟩, -1 if qubit is |1⟩
        bit = (idx >> (n - 1 - qubit)) & 1
        exp += prob * (1 - 2 * bit)
    return float(exp)


# ── VQC Class ───────────────────────────────────────────────────────

class VQCSmokeTest:
    """
    Minimal 8-qubit variational quantum circuit for Phase 1A.

    Circuit structure:
        |0⟩^8 → [RY encoding] → [CNOT chain] → [RY variational] → measure ⟨Z₀⟩

    Parameters:
        8 trainable angles (one per qubit in the variational layer)
    """

    def __init__(self, params: Optional[List[float]] = None, seed: int = 42):
        """
        Initialize VQC with random or specified parameters.

        Args:
            params: 8 trainable parameters. If None, initialized randomly.
            seed: Random seed for parameter initialization.
        """
        if params is not None:
            assert len(params) == N_QUBITS, f"Need {N_QUBITS} params, got {len(params)}"
            self.params = np.array(params, dtype=float)
        else:
            rng = np.random.RandomState(seed)
            self.params = rng.uniform(-math.pi, math.pi, N_QUBITS)

    def _run_circuit(self, features: List[float], params: np.ndarray) -> np.ndarray:
        """
        Execute the full circuit and return the final statevector.

        Args:
            features: 8 input features from to_reduced_8q()
            params: 8 variational parameters

        Returns:
            Complex statevector of length 256
        """
        assert len(features) == N_QUBITS, f"Need {N_QUBITS} features, got {len(features)}"

        # Initialize |00...0⟩
        state = np.zeros(DIM, dtype=complex)
        state[0] = 1.0

        # Layer 1: Angle encoding — RY(feature_i × π)
        for i in range(N_QUBITS):
            angle = features[i] * math.pi
            state = _apply_single(state, _ry(angle), i)

        # Layer 2: Entanglement — linear CNOT chain
        for i in range(N_QUBITS - 1):
            state = _apply_cnot(state, i, i + 1)

        # Layer 3: Variational — RY(θ_i)
        for i in range(N_QUBITS):
            state = _apply_single(state, _ry(params[i]), i)

        return state

    def forward(self, features: List[float]) -> float:
        """
        Run the circuit and return ⟨Z₀⟩ expectation value.

        Args:
            features: 8 floats from QuantumInputVector.to_reduced_8q()

        Returns:
            float: Expectation value in [-1, 1]
        """
        state = self._run_circuit(features, self.params)
        return _expectation_z(state, qubit=0)

    def forward_all_qubits(self, features: List[float]) -> List[float]:
        """
        Run the circuit and return ⟨Z⟩ on all 8 qubits.

        Args:
            features: 8 floats from to_reduced_8q()

        Returns:
            list[float]: 8 expectation values, each in [-1, 1]
        """
        state = self._run_circuit(features, self.params)
        return [_expectation_z(state, q) for q in range(N_QUBITS)]

    def gradient(self, features: List[float], shift: float = math.pi / 2) -> np.ndarray:
        """
        Compute parameter gradients using the parameter-shift rule.

        ∂⟨Z⟩/∂θᵢ = [⟨Z⟩(θᵢ+s) − ⟨Z⟩(θᵢ−s)] / (2 sin(s))

        Args:
            features: 8 input features
            shift: Shift amount (default π/2 for exact gradients)

        Returns:
            np.ndarray: 8 gradient values
        """
        grad = np.zeros(N_QUBITS)
        for i in range(N_QUBITS):
            # Forward shift
            params_plus = self.params.copy()
            params_plus[i] += shift
            state_plus = self._run_circuit(features, params_plus)
            exp_plus = _expectation_z(state_plus, qubit=0)

            # Backward shift
            params_minus = self.params.copy()
            params_minus[i] -= shift
            state_minus = self._run_circuit(features, params_minus)
            exp_minus = _expectation_z(state_minus, qubit=0)

            grad[i] = (exp_plus - exp_minus) / (2 * math.sin(shift))

        return grad

    def state_fidelity(self, features_a: List[float], features_b: List[float]) -> float:
        """
        Compute fidelity between two encoded states: |⟨ψ_a|ψ_b⟩|²

        This measures how distinguishable two market states are in the quantum encoding.

        Args:
            features_a, features_b: Two 8-feature vectors

        Returns:
            float: Fidelity in [0, 1]. Lower = more distinguishable.
        """
        state_a = self._run_circuit(features_a, self.params)
        state_b = self._run_circuit(features_b, self.params)
        overlap = abs(np.vdot(state_a, state_b)) ** 2
        return float(overlap)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit configuration."""
        return {
            "n_qubits": N_QUBITS,
            "n_params": N_QUBITS,
            "params": self.params.tolist(),
            "encoding": "RY(feature × π)",
            "entanglement": "linear_cnot",
            "ansatz": "single_RY_layer",
            "measurement": "Z_qubit0",
            "version": "1A-smoke",
        }
