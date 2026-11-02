"""
Trainable 8-Qubit VQC — Phase 1B
==================================
Upgraded variational quantum circuit with full parameter trainability.

Key improvements over Phase 1A smoke test:
  1. Circular entanglement (qubit 7 → qubit 0) so all parameters affect measurement
  2. Two variational layers with entanglement between them
  3. Multi-qubit measurement: weighted sum of ⟨Z⟩ on all qubits → single output
  4. Proper Adam-like optimizer with momentum
  5. Batch gradient computation for efficient training

Architecture (8 qubits, 24 trainable parameters):
  |0⟩^8 → [RY encoding] → [CNOT ring + RY layer 1] → [CNOT ring + RY layer 2] → measure

Parameter count: 8 (layer 1) + 8 (layer 2) + 8 (output weights) = 24

Contract compliance:
  - Still consumes exactly 8 floats from to_reduced_8q()
  - Encoding unchanged: RY(feature × π)
  - No changes to QuantumInputVector v1.0
"""

import math
import json
from typing import List, Tuple, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    raise ImportError("Phase 1B VQC requires numpy. Install with: pip install numpy")

N_QUBITS = 8
DIM = 2 ** N_QUBITS  # 256
N_LAYERS = 2
N_VARIATIONAL_PARAMS = N_QUBITS * N_LAYERS  # 16 rotation params
N_OUTPUT_WEIGHTS = N_QUBITS                  # 8 measurement weights
N_TOTAL_PARAMS = N_VARIATIONAL_PARAMS + N_OUTPUT_WEIGHTS + 1  # +1 bias = 25


# ── Gate primitives (optimized) ──────────────────────────────────────

def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_single(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    tensor = state.reshape([2] * N_QUBITS)
    tensor = np.tensordot(gate, tensor, axes=([1], [qubit]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(DIM)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    n = N_QUBITS
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
    n = N_QUBITS
    exp = 0.0
    for idx in range(DIM):
        prob = abs(state[idx]) ** 2
        bit = (idx >> (n - 1 - qubit)) & 1
        exp += prob * (1 - 2 * bit)
    return float(exp)


def _cnot_ring(state: np.ndarray) -> np.ndarray:
    """Apply circular CNOT entanglement: 0→1, 1→2, ..., 6→7, 7→0"""
    for i in range(N_QUBITS - 1):
        state = _apply_cnot(state, i, i + 1)
    state = _apply_cnot(state, N_QUBITS - 1, 0)  # Circular: 7→0
    return state


# ── VQC Phase 1B ────────────────────────────────────────────────────

class TrainableVQC:
    """
    Phase 1B trainable variational quantum circuit.

    Architecture:
        |0⟩^8 → [RY encoding] → [CNOT ring + RY_L1] → [CNOT ring + RY_L2] → weighted ⟨Z⟩

    Output: σ(Σ wᵢ·⟨Zᵢ⟩ + b) where σ is sigmoid, producing p ∈ (0, 1)

    Total parameters: 25
      - 8 × 2 = 16 variational RY angles
      - 8 output weights
      - 1 output bias
    """

    def __init__(self, params: Optional[np.ndarray] = None, seed: int = 42):
        if params is not None:
            assert len(params) == N_TOTAL_PARAMS
            self.params = np.array(params, dtype=float)
        else:
            rng = np.random.RandomState(seed)
            self.params = np.zeros(N_TOTAL_PARAMS)
            # Variational params: small random init
            self.params[:N_VARIATIONAL_PARAMS] = rng.uniform(
                -0.5, 0.5, N_VARIATIONAL_PARAMS
            )
            # Output weights: uniform small
            self.params[N_VARIATIONAL_PARAMS:N_VARIATIONAL_PARAMS + N_OUTPUT_WEIGHTS] = (
                rng.uniform(-0.3, 0.3, N_OUTPUT_WEIGHTS)
            )
            # Bias: zero
            self.params[-1] = 0.0

    @property
    def variational_params(self) -> np.ndarray:
        return self.params[:N_VARIATIONAL_PARAMS]

    @property
    def output_weights(self) -> np.ndarray:
        return self.params[N_VARIATIONAL_PARAMS:N_VARIATIONAL_PARAMS + N_OUTPUT_WEIGHTS]

    @property
    def output_bias(self) -> float:
        return self.params[-1]

    def _run_circuit(self, features: List[float], var_params: np.ndarray) -> np.ndarray:
        """Execute circuit, return statevector."""
        assert len(features) == N_QUBITS

        state = np.zeros(DIM, dtype=complex)
        state[0] = 1.0

        # Encoding layer: RY(feature_i × π)
        for i in range(N_QUBITS):
            state = _apply_single(state, _ry(features[i] * math.pi), i)

        # Variational layers
        for layer in range(N_LAYERS):
            # Entanglement: circular CNOT ring
            state = _cnot_ring(state)
            # Variational rotations
            offset = layer * N_QUBITS
            for i in range(N_QUBITS):
                state = _apply_single(state, _ry(var_params[offset + i]), i)

        return state

    def _get_expectations(self, features: List[float], var_params: np.ndarray) -> np.ndarray:
        """Get ⟨Z⟩ expectations on all qubits."""
        state = self._run_circuit(features, var_params)
        return np.array([_expectation_z(state, q) for q in range(N_QUBITS)])

    def forward_raw(self, features: List[float]) -> float:
        """Compute raw logit (before sigmoid)."""
        exps = self._get_expectations(features, self.variational_params)
        logit = float(np.dot(self.output_weights, exps) + self.output_bias)
        return logit

    def forward(self, features: List[float]) -> float:
        """Compute probability output via sigmoid."""
        logit = self.forward_raw(features)
        return _sigmoid(logit)

    def predict(self, features: List[float]) -> int:
        """Binary prediction: 1 if p > 0.5, else 0."""
        return 1 if self.forward(features) > 0.5 else 0

    def gradient(self, features: List[float], y_true: float,
                 loss_fn: str = "bce") -> np.ndarray:
        """
        Compute gradient of loss w.r.t. all 25 parameters.

        Uses parameter-shift rule for variational params,
        and analytic gradients for output weights/bias.
        """
        grad = np.zeros(N_TOTAL_PARAMS)
        shift = math.pi / 2

        # Current expectations for output weight gradients
        exps = self._get_expectations(features, self.variational_params)
        p = _sigmoid(float(np.dot(self.output_weights, exps) + self.output_bias))

        # dL/dp for BCE loss
        eps = 1e-7
        p_clip = np.clip(p, eps, 1 - eps)
        if loss_fn == "bce":
            dl_dp = -y_true / p_clip + (1 - y_true) / (1 - p_clip)
        else:
            dl_dp = 2 * (p - y_true)  # MSE

        # dp/dlogit = p(1-p)  (sigmoid derivative)
        dp_dlogit = p_clip * (1 - p_clip)
        dl_dlogit = dl_dp * dp_dlogit

        # --- Gradient for output weights: dL/dw_i = dL/dlogit × ⟨Z_i⟩ ---
        for i in range(N_OUTPUT_WEIGHTS):
            grad[N_VARIATIONAL_PARAMS + i] = dl_dlogit * exps[i]

        # --- Gradient for bias: dL/db = dL/dlogit ---
        grad[-1] = dl_dlogit

        # --- Gradient for variational params: parameter-shift rule ---
        for i in range(N_VARIATIONAL_PARAMS):
            # Shift parameter up
            params_plus = self.variational_params.copy()
            params_plus[i] += shift
            exps_plus = self._get_expectations(features, params_plus)
            logit_plus = float(np.dot(self.output_weights, exps_plus) + self.output_bias)
            p_plus = _sigmoid(logit_plus)

            # Shift parameter down
            params_minus = self.variational_params.copy()
            params_minus[i] -= shift
            exps_minus = self._get_expectations(features, params_minus)
            logit_minus = float(np.dot(self.output_weights, exps_minus) + self.output_bias)
            p_minus = _sigmoid(logit_minus)

            # Loss at shifted points
            if loss_fn == "bce":
                loss_plus = _bce_loss(y_true, p_plus)
                loss_minus = _bce_loss(y_true, p_minus)
            else:
                loss_plus = (p_plus - y_true) ** 2
                loss_minus = (p_minus - y_true) ** 2

            grad[i] = (loss_plus - loss_minus) / (2 * math.sin(shift))

        return grad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "n_params": N_TOTAL_PARAMS,
            "params": self.params.tolist(),
            "encoding": "RY(feature × π)",
            "entanglement": "circular_cnot_ring",
            "ansatz": f"{N_LAYERS}-layer RY",
            "measurement": "weighted_all_Z + sigmoid",
            "version": "1B-trainable",
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainableVQC":
        return cls(params=np.array(d["params"]))


# ── Utilities ───────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _bce_loss(y_true: float, y_pred: float) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


# ── Optimizer ───────────────────────────────────────────────────────

class AdamOptimizer:
    """Simple Adam optimizer for VQC parameter updates."""

    def __init__(self, n_params: int, lr: float = 0.05,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
