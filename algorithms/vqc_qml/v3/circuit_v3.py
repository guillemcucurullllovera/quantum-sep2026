"""
Trainable 8-Qubit VQC — Phase 2 (v3)
=======================================
Three key improvements over Phase 1B circuit:

1. DATA RE-UPLOADING: Features are re-encoded between variational layers,
   not just at the start. This dramatically improves expressibility by letting
   the circuit learn how features interact with the quantum state at different
   depths. Published research shows this is the single biggest improvement
   for quantum classifiers on small datasets.

2. LEARNED ENCODING SCALES: Each feature gets a trainable scaling parameter
   instead of fixed RY(feature × π). The circuit learns how much to "open"
   each qubit's gate based on the feature's importance. Features with more
   predictive power get larger scales automatically.

3. RICHER ENTANGLEMENT: In addition to the circular CNOT ring, we add
   cross-connections between non-adjacent qubits that encode potentially
   correlated features (e.g., carry↔risk_sentiment, momentum↔vol).

Architecture (8 qubits, 49 trainable parameters):
  |0⟩^8 → [Scaled RY encoding] → [CNOT ring + cross + RY_L1]
        → [Scaled RY re-upload]  → [CNOT ring + cross + RY_L2]
        → weighted ⟨Z⟩ readout

Parameter count:
  - 8 encoding scales (learned per-feature)
  - 8 × 2 = 16 variational RY angles (same as v2)
  - 8 output weights + 1 bias (same as v2)
  Total: 8 + 16 + 8 + 1 = 33

Contract compliance:
  - Still consumes exactly 8 floats from to_reduced_8q()
  - Backward-compatible output: sigmoid probability in (0, 1)
  - No changes to QuantumInputVector v1.0
"""

import math
import json
from typing import List, Tuple, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    raise ImportError("VQC v3 requires numpy. Install with: pip install numpy")

N_QUBITS = 8
DIM = 2 ** N_QUBITS  # 256
N_LAYERS = 2

# Parameter layout:
# [0:8]     = encoding scales (one per feature)
# [8:24]    = variational RY angles (8 per layer × 2 layers)
# [24:32]   = output weights
# [32]      = output bias
N_ENCODING_SCALES = N_QUBITS                    # 8
N_VARIATIONAL_PARAMS = N_QUBITS * N_LAYERS      # 16
N_OUTPUT_WEIGHTS = N_QUBITS                      # 8
N_TOTAL_PARAMS = N_ENCODING_SCALES + N_VARIATIONAL_PARAMS + N_OUTPUT_WEIGHTS + 1  # 33

# Cross-entanglement pairs: connect features that interact in financial markets
# (momentum↔vol, carry↔risk, sentiment↔cot, regime↔event)
CROSS_ENTANGLE_PAIRS = [
    (0, 1),  # momentum ↔ vol
    (3, 7),  # carry ↔ risk_sentiment
    (5, 6),  # sentiment ↔ cot
    (2, 4),  # regime ↔ event_risk
]


# ── Gate primitives (same as v2, proven correct) ────────────────────

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
    state = _apply_cnot(state, N_QUBITS - 1, 0)
    return state


def _cross_entangle(state: np.ndarray) -> np.ndarray:
    """Apply cross-entanglement between correlated feature pairs."""
    for control, target in CROSS_ENTANGLE_PAIRS:
        state = _apply_cnot(state, control, target)
    return state


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


# ── VQC v3 ──────────────────────────────────────────────────────────

class TrainableVQC_v3:
    """
    Phase 2 variational quantum circuit with data re-uploading,
    learned encoding scales, and cross-entanglement.

    Architecture:
        |0⟩^8 → [Scaled encoding] → [Ring + Cross + RY_L1]
              → [Scaled re-upload]  → [Ring + Cross + RY_L2]
              → weighted ⟨Z⟩ + bias → sigmoid

    Total parameters: 33
      - 8 encoding scales
      - 8 × 2 = 16 variational RY angles
      - 8 output weights + 1 bias
    """

    def __init__(self, params: Optional[np.ndarray] = None, seed: int = 42):
        if params is not None:
            assert len(params) == N_TOTAL_PARAMS, \
                f"Expected {N_TOTAL_PARAMS} params, got {len(params)}"
            self.params = np.array(params, dtype=float)
        else:
            rng = np.random.RandomState(seed)
            self.params = np.zeros(N_TOTAL_PARAMS)

            # Encoding scales: init near π (preserving v2 behavior)
            self.params[:N_ENCODING_SCALES] = (
                math.pi + rng.uniform(-0.3, 0.3, N_ENCODING_SCALES)
            )

            # Variational params: small random init
            self.params[N_ENCODING_SCALES:N_ENCODING_SCALES + N_VARIATIONAL_PARAMS] = (
                rng.uniform(-0.5, 0.5, N_VARIATIONAL_PARAMS)
            )

            # Output weights: uniform small
            w_start = N_ENCODING_SCALES + N_VARIATIONAL_PARAMS
            self.params[w_start:w_start + N_OUTPUT_WEIGHTS] = (
                rng.uniform(-0.3, 0.3, N_OUTPUT_WEIGHTS)
            )

            # Bias: zero
            self.params[-1] = 0.0

    @property
    def encoding_scales(self) -> np.ndarray:
        return self.params[:N_ENCODING_SCALES]

    @property
    def variational_params(self) -> np.ndarray:
        return self.params[N_ENCODING_SCALES:N_ENCODING_SCALES + N_VARIATIONAL_PARAMS]

    @property
    def output_weights(self) -> np.ndarray:
        w_start = N_ENCODING_SCALES + N_VARIATIONAL_PARAMS
        return self.params[w_start:w_start + N_OUTPUT_WEIGHTS]

    @property
    def output_bias(self) -> float:
        return self.params[-1]

    def _encode_features(self, state: np.ndarray, features: List[float],
                         scales: np.ndarray) -> np.ndarray:
        """Apply scaled feature encoding: RY(feature_i × scale_i)"""
        for i in range(N_QUBITS):
            angle = features[i] * scales[i]
            state = _apply_single(state, _ry(angle), i)
        return state

    def _run_circuit(self, features: List[float], var_params: np.ndarray,
                     scales: np.ndarray) -> np.ndarray:
        """Execute circuit with data re-uploading, return statevector."""
        assert len(features) == N_QUBITS

        state = np.zeros(DIM, dtype=complex)
        state[0] = 1.0

        # Initial encoding: RY(feature_i × scale_i)
        state = self._encode_features(state, features, scales)

        # Variational layers with re-uploading
        for layer in range(N_LAYERS):
            # Entanglement: circular CNOT ring + cross connections
            state = _cnot_ring(state)
            state = _cross_entangle(state)

            # Variational rotations
            offset = layer * N_QUBITS
            for i in range(N_QUBITS):
                state = _apply_single(state, _ry(var_params[offset + i]), i)

            # DATA RE-UPLOADING: re-encode features after each layer
            # (except the last layer — re-uploading before measurement
            #  adds no expressibility)
            if layer < N_LAYERS - 1:
                state = self._encode_features(state, features, scales)

        return state

    def _get_expectations(self, features: List[float], var_params: np.ndarray,
                          scales: np.ndarray) -> np.ndarray:
        """Get ⟨Z⟩ expectations on all qubits."""
        state = self._run_circuit(features, var_params, scales)
        return np.array([_expectation_z(state, q) for q in range(N_QUBITS)])

    def forward_raw(self, features: List[float]) -> float:
        """Compute raw logit (before sigmoid)."""
        exps = self._get_expectations(
            features, self.variational_params, self.encoding_scales
        )
        logit = float(np.dot(self.output_weights, exps) + self.output_bias)
        return logit

    def forward(self, features: List[float]) -> float:
        """Compute probability output via sigmoid."""
        return _sigmoid(self.forward_raw(features))

    def predict(self, features: List[float]) -> int:
        """Binary prediction: 1 if p > 0.5, else 0."""
        return 1 if self.forward(features) > 0.5 else 0

    def gradient(self, features: List[float], y_true: float,
                 loss_fn: str = "bce") -> np.ndarray:
        """
        Compute gradient of loss w.r.t. all 33 parameters.

        Uses parameter-shift rule for encoding scales + variational params,
        and analytic gradients for output weights/bias.
        """
        grad = np.zeros(N_TOTAL_PARAMS)
        shift = math.pi / 2

        # Current forward pass
        exps = self._get_expectations(
            features, self.variational_params, self.encoding_scales
        )
        logit = float(np.dot(self.output_weights, exps) + self.output_bias)
        p = _sigmoid(logit)

        # dL/dp for BCE loss
        eps = 1e-7
        p_clip = np.clip(p, eps, 1 - eps)
        if loss_fn == "bce":
            dl_dp = -y_true / p_clip + (1 - y_true) / (1 - p_clip)
        else:
            dl_dp = 2 * (p - y_true)

        dp_dlogit = p_clip * (1 - p_clip)
        dl_dlogit = dl_dp * dp_dlogit

        # Gradient for output weights: dL/dw_i = dL/dlogit × ⟨Z_i⟩
        w_start = N_ENCODING_SCALES + N_VARIATIONAL_PARAMS
        for i in range(N_OUTPUT_WEIGHTS):
            grad[w_start + i] = dl_dlogit * exps[i]

        # Gradient for bias
        grad[-1] = dl_dlogit

        # Gradient for encoding scales: parameter-shift rule
        for i in range(N_ENCODING_SCALES):
            scales_plus = self.encoding_scales.copy()
            scales_plus[i] += shift
            exps_plus = self._get_expectations(features, self.variational_params, scales_plus)
            logit_plus = float(np.dot(self.output_weights, exps_plus) + self.output_bias)
            p_plus = _sigmoid(logit_plus)

            scales_minus = self.encoding_scales.copy()
            scales_minus[i] -= shift
            exps_minus = self._get_expectations(features, self.variational_params, scales_minus)
            logit_minus = float(np.dot(self.output_weights, exps_minus) + self.output_bias)
            p_minus = _sigmoid(logit_minus)

            if loss_fn == "bce":
                loss_plus = _bce_loss(y_true, p_plus)
                loss_minus = _bce_loss(y_true, p_minus)
            else:
                loss_plus = (p_plus - y_true) ** 2
                loss_minus = (p_minus - y_true) ** 2

            grad[i] = (loss_plus - loss_minus) / (2 * math.sin(shift))

        # Gradient for variational params: parameter-shift rule
        for i in range(N_VARIATIONAL_PARAMS):
            params_plus = self.variational_params.copy()
            params_plus[i] += shift
            exps_plus = self._get_expectations(features, params_plus, self.encoding_scales)
            logit_plus = float(np.dot(self.output_weights, exps_plus) + self.output_bias)
            p_plus = _sigmoid(logit_plus)

            params_minus = self.variational_params.copy()
            params_minus[i] -= shift
            exps_minus = self._get_expectations(features, params_minus, self.encoding_scales)
            logit_minus = float(np.dot(self.output_weights, exps_minus) + self.output_bias)
            p_minus = _sigmoid(logit_minus)

            if loss_fn == "bce":
                loss_plus = _bce_loss(y_true, p_plus)
                loss_minus = _bce_loss(y_true, p_minus)
            else:
                loss_plus = (p_plus - y_true) ** 2
                loss_minus = (p_minus - y_true) ** 2

            grad[N_ENCODING_SCALES + i] = (loss_plus - loss_minus) / (2 * math.sin(shift))

        return grad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "n_params": N_TOTAL_PARAMS,
            "params": self.params.tolist(),
            "encoding": "learned_scale × feature",
            "entanglement": "circular_cnot_ring + cross_pairs",
            "cross_pairs": CROSS_ENTANGLE_PAIRS,
            "ansatz": f"{N_LAYERS}-layer RY with data re-uploading",
            "measurement": "weighted_all_Z + sigmoid",
            "version": "v3-reupload",
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainableVQC_v3":
        return cls(params=np.array(d["params"]))


# ── Optimizer (same as v2) ──────────────────────────────────────────

class AdamOptimizer:
    """Adam optimizer for VQC parameter updates."""

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
