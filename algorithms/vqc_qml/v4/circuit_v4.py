"""
Trainable 14-Qubit VQC — Phase 3B (v4)
=======================================
Scaled-up circuit for 1:1 feature-to-qubit mapping.

Key upgrades over v3 (8-qubit):

1. 14 QUBITS: One qubit per QIV feature. Eliminates the lossy compression
   from 14→8 features (to_reduced_8q). All information from the quantum input
   vector is preserved. State vector: 2^14 = 16,384 complex amplitudes.

2. DOMAIN-AWARE CROSS-ENTANGLEMENT: Bridges connect features with known
   financial correlations (momentum↔accel, vol↔regime, sentiment↔momentum,
   macro↔risk). Designed from QIV feature correlation analysis.

3. DATA RE-UPLOADING preserved from v3 — still the highest-impact technique
   for small-dataset quantum classifiers.

4. CONFIGURABLE ARCHITECTURE: n_qubits and n_layers are constructor params,
   not module-level globals. Supports future scaling to 16-20 qubits.

Architecture (14 qubits, 2 layers, 57 trainable parameters):
  |0⟩^14 → [Scaled RY encoding] → [CNOT ring + cross + RY_L1]
         → [Scaled RY re-upload]  → [CNOT ring + cross + RY_L2]
         → weighted ⟨Z⟩ readout + bias → sigmoid

Parameter count:
  - 14 encoding scales (learned per-feature)
  - 14 × 2 = 28 variational RY angles
  - 14 output weights + 1 bias
  Total: 14 + 28 + 14 + 1 = 57

QIV v1.0 Feature-to-Qubit Mapping:
  q0:  momentum_composite     [-1, 1]
  q1:  momentum_acceleration  [-1, 1]
  q2:  realized_vol_20d       [0, 1]
  q3:  vol_regime             {0, 0.5, 1.0}
  q4:  atr_normalized         [0, 1]
  q5:  carry_differential     [-1, 1]
  q6:  yield_curve_slope      [-1, 1]
  q7:  event_risk_score       [0, 1]
  q8:  net_sentiment          [-1, 1]
  q9:  sentiment_momentum     [-1, 1]
  q10: cot_positioning        [-1, 1]
  q11: cot_extreme            [-1, 1]
  q12: dxy_momentum           [-1, 1]
  q13: risk_sentiment         [0, 1]

Benchmark target: >72% directional accuracy (v3 baseline: 69.7% live / 52.7% val avg)
JPMorgan reference: 20-qubit circuits on Quantinuum H1-1, depths up to 159 2Q gates.
"""

import math
import json
from typing import List, Tuple, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    raise ImportError("VQC v4 requires numpy. Install with: pip install numpy")


# ── Cross-entanglement pairs: domain-aware financial correlations ──
# Designed from QIV feature index mapping and known market structure.
#
# Group 1 — Momentum cluster:
#   q0 (momentum_composite) ↔ q1 (momentum_acceleration)  — acceleration is derivative of momentum
#
# Group 2 — Volatility cluster:
#   q2 (realized_vol_20d) ↔ q3 (vol_regime)               — regime is derived from vol
#   q4 (atr_normalized) ↔ q2 (realized_vol_20d)           — ATR and vol are correlated
#
# Group 3 — Sentiment cluster:
#   q8 (net_sentiment) ↔ q9 (sentiment_momentum)          — momentum is derivative of sentiment
#   q10 (cot_positioning) ↔ q11 (cot_extreme)             — extreme is derived from positioning
#
# Group 4 — Macro-risk cluster:
#   q5 (carry_differential) ↔ q6 (yield_curve_slope)      — both rate-related
#   q7 (event_risk_score) ↔ q13 (risk_sentiment)          — risk indicators
#   q12 (dxy_momentum) ↔ q5 (carry_differential)          — DXY relates to rate differentials
#
# Group 5 — Cross-cluster bridges (non-obvious but empirically useful):
#   q0 (momentum) ↔ q13 (risk_sentiment)                  — risk-on/off drives momentum
#   q8 (sentiment) ↔ q2 (vol)                             — sentiment spikes in high vol

DEFAULT_CROSS_PAIRS_14Q = [
    # Momentum cluster
    (0, 1),   # momentum ↔ acceleration
    # Volatility cluster
    (2, 3),   # vol ↔ regime
    (4, 2),   # ATR ↔ vol
    # Sentiment cluster
    (8, 9),   # sentiment ↔ sentiment_momentum
    (10, 11), # COT ↔ COT_extreme
    # Macro-risk cluster
    (5, 6),   # carry ↔ yield_slope
    (7, 13),  # event_risk ↔ risk_sentiment
    (12, 5),  # DXY ↔ carry
    # Cross-cluster bridges
    (0, 13),  # momentum ↔ risk_sentiment
    (8, 2),   # sentiment ↔ vol
]


# ── Gate primitives (parameterized by n_qubits) ────────────────────

def _ry(theta: float) -> np.ndarray:
    """Single-qubit RY rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_single(state: np.ndarray, gate: np.ndarray,
                   qubit: int, n_qubits: int) -> np.ndarray:
    """Apply single-qubit gate via tensor contraction."""
    dim = 2 ** n_qubits
    tensor = state.reshape([2] * n_qubits)
    tensor = np.tensordot(gate, tensor, axes=([1], [qubit]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(dim)


def _apply_cnot(state: np.ndarray, control: int, target: int,
                n_qubits: int) -> np.ndarray:
    """Apply CNOT gate via vectorized index manipulation.
    Swaps amplitudes where control=1, flipping target bit."""
    dim = 2 ** n_qubits
    ctrl_mask = 1 << (n_qubits - 1 - control)
    tgt_mask = 1 << (n_qubits - 1 - target)

    indices = np.arange(dim)
    # Find indices where control=1 AND target=0
    mask = ((indices & ctrl_mask) != 0) & ((indices & tgt_mask) == 0)
    src = indices[mask]
    dst = src ^ tgt_mask  # flip target bit

    new_state = state.copy()
    new_state[src] = state[dst]
    new_state[dst] = state[src]
    return new_state


def _expectation_z(state: np.ndarray, qubit: int, n_qubits: int) -> float:
    """Compute ⟨Z⟩ expectation for a single qubit (vectorized)."""
    dim = 2 ** n_qubits
    probs = np.abs(state) ** 2
    indices = np.arange(dim)
    bits = (indices >> (n_qubits - 1 - qubit)) & 1
    return float(np.sum(probs * (1 - 2 * bits)))


def _cnot_ring(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply circular CNOT entanglement: 0→1, 1→2, ..., (n-1)→0."""
    for i in range(n_qubits - 1):
        state = _apply_cnot(state, i, i + 1, n_qubits)
    state = _apply_cnot(state, n_qubits - 1, 0, n_qubits)
    return state


def _cross_entangle(state: np.ndarray, pairs: List[Tuple[int, int]],
                    n_qubits: int) -> np.ndarray:
    """Apply cross-entanglement between specified qubit pairs."""
    for control, target in pairs:
        state = _apply_cnot(state, control, target, n_qubits)
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


# ── VQC v4 ─────────────────────────────────────────────────────────

class TrainableVQC_v4:
    """
    Phase 3B variational quantum circuit — 14-qubit, 1:1 feature mapping.

    Configurable architecture supporting 8-20 qubits.
    Default: 14 qubits (matching QIV v1.0 full feature set).

    Architecture:
        |0⟩^n → [Scaled encoding] → [Ring + Cross + RY_L1]
              → [Scaled re-upload]  → [Ring + Cross + RY_L2]
              → weighted ⟨Z⟩ + bias → sigmoid
    """

    def __init__(self, n_qubits: int = 14, n_layers: int = 2,
                 cross_pairs: Optional[List[Tuple[int, int]]] = None,
                 params: Optional[np.ndarray] = None, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits
        self.cross_pairs = cross_pairs if cross_pairs is not None else (
            DEFAULT_CROSS_PAIRS_14Q if n_qubits == 14 else
            self._auto_cross_pairs(n_qubits)
        )

        # Parameter layout
        self.n_encoding_scales = n_qubits
        self.n_variational_params = n_qubits * n_layers
        self.n_output_weights = n_qubits
        self.n_total_params = (self.n_encoding_scales +
                               self.n_variational_params +
                               self.n_output_weights + 1)

        if params is not None:
            assert len(params) == self.n_total_params, \
                f"Expected {self.n_total_params} params, got {len(params)}"
            self.params = np.array(params, dtype=float)
        else:
            rng = np.random.RandomState(seed)
            self.params = np.zeros(self.n_total_params)

            # Encoding scales: init near π (preserving v3 behavior)
            self.params[:self.n_encoding_scales] = (
                math.pi + rng.uniform(-0.3, 0.3, self.n_encoding_scales)
            )

            # Variational params: small random init (Xavier-like scaling)
            var_scale = 1.0 / math.sqrt(n_qubits)
            self.params[self.n_encoding_scales:
                        self.n_encoding_scales + self.n_variational_params] = (
                rng.uniform(-var_scale, var_scale, self.n_variational_params)
            )

            # Output weights: uniform small
            w_start = self.n_encoding_scales + self.n_variational_params
            self.params[w_start:w_start + self.n_output_weights] = (
                rng.uniform(-0.3, 0.3, self.n_output_weights)
            )

            # Bias: zero
            self.params[-1] = 0.0

    @staticmethod
    def _auto_cross_pairs(n_qubits: int) -> List[Tuple[int, int]]:
        """Generate cross-entanglement pairs for arbitrary qubit counts.
        Connects every other qubit for non-local correlations."""
        pairs = []
        for i in range(0, n_qubits - 2, 2):
            pairs.append((i, i + 2))
        if n_qubits > 4:
            pairs.append((0, n_qubits - 1))
        return pairs

    @property
    def encoding_scales(self) -> np.ndarray:
        return self.params[:self.n_encoding_scales]

    @property
    def variational_params(self) -> np.ndarray:
        return self.params[self.n_encoding_scales:
                           self.n_encoding_scales + self.n_variational_params]

    @property
    def output_weights(self) -> np.ndarray:
        w_start = self.n_encoding_scales + self.n_variational_params
        return self.params[w_start:w_start + self.n_output_weights]

    @property
    def output_bias(self) -> float:
        return self.params[-1]

    def _encode_features(self, state: np.ndarray, features: List[float],
                         scales: np.ndarray) -> np.ndarray:
        """Apply scaled feature encoding: RY(feature_i × scale_i) on each qubit."""
        for i in range(self.n_qubits):
            angle = features[i] * scales[i]
            state = _apply_single(state, _ry(angle), i, self.n_qubits)
        return state

    def _run_circuit(self, features: List[float], var_params: np.ndarray,
                     scales: np.ndarray) -> np.ndarray:
        """Execute circuit with data re-uploading, return statevector."""
        assert len(features) == self.n_qubits, \
            f"Expected {self.n_qubits} features, got {len(features)}"

        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        # Initial encoding
        state = self._encode_features(state, features, scales)

        # Variational layers with re-uploading
        for layer in range(self.n_layers):
            # Entanglement: circular CNOT ring + cross connections
            state = _cnot_ring(state, self.n_qubits)
            state = _cross_entangle(state, self.cross_pairs, self.n_qubits)

            # Variational rotations
            offset = layer * self.n_qubits
            for i in range(self.n_qubits):
                state = _apply_single(
                    state, _ry(var_params[offset + i]), i, self.n_qubits
                )

            # Data re-uploading between layers (not after last)
            if layer < self.n_layers - 1:
                state = self._encode_features(state, features, scales)

        return state

    def _get_expectations(self, features: List[float], var_params: np.ndarray,
                          scales: np.ndarray) -> np.ndarray:
        """Get ⟨Z⟩ expectations on all qubits."""
        state = self._run_circuit(features, var_params, scales)
        return np.array([
            _expectation_z(state, q, self.n_qubits)
            for q in range(self.n_qubits)
        ])

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
        Compute gradient of loss w.r.t. all parameters.

        Uses parameter-shift rule for encoding scales + variational params,
        and analytic gradients for output weights/bias.

        For 14 qubits this requires 2×(14+28) = 84 forward passes per sample.
        At ~10ms per forward pass, total ~840ms per gradient — acceptable for
        training but we use mini-batches to amortize.
        """
        grad = np.zeros(self.n_total_params)
        shift = math.pi / 2

        # Current forward pass
        exps = self._get_expectations(
            features, self.variational_params, self.encoding_scales
        )
        logit = float(np.dot(self.output_weights, exps) + self.output_bias)
        p = _sigmoid(logit)

        # dL/dp for loss
        eps = 1e-7
        p_clip = np.clip(p, eps, 1 - eps)
        if loss_fn == "bce":
            dl_dp = -y_true / p_clip + (1 - y_true) / (1 - p_clip)
        else:
            dl_dp = 2 * (p - y_true)

        dp_dlogit = p_clip * (1 - p_clip)
        dl_dlogit = dl_dp * dp_dlogit

        # Gradient for output weights: dL/dw_i = dL/dlogit × ⟨Z_i⟩
        w_start = self.n_encoding_scales + self.n_variational_params
        for i in range(self.n_output_weights):
            grad[w_start + i] = dl_dlogit * exps[i]

        # Gradient for bias
        grad[-1] = dl_dlogit

        # Gradient for encoding scales: parameter-shift rule
        for i in range(self.n_encoding_scales):
            scales_plus = self.encoding_scales.copy()
            scales_plus[i] += shift
            exps_plus = self._get_expectations(
                features, self.variational_params, scales_plus
            )
            logit_plus = float(
                np.dot(self.output_weights, exps_plus) + self.output_bias
            )
            p_plus = _sigmoid(logit_plus)

            scales_minus = self.encoding_scales.copy()
            scales_minus[i] -= shift
            exps_minus = self._get_expectations(
                features, self.variational_params, scales_minus
            )
            logit_minus = float(
                np.dot(self.output_weights, exps_minus) + self.output_bias
            )
            p_minus = _sigmoid(logit_minus)

            if loss_fn == "bce":
                loss_plus = _bce_loss(y_true, p_plus)
                loss_minus = _bce_loss(y_true, p_minus)
            else:
                loss_plus = (p_plus - y_true) ** 2
                loss_minus = (p_minus - y_true) ** 2

            grad[i] = (loss_plus - loss_minus) / (2 * math.sin(shift))

        # Gradient for variational params: parameter-shift rule
        for i in range(self.n_variational_params):
            params_plus = self.variational_params.copy()
            params_plus[i] += shift
            exps_plus = self._get_expectations(
                features, params_plus, self.encoding_scales
            )
            logit_plus = float(
                np.dot(self.output_weights, exps_plus) + self.output_bias
            )
            p_plus = _sigmoid(logit_plus)

            params_minus = self.variational_params.copy()
            params_minus[i] -= shift
            exps_minus = self._get_expectations(
                features, params_minus, self.encoding_scales
            )
            logit_minus = float(
                np.dot(self.output_weights, exps_minus) + self.output_bias
            )
            p_minus = _sigmoid(logit_minus)

            if loss_fn == "bce":
                loss_plus = _bce_loss(y_true, p_plus)
                loss_minus = _bce_loss(y_true, p_minus)
            else:
                loss_plus = (p_plus - y_true) ** 2
                loss_minus = (p_minus - y_true) ** 2

            grad[self.n_encoding_scales + i] = (
                (loss_plus - loss_minus) / (2 * math.sin(shift))
            )

        return grad

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_total_params,
            "params": self.params.tolist(),
            "encoding": "learned_scale × feature",
            "entanglement": "circular_cnot_ring + domain_aware_cross_pairs",
            "cross_pairs": self.cross_pairs,
            "ansatz": f"{self.n_layers}-layer RY with data re-uploading",
            "measurement": "weighted_all_Z + sigmoid",
            "feature_map": [
                "q0:momentum_composite", "q1:momentum_acceleration",
                "q2:realized_vol_20d", "q3:vol_regime",
                "q4:atr_normalized", "q5:carry_differential",
                "q6:yield_curve_slope", "q7:event_risk_score",
                "q8:net_sentiment", "q9:sentiment_momentum",
                "q10:cot_positioning", "q11:cot_extreme",
                "q12:dxy_momentum", "q13:risk_sentiment",
            ],
            "version": "v4-14q",
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainableVQC_v4":
        n_qubits = d.get("n_qubits", 14)
        n_layers = d.get("n_layers", 2)
        cross_pairs = d.get("cross_pairs", None)
        if cross_pairs is not None:
            cross_pairs = [tuple(p) for p in cross_pairs]
        return cls(
            n_qubits=n_qubits,
            n_layers=n_layers,
            cross_pairs=cross_pairs,
            params=np.array(d["params"]),
        )


# ── Optimizer (enhanced Adam with optional gradient clipping) ──────

class AdamOptimizer:
    """Adam optimizer with optional gradient clipping for VQC training."""

    def __init__(self, n_params: int, lr: float = 0.03,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, grad_clip: Optional[float] = 5.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # Optional gradient clipping (important for 14q — larger gradients)
        if self.grad_clip is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.grad_clip:
                grad = grad * (self.grad_clip / grad_norm)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Cosine warmup schedule (from v3 training) ─────────────────────

class CosineWarmupSchedule:
    """Learning rate schedule: linear warmup → cosine decay."""

    def __init__(self, lr_max: float = 0.03, lr_min: float = 0.003,
                 warmup_end: int = 10, decay_start: int = 30,
                 total_epochs: int = 60):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_end = warmup_end
        self.decay_start = decay_start
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_end:
            return self.lr_min + (self.lr_max - self.lr_min) * epoch / self.warmup_end
        elif epoch < self.decay_start:
            return self.lr_max
        else:
            progress = (epoch - self.decay_start) / max(1, self.total_epochs - self.decay_start)
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
