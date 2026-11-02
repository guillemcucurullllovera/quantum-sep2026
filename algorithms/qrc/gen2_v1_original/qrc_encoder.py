"""
QRC Angle Encoder — IsingXX Reservoir Input Encoding

Implements angle encoding for the QRC-10 reservoir. Input financial features
are encoded as RZ rotations on the first n1 qubits (input register), which
then propagate into the reservoir via the IsingXX Hamiltonian evolution.

Used by qrc_v1.py as the input preparation step.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def encode_inputs(
    x: np.ndarray,
    n1: int,
    scale: float = np.pi,
    clip: bool = True,
) -> np.ndarray:
    """
    Encode a 1-D feature vector x ∈ R^{n1} as rotation angles θ ∈ [-π, π].

    Maps each feature to a rotation angle: θ_i = scale * tanh(x_i / sigma)
    where sigma = 1.0 normalises the tanh to map [-1,1] inputs to [-scale, scale].
    tanh is used because:
      - It maps R → (-1,1), bounding the angle and preventing reservoir saturation.
      - Derivative is well-behaved — avoids gradient issues if we ever want gradients.
      - Preserves the sign (directional) of the input.

    Args:
        x: 1-D feature vector, length n1.
        n1: Number of input qubits in the reservoir.
        scale: Maximum angle magnitude (default π).
        clip: If True, clip output to [-π, π] (safety for numerical edge cases).

    Returns:
        angles: 1-D array of rotation angles, length n1, in radians.
    """
    if len(x) < n1:
        raise ValueError(
            f"Feature vector has length {len(x)}, but n1={n1} input qubits requested."
        )
    x_in = x[:n1].astype(float)
    angles = scale * np.tanh(x_in)
    if clip:
        angles = np.clip(angles, -np.pi, np.pi)
    return angles


def encode_sequence(
    X: np.ndarray,
    n1: int,
    scale: float = np.pi,
) -> np.ndarray:
    """
    Encode a 2-D feature matrix X ∈ R^{T × F} row-by-row.

    Args:
        X: Feature matrix, shape (T, F) where T = time steps, F ≥ n1 features.
        n1: Number of input qubits.
        scale: Angle scaling factor.

    Returns:
        angles: Array of shape (T, n1).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T × F), got shape {X.shape}")
    if X.shape[1] < n1:
        raise ValueError(
            f"X has {X.shape[1]} features, but n1={n1} qubits required."
        )
    return np.stack([encode_inputs(X[t], n1, scale) for t in range(len(X))])


def apply_rz_encoding(state: np.ndarray, angles: np.ndarray, n1: int) -> np.ndarray:
    """
    Apply RZ(θ_i) gates to the input register of the statevector.

    RZ(θ)|ψ⟩: applies a phase exp(-iθ/2) to |1⟩ component of qubit i,
    keeping the overall state normalised.

    The n1 input qubits are assumed to be the LAST n1 qubits in the tensor
    product (qubit ordering: reservoir qubits 0..n2-1, input qubits n2..n-1).

    Args:
        state: Complex statevector, shape (2**n,).
        angles: RZ angles, shape (n1,).
        n1: Number of input qubits.

    Returns:
        state: Encoded statevector (in-place modified copy).
    """
    n = int(round(np.log2(len(state))))
    n2 = n - n1
    state = state.copy()

    for k, theta in enumerate(angles):
        qubit_idx = n2 + k  # input qubits occupy the high-index positions
        phase_0 = np.exp(1j * theta / 2)   # |0⟩ component gets exp(+iθ/2)
        phase_1 = np.exp(-1j * theta / 2)  # |1⟩ component gets exp(-iθ/2)

        for basis_idx in range(len(state)):
            bit = (basis_idx >> qubit_idx) & 1
            state[basis_idx] *= phase_1 if bit else phase_0

    return state


def build_input_hamiltonian(n: int, n1: int, angles: np.ndarray) -> np.ndarray:
    """
    Build the input driving Hamiltonian H_in = Σ_i θ_i Z_i for the input register.

    This is a diagonal Hamiltonian where H_in[k] = Σ_{i∈input} θ_i * z_i(k)
    with z_i(k) = +1 if qubit i is |0⟩ in basis state k, else -1.

    Used in combined time evolution: H_total = H_reservoir + H_input

    Args:
        n: Total qubits (reservoir + input).
        n1: Number of input qubits (last n1 qubits in the ordering).
        angles: Encoding angles, shape (n1,).

    Returns:
        H_in_diag: Diagonal of H_input, shape (2**n,), real-valued.
    """
    n2 = n - n1
    N = 2**n
    H_in_diag = np.zeros(N)
    for k_local, theta in enumerate(angles):
        qubit_idx = n2 + k_local
        for basis_idx in range(N):
            bit = (basis_idx >> qubit_idx) & 1
            z_val = 1 - 2 * bit  # +1 if |0⟩, -1 if |1⟩
            H_in_diag[basis_idx] += theta * z_val
    return H_in_diag


def output_signal_format(
    z_expectations: np.ndarray,
    timestamp: Optional[str] = None,
    backend: str = "numpy_statevector",
    reservoir_alpha: float = 0.9,
    n_shots: int = 1000,
) -> dict:
    """
    Format QRC output as the C1.4 interface contract dict.

    Uses a softmax over the reservoir state to compute p_buy, p_sell, p_hold.
    The three outputs are: mean(first third), mean(middle third), mean(last third)
    of z-expectations — a coarse projection into the directional signal space.

    Args:
        z_expectations: Z-expectation values, shape (n,), in [-1, 1].
        timestamp: ISO-8601 timestamp string (or None).
        backend: Backend identifier string.
        reservoir_alpha: Spectral radius used for coupling matrix.
        n_shots: Number of measurement shots (for shot-based backends).

    Returns:
        signal: Dict matching the C1.4 weight-vector interface contract.
    """
    n = len(z_expectations)
    third = n // 3
    if third == 0:
        third = 1

    # Project z-expectations into (buy, sell, hold) via triple-band averaging
    buy_signal = float(np.mean(z_expectations[:third]))
    sell_signal = float(np.mean(z_expectations[third:2*third]))
    hold_signal = float(np.mean(z_expectations[2*third:]))

    # Softmax to convert to probability vector
    raw = np.array([buy_signal, sell_signal, hold_signal])
    shifted = raw - raw.max()
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()

    p_buy = float(probs[0])
    p_sell = float(probs[1])
    p_hold = float(probs[2])
    confidence = float(np.max(probs))

    return {
        "signal_type": "qrc_directional",
        "timestamp": timestamp,
        "p_buy": p_buy,
        "p_sell": p_sell,
        "p_hold": p_hold,
        "confidence": confidence,
        "backend": backend,
        "n_shots": n_shots,
        "reservoir_alpha": reservoir_alpha,
    }
