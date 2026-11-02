"""
QRC-10 — Quantum Reservoir Computing, Version 1.

Adapted from phase0/quantum_qiskit/diagnostics/p0_3_replication/_qrc.py
(Li, Mukhopadhyay, Bayat, Habibnia — arXiv:2505.13933) with:
    1. Eigenvalue normalization fix in random_couplings()  ← CRITICAL BUG FIX
    2. LGBM classifier for triple-barrier directional labels
    3. numpy-only matrix exponential (eigh, no scipy required)
    4. Output contract for C1.4 Bayesian weighter integration

Architecture: 10q total (n1 input + n2 hidden), IsingXX transverse-field
Hamiltonian, 3-step temporal window, Z-expectation readout.

Readout for directional task: LGBM classifier on reservoir states.
Readout for continuous task: ridge regression (preserved from phase0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

# Optional heavy dependencies — imported lazily so the module is always importable
def _require_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for QRCClassifier.fit(). "
            "Install with: pip install lightgbm"
        )


# ── Pauli matrices ──────────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def _kron_chain(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def pauli_on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """Place a 2×2 Pauli op on qubit i of an n-qubit system."""
    return _kron_chain([op if k == i else _I2 for k in range(n)])


def two_pauli_on(op: np.ndarray, i: int, j: int, n: int) -> np.ndarray:
    """Place op on qubits i and j."""
    return _kron_chain([op if k in (i, j) else _I2 for k in range(n)])


# ── Hamiltonian ─────────────────────────────────────────────────────────────

def build_ising_hamiltonian(
    n: int,
    couplings: np.ndarray,
    field_nu: float = 1.0,
) -> np.ndarray:
    """
    Transverse-field Ising Hamiltonian:
        H = Σ_{i<j} J_ij X_i X_j  +  ν Σ_i Z_i

    Args:
        n         : total qubits
        couplings : (n, n) symmetric coupling matrix
        field_nu  : transverse field strength (paper: 1.0)
    Returns:
        (2^n, 2^n) Hermitian matrix
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        for j in range(i + 1, n):
            H += couplings[i, j] * two_pauli_on(_X, i, j, n)
    for i in range(n):
        H += field_nu * pauli_on(_Z, i, n)
    return H


def random_couplings(
    n: int,
    rng: np.random.Generator,
    alpha: float = 0.9,
) -> np.ndarray:
    """
    Generate random coupling matrix J with spectral radius = alpha.

    BUG FIX (vs phase0 _qrc.py): the original code returned U(0,1) values
    without spectral radius normalization. For n=10 the spectral radius of a
    random positive symmetric matrix is O(n/2) ≈ 5, which is >> 1. This puts
    the system deep in the ferromagnetic phase (not near the edge of chaos),
    destroys the echo state property, and makes reservoir dynamics diverge.

    Fix: normalize so max|λ(J)| = alpha. With alpha=0.9 (just below 1), the
    XX coupling is calibrated relative to the transverse field ν=1, placing
    the system near the quantum phase transition — maximum memory capacity.

    Args:
        n     : number of qubits
        rng   : numpy random generator
        alpha : target spectral radius (0 < alpha < 1; default 0.9)
    Returns:
        (n, n) symmetric coupling matrix with spectral radius = alpha
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    U = rng.random((n, n))
    U = np.triu(U, k=1)
    J = U + U.T
    # Normalize: eigh is valid because J is real symmetric
    eigvals = np.linalg.eigvalsh(J)
    spectral_radius = np.max(np.abs(eigvals))
    if spectral_radius < 1e-12:
        raise ValueError("Coupling matrix has near-zero spectral radius.")
    J = J / spectral_radius * alpha
    return J


def evolution_operator(H: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    U(τ) = exp(-i H τ) via eigendecomposition (numpy only, no scipy).

    For Hermitian H: U = V · diag(exp(-i λ τ)) · V†
    where V, λ = eigh(H).  Valid because H is Hermitian.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    exp_eigs = np.exp(-1j * eigenvalues * tau)
    return eigenvectors @ (exp_eigs[:, None] * eigenvectors.conj().T)


# ── Encoding and partial trace ───────────────────────────────────────────────

def ry_encoding_matrix(angles: np.ndarray) -> np.ndarray:
    """R_Y(x) = ⊗_j exp(-i (x_j / 2) Y_j) as (2^n, 2^n) matrix."""
    R = np.array([[1.0]], dtype=np.complex128)
    for a in angles:
        c, s = np.cos(a / 2.0), np.sin(a / 2.0)
        ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
        R = np.kron(R, ry)
    return R


def encode_input_state(angles: np.ndarray) -> np.ndarray:
    """ρ_I = R_Y(x) |0⟩⟨0|^{⊗n1} R_Y(x)†."""
    R = ry_encoding_matrix(angles)
    col = R[:, 0:1]
    return col @ col.conj().T


def partial_trace_input(rho: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Trace out the first n1 qubits; return (2^n2, 2^n2) hidden state."""
    d1, d2 = 1 << n1, 1 << n2
    rho_re = rho.reshape(d1, d2, d1, d2)
    return np.einsum("ijik->jk", rho_re)


def precompute_z_diagonals(n: int) -> np.ndarray:
    """
    Returns (n, 2^n) array where row j = diagonal of Z_j in computational basis.
    Enables fast Tr(ρ Z_j) = z_diags[j] @ diag(ρ).
    """
    dim = 1 << n
    indices = np.arange(dim)
    diags = np.empty((n, dim), dtype=np.float64)
    for j in range(n):
        bit_j = (indices >> (n - 1 - j)) & 1
        diags[j] = 1.0 - 2.0 * bit_j
    return diags


# ── QRC forward pass ────────────────────────────────────────────────────────

def qrc_forward(
    inputs: np.ndarray,
    U: np.ndarray,
    n1: int,
    n2: int,
    U_dag: Optional[np.ndarray] = None,
    z_diags: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    n_windows-step reservoir forward pass for one prediction window.

    Args:
        inputs  : (n_windows, n1) — n_windows time steps × n1 input features
        U       : (2^n, 2^n) evolution operator
        n1, n2  : input / hidden qubit counts
        U_dag   : precomputed U†  (optional, computed if None)
        z_diags : precomputed Z diagonals (optional)
    Returns:
        (n1+n2,) Z-expectation vector in [-1, 1]
    """
    n = n1 + n2
    n_steps = len(inputs)
    dim_hidden = 1 << n2
    if U_dag is None:
        U_dag = U.conj().T
    if z_diags is None:
        z_diags = precompute_z_diagonals(n)

    rho_hidden = np.zeros((dim_hidden, dim_hidden), dtype=np.complex128)
    rho_hidden[0, 0] = 1.0

    for step in range(n_steps):
        rho_input = encode_input_state(inputs[step])
        rho = np.kron(rho_input, rho_hidden)
        rho = U @ rho @ U_dag
        if step < n_steps - 1:
            rho_hidden = partial_trace_input(rho, n1, n2)

    diag = np.real(np.diag(rho))
    return z_diags @ diag


# ── Persistent-state QRC (correct recurrent architecture) ────────────────────

def qrc_sequence_features(
    X: np.ndarray,
    U: np.ndarray,
    n1: int,
    n2: int,
    washout: int = 100,
    U_dag: Optional[np.ndarray] = None,
    z_diags: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Persistent-state QRC forward pass over a full sequence.

    Unlike qrc_forward() (windowed), this function maintains a persistent
    hidden state ρ_hidden across all T timesteps. At each step t:
        1. Encode input x_t as ρ_input = RY(x_t)|0⟩⟨0|RY(x_t)†
        2. Form joint state ρ = ρ_input ⊗ ρ_hidden
        3. Apply U: ρ ← U ρ U†
        4. Extract Z-expectation features from ρ
        5. Partial-trace out input qubits: ρ_hidden ← Tr_in[ρ]

    This is the correct QRC architecture for tasks requiring long memory
    (e.g. NARMA-10, which needs ≥10-step history). The windowed approach
    (qrc_forward) resets ρ_hidden each call and cannot capture dependencies
    beyond n_windows steps.

    Args:
        X       : (T, n1) input sequence (angles, e.g. from encode_sequence)
        U       : (2^n, 2^n) time-evolution operator
        n1, n2  : input / hidden qubit counts (n = n1 + n2)
        washout : initial timesteps to discard (hidden state warm-up)
        U_dag   : precomputed U†
        z_diags : precomputed Z diagonals (n, 2^n)
    Returns:
        (T - washout, n1+n2) reservoir feature matrix
    """
    T, d = X.shape
    if d != n1:
        raise ValueError(f"X has {d} features but n1={n1}")
    n = n1 + n2
    dim_hidden = 1 << n2
    if U_dag is None:
        U_dag = U.conj().T
    if z_diags is None:
        z_diags = precompute_z_diagonals(n)

    rho_hidden = np.zeros((dim_hidden, dim_hidden), dtype=np.complex128)
    rho_hidden[0, 0] = 1.0

    features = []
    for t in range(T):
        rho_input = encode_input_state(X[t])
        rho = np.kron(rho_input, rho_hidden)
        rho = U @ rho @ U_dag
        rho_hidden = partial_trace_input(rho, n1, n2)
        if t >= washout:
            diag = np.real(np.diag(rho))
            features.append(z_diags @ diag)

    return np.array(features)


# ── Ridge readout (for continuous targets) ──────────────────────────────────

def ridge_fit(M: np.ndarray, y: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    """Solve W = (MM^T + δI)^{-1} My (ridge regression readout)."""
    A = M @ M.T + delta * np.eye(M.shape[0])
    return np.linalg.solve(A, M @ y)


def ridge_predict(W: np.ndarray, M: np.ndarray) -> np.ndarray:
    """ŷ = W^T M."""
    return W @ M


# ── QRCClassifier: full pipeline for directional signals ────────────────────

@dataclass
class QRCConfig:
    n1: int = 5                  # input qubits
    n2: int = 5                  # hidden qubits (n = n1 + n2 = 10)
    alpha: float = 0.9           # spectral radius (just below edge of chaos)
    tau: float = 1.0             # evolution time
    field_nu: float = 1.0        # transverse field strength
    n_windows: int = 3           # look-back window length
    lgbm_n_estimators: int = 200
    lgbm_learning_rate: float = 0.05
    lgbm_max_depth: int = 4
    random_seed: int = 42
    backend: str = field(default_factory=lambda: os.getenv("QUANTUM_BACKEND", "statevector"))


class QRCClassifier:
    """
    QRC-10 directional classifier for financial time series.

    Output: triple-barrier labels — buy (1), sell (-1), hold (0).
    Plugs into C1.4 Bayesian weighter as alpha/challenger signal.

    Usage:
        clf = QRCClassifier(QRCConfig())
        clf.fit(X_train, y_train)   # y in {-1, 0, 1}
        signal = clf.predict(X_test)
    """

    def __init__(self, config: Optional[QRCConfig] = None):
        self.config = config or QRCConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._U: Optional[np.ndarray] = None
        self._U_dag: Optional[np.ndarray] = None
        self._z_diags: Optional[np.ndarray] = None
        self._lgbm_model = None
        self._fitted = False

    def _build_reservoir(self) -> None:
        cfg = self.config
        J = random_couplings(cfg.n1 + cfg.n2, self._rng, cfg.alpha)
        H = build_ising_hamiltonian(cfg.n1 + cfg.n2, J, cfg.field_nu)
        self._U = evolution_operator(H, cfg.tau)
        self._U_dag = self._U.conj().T
        self._z_diags = precompute_z_diagonals(cfg.n1 + cfg.n2)

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract reservoir features from input matrix X.

        Args:
            X : (T, n1) feature matrix (time-series, must be scaled to [-π, π])
        Returns:
            (T - n_windows + 1, n1+n2) reservoir feature matrix
        """
        cfg = self.config
        n_windows = cfg.n_windows
        T, d = X.shape
        if d != cfg.n1:
            raise ValueError(f"X has {d} features; expected {cfg.n1} (n1).")
        if T < n_windows:
            raise ValueError(f"X has {T} rows; need at least {n_windows}.")

        n_samples = T - n_windows + 1
        n_features = cfg.n1 + cfg.n2
        reservoir_features = np.empty((n_samples, n_features), dtype=np.float64)

        for i in range(n_samples):
            window = X[i : i + n_windows]          # (3, n1)
            z_exp = qrc_forward(
                window, self._U, cfg.n1, cfg.n2,
                self._U_dag, self._z_diags,
            )
            reservoir_features[i] = z_exp

        return reservoir_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QRCClassifier":
        """
        Build reservoir and train LGBM readout.

        Args:
            X : (T, n1) input features, scaled to [-π, π]
            y : (T - n_windows + 1,) labels in {-1, 0, 1}
        """
        lgb = _require_lgbm()
        self._build_reservoir()
        features = self._extract_features(X)

        # Align labels with reservoir output (first n_windows-1 rows dropped)
        n_drop = self.config.n_windows - 1
        y_aligned = y[n_drop:] if len(y) == len(X) else y
        if len(y_aligned) != len(features):
            raise ValueError(
                f"Label length mismatch: {len(y_aligned)} labels vs "
                f"{len(features)} reservoir outputs."
            )

        # LightGBM expects class labels 0, 1, 2 — map from {-1, 0, 1}
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_map[int(v)] for v in y_aligned])

        self._lgbm_model = lgb.LGBMClassifier(
            n_estimators=self.config.lgbm_n_estimators,
            learning_rate=self.config.lgbm_learning_rate,
            max_depth=self.config.lgbm_max_depth,
            class_weight="balanced",
            random_state=self.config.random_seed,
            verbose=-1,
        )
        self._lgbm_model.fit(features, y_mapped)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (n_samples, 3) probability matrix: [P(sell), P(hold), P(buy)]
        columns correspond to labels [-1, 0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        features = self._extract_features(X)
        return self._lgbm_model.predict_proba(features)

    def predict_signal(self, X: np.ndarray) -> dict:
        """
        Produce signal dict compatible with the meta-agent Redis schema.

        Returns:
            {
                "signal_type": "qrc_directional",
                "timestamp": "...",
                "p_buy": float,
                "p_sell": float,
                "p_hold": float,
                "confidence": float,   # max(p_buy, p_sell, p_hold)
                "backend": str,
                "reservoir_alpha": float,
            }
        """
        proba = self.predict_proba(X)
        last = proba[-1]           # most recent window
        p_sell, p_hold, p_buy = float(last[0]), float(last[1]), float(last[2])
        return {
            "signal_type": "qrc_directional",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "p_buy": p_buy,
            "p_sell": p_sell,
            "p_hold": p_hold,
            "confidence": float(max(p_buy, p_sell, p_hold)),
            "backend": self.config.backend,
            "reservoir_alpha": self.config.alpha,
        }


__all__ = [
    "QRCConfig",
    "QRCClassifier",
    "build_ising_hamiltonian",
    "random_couplings",
    "evolution_operator",
    "qrc_forward",
    "ridge_fit",
    "ridge_predict",
]
