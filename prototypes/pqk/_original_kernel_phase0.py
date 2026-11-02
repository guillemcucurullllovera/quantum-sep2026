"""
[HISTORICAL PROTOTYPE — Gen1, preserved verbatim, no changes]
Preserved because it exposes the original high-level
  `FEATURE_MAPS`/`train_fqk_svm`/`train_pqk_svm` API that `phase2b_kernel_train.py`
  (also in this directory) depends on. currently-maintained port of this
  module lives at `algorithms/pqk/kernels_gen1.py` + `algorithms/pqk/feature_maps.py`
  — that version refactored the API (unified `encode()` interface, no top-level
  training-wrapper functions) rather than preserving it 1:1, so this file is kept
  separately as the true historical artifact, not a duplicate of the maintained code.
  Note: `CPMap` below cites Singh et al. (arXiv:2507.03689) — this predates
  LICENSES/NOTICE process; if `CPMap`'s construction is materially
  derived from that paper rather than independently designed, it should get a
  NOTICE entry too — flagged here for human/ledger follow-up, not resolved by
  this workstream.

Quantum Kernel Methods — Phase 2B
====================================
Implements Fidelity Quantum Kernel (FQK) and Projected Quantum Kernel (PQK)
for binary classification via SVM.

Three feature maps:
  1. AngleCNOT  — our VQC v2 encoding (RY(x·π) + CNOT ring), 2 layers
  2. ZZFeatureMap — RZ(x) encoding + RZZ(xi·xj) entangling interactions
  3. CPMap — resource-efficient map from Singh et al. (2507.03689):
             controlled-phase entanglement, linear nearest-neighbor

Kernel types:
  - FQK: K(xi,xj) = |⟨ψ(xi)|ψ(xj)⟩|²  (state overlap / fidelity)
  - PQK: project ψ(x) → [⟨Z₁⟩,...,⟨Z₈⟩], then RBF kernel on projections

Paper references:
  - Non-variational QKMs (2604.07896): fixed feature map, classical SVM
  - Benchmarking QKT (2408.10274): skip variational training, tune hyperparams
  - Projected QK (2505.14593): PQK robust with shallow circuits
  - Resource efficient QK (2507.03689): CP map, fewer entangling gates
  - Quantum temporal kernels (2312.01602): QK beats classical on price direction

All computations on statevector simulator (8 qubits = 256-dim, pure numpy).
"""

import math
import numpy as np
from typing import List, Tuple, Optional

N_QUBITS = 8
DIM = 2 ** N_QUBITS  # 256


# ── Gate primitives (shared with circuit_v2) ──────────────────────

def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def _hadamard() -> np.ndarray:
    s = 1.0 / math.sqrt(2)
    return np.array([[s, s], [s, -s]], dtype=complex)


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


def _apply_cphase(state: np.ndarray, control: int, target: int,
                  phase: float) -> np.ndarray:
    """Controlled-phase gate: adds phase to |11⟩ component."""
    n = N_QUBITS
    new_state = state.copy()
    phase_factor = np.exp(1j * phase)
    for idx in range(DIM):
        c_bit = (idx >> (n - 1 - control)) & 1
        t_bit = (idx >> (n - 1 - target)) & 1
        if c_bit and t_bit:
            new_state[idx] *= phase_factor
    return new_state


def _apply_rzz(state: np.ndarray, q1: int, q2: int, theta: float) -> np.ndarray:
    """RZZ(θ) gate: exp(-i·θ/2 · Z⊗Z)"""
    n = N_QUBITS
    new_state = state.copy()
    for idx in range(DIM):
        b1 = (idx >> (n - 1 - q1)) & 1
        b2 = (idx >> (n - 1 - q2)) & 1
        parity = 1 - 2 * (b1 ^ b2)  # +1 if same, -1 if different
        new_state[idx] *= np.exp(-1j * theta / 2 * parity)
    return new_state


def _cnot_ring(state: np.ndarray) -> np.ndarray:
    for i in range(N_QUBITS - 1):
        state = _apply_cnot(state, i, i + 1)
    state = _apply_cnot(state, N_QUBITS - 1, 0)
    return state


def _expectation_z(state: np.ndarray, qubit: int) -> float:
    n = N_QUBITS
    exp = 0.0
    for idx in range(DIM):
        prob = abs(state[idx]) ** 2
        bit = (idx >> (n - 1 - qubit)) & 1
        exp += prob * (1 - 2 * bit)
    return float(exp)


def _init_state() -> np.ndarray:
    state = np.zeros(DIM, dtype=complex)
    state[0] = 1.0
    return state


# ── Feature Maps ──────────────────────────────────────────────────

class AngleCNOTMap:
    """
    Our VQC v2 encoding layer (no variational params, just encoding).
    RY(xₖ·π) on each qubit → CNOT ring → RY(xₖ·π) second layer → CNOT ring.

    Two repetitions of encode+entangle to create richer feature interactions.
    """
    name = "AngleCNOT"
    reps = 2

    def encode(self, features: List[float]) -> np.ndarray:
        state = _init_state()
        for rep in range(self.reps):
            for i in range(N_QUBITS):
                state = _apply_single(state, _ry(features[i] * math.pi), i)
            state = _cnot_ring(state)
        return state


class ZZFeatureMap:
    """
    Standard ZZFeatureMap from Havlicek et al. (2019).
    H → RZ(xₖ) → RZZ(xᵢ·xⱼ) for connected pairs → repeat.

    The ZZ interactions create feature cross-terms in the kernel that
    capture pairwise correlations (e.g., momentum × volatility).
    """
    name = "ZZFeatureMap"
    reps = 2

    def encode(self, features: List[float]) -> np.ndarray:
        state = _init_state()
        H = _hadamard()

        for rep in range(self.reps):
            # Hadamard layer
            for i in range(N_QUBITS):
                state = _apply_single(state, H, i)

            # Single-qubit RZ encoding
            for i in range(N_QUBITS):
                state = _apply_single(state, _rz(features[i] * math.pi), i)

            # ZZ entangling: linear nearest-neighbor pairs
            for i in range(N_QUBITS - 1):
                theta_zz = (math.pi - features[i]) * (math.pi - features[i + 1])
                state = _apply_rzz(state, i, i + 1, theta_zz)

        return state


class CPMap:
    """
    Resource-efficient feature map inspired by Singh et al. (2507.03689).
    RY encoding + controlled-phase entanglement (linear chain).
    Fewer entangling gates than ZZ, but still creates feature interactions.
    Designed to work well with limited qubits.
    """
    name = "CPMap"
    reps = 2

    def encode(self, features: List[float]) -> np.ndarray:
        state = _init_state()

        for rep in range(self.reps):
            # RY encoding
            for i in range(N_QUBITS):
                state = _apply_single(state, _ry(features[i] * math.pi), i)

            # Controlled-phase entanglement: nearest-neighbor chain
            for i in range(N_QUBITS - 1):
                phase = features[i] * features[i + 1] * math.pi
                state = _apply_cphase(state, i, i + 1, phase)

        return state


# All feature maps
FEATURE_MAPS = {
    "AngleCNOT": AngleCNOTMap,
    "ZZFeatureMap": ZZFeatureMap,
    "CPMap": CPMap,
}


# ── Fidelity Quantum Kernel (FQK) ────────────────────────────────

def fidelity_kernel_element(state_i: np.ndarray, state_j: np.ndarray) -> float:
    """
    K(xi, xj) = |⟨ψ(xi)|ψ(xj)⟩|²

    The squared overlap between quantum states. Measures how similar
    two data points are in the exponentially large Hilbert space.
    """
    overlap = np.dot(state_i.conj(), state_j)
    return float(abs(overlap) ** 2)


def compute_fidelity_gram(
    X: np.ndarray,
    feature_map,
    X2: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the Gram matrix K[i,j] = fidelity(X[i], X[j]).

    If X2 is provided, compute K[i,j] = fidelity(X[i], X2[j]).
    This is used for train-vs-val kernel computation.

    Returns: (N1 × N2) Gram matrix
    """
    # Pre-compute all quantum states
    if verbose:
        print(f"    Encoding {len(X)} samples...", end="", flush=True)
    states_1 = [feature_map.encode(list(x)) for x in X]
    if verbose:
        print(" done.", flush=True)

    if X2 is not None:
        if verbose:
            print(f"    Encoding {len(X2)} samples...", end="", flush=True)
        states_2 = [feature_map.encode(list(x)) for x in X2]
        if verbose:
            print(" done.", flush=True)
    else:
        states_2 = states_1

    N1, N2 = len(states_1), len(states_2)
    K = np.zeros((N1, N2))

    total = N1 * N2
    if verbose:
        print(f"    Computing {N1}×{N2} = {total:,} kernel elements...", flush=True)

    for i in range(N1):
        for j in range(N2):
            K[i, j] = fidelity_kernel_element(states_1[i], states_2[j])

        if verbose and (i + 1) % max(1, N1 // 10) == 0:
            pct = (i + 1) / N1 * 100
            print(f"      {pct:.0f}% ({i+1}/{N1} rows)", flush=True)

    return K


# ── Projected Quantum Kernel (PQK) ───────────────────────────────

def project_state(state: np.ndarray) -> np.ndarray:
    """
    Project quantum state to classical feature vector via per-qubit ⟨Z⟩.
    Returns 8-dimensional vector of expectation values in [-1, 1].
    """
    return np.array([_expectation_z(state, q) for q in range(N_QUBITS)])


def compute_projected_features(
    X: np.ndarray,
    feature_map,
    verbose: bool = True,
) -> np.ndarray:
    """
    Encode all samples and project to classical feature vectors.
    Returns: (N × 8) matrix of projected features.
    """
    N = len(X)
    projections = np.zeros((N, N_QUBITS))

    for i in range(N):
        state = feature_map.encode(list(X[i]))
        projections[i] = project_state(state)

        if verbose and (i + 1) % max(1, N // 10) == 0:
            pct = (i + 1) / N * 100
            print(f"      {pct:.0f}% ({i+1}/{N})", flush=True)

    return projections


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """RBF (Gaussian) kernel on classical feature vectors."""
    # ||x - y||² = ||x||² + ||y||² - 2·x·y
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
    dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T
    return np.exp(-gamma * dist_sq)


def compute_projected_gram(
    X: np.ndarray,
    feature_map,
    X2: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute PQK Gram matrix:
    1. Encode + project all samples to ⟨Z⟩ vectors
    2. Apply RBF kernel on projected vectors

    Returns: (gram_matrix, projections_X, projections_X2_or_None)
    """
    if verbose:
        print(f"    Projecting {len(X)} train samples...", flush=True)
    proj_X = compute_projected_features(X, feature_map, verbose)

    proj_X2 = None
    if X2 is not None:
        if verbose:
            print(f"    Projecting {len(X2)} val samples...", flush=True)
        proj_X2 = compute_projected_features(X2, feature_map, verbose)
        K = rbf_kernel(proj_X, proj_X2, gamma=gamma)
    else:
        K = rbf_kernel(proj_X, proj_X, gamma=gamma)

    return K, proj_X, proj_X2


# ── SVM (pure numpy, no sklearn dependency) ───────────────────────

class SimpleSVM:
    """
    Simplified SVM using kernel + dual coordinate ascent.
    For small datasets (N < 5000), this is fast enough.

    Solves: max Σαᵢ - 0.5·ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    subject to: 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0.0
        self.K_train = None
        self.y_train = None
        self.support_mask = None

    def fit(self, K: np.ndarray, y: np.ndarray):
        """
        Train SVM on precomputed kernel matrix K (N×N) with labels y ∈ {-1, +1}.
        Uses Sequential Minimal Optimization (SMO)-style coordinate ascent.
        """
        N = len(y)
        y = y.astype(float)
        self.alpha = np.zeros(N)
        self.y_train = y.copy()
        self.K_train = K.copy()

        for iteration in range(self.max_iter):
            alpha_old = self.alpha.copy()
            num_changed = 0

            for i in range(N):
                # Compute decision function for sample i
                f_i = np.dot(self.alpha * y, K[i]) + self.b
                E_i = f_i - y[i]

                # Check KKT violation
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):

                    # Pick random j != i
                    j = i
                    while j == i:
                        j = np.random.randint(N)

                    f_j = np.dot(self.alpha * y, K[j]) + self.b
                    E_j = f_j - y[j]

                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if abs(L - H) < 1e-10:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alpha_j_new = self.alpha[j] - y[j] * (E_i - E_j) / eta
                    alpha_j_new = np.clip(alpha_j_new, L, H)

                    if abs(alpha_j_new - self.alpha[j]) < 1e-8:
                        continue

                    # Update alpha_i
                    alpha_i_new = self.alpha[i] + y[i] * y[j] * (self.alpha[j] - alpha_j_new)

                    # Update bias
                    b1 = self.b - E_i - y[i] * (alpha_i_new - self.alpha[i]) * K[i, i] \
                         - y[j] * (alpha_j_new - self.alpha[j]) * K[i, j]
                    b2 = self.b - E_j - y[i] * (alpha_i_new - self.alpha[i]) * K[i, j] \
                         - y[j] * (alpha_j_new - self.alpha[j]) * K[j, j]

                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new
                    num_changed += 1

            # Convergence check
            if num_changed == 0:
                break

        self.support_mask = self.alpha > 1e-6
        n_sv = int(np.sum(self.support_mask))
        return n_sv

    def predict_kernel(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict using precomputed kernel matrix K_test (M × N_train).
        Returns class predictions in {-1, +1}.
        """
        # f(x) = Σ αᵢyᵢK(x, xᵢ) + b
        decision = K_test @ (self.alpha * self.y_train) + self.b
        return np.sign(decision)

    def decision_function(self, K_test: np.ndarray) -> np.ndarray:
        """Raw decision values (before sign)."""
        return K_test @ (self.alpha * self.y_train) + self.b

    def accuracy(self, K_test: np.ndarray, y_test: np.ndarray) -> float:
        preds = self.predict_kernel(K_test)
        return float(np.mean(preds == y_test))


# ── Convenience: full train+eval pipeline ─────────────────────────

def train_fqk_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_map_name: str = "ZZFeatureMap",
    C: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Full FQK pipeline: encode → gram matrix → SVM → evaluate.

    Args:
        X_train: (N_train, 8) features
        y_train: (N_train,) labels in {0, 1}
        X_val: (N_val, 8) features
        y_val: (N_val,) labels in {0, 1}
        feature_map_name: which feature map to use
        C: SVM regularization
    """
    import time

    fmap = FEATURE_MAPS[feature_map_name]()

    # Convert labels to {-1, +1} for SVM
    y_tr_svm = 2 * y_train.astype(float) - 1
    y_va_svm = 2 * y_val.astype(float) - 1

    # Train gram matrix
    if verbose:
        print(f"\n  FQK [{fmap.name}]:")
    t0 = time.time()
    K_train = compute_fidelity_gram(X_train, fmap, verbose=verbose)
    t_gram_train = time.time() - t0

    # Val gram matrix (val vs train)
    t0 = time.time()
    K_val = compute_fidelity_gram(X_val, fmap, X2=X_train, verbose=verbose)
    t_gram_val = time.time() - t0

    # Train SVM
    if verbose:
        print(f"    Training SVM (C={C})...", flush=True)
    t0 = time.time()
    svm = SimpleSVM(C=C, max_iter=2000)
    n_sv = svm.fit(K_train, y_tr_svm)
    t_svm = time.time() - t0

    # Evaluate
    train_acc = svm.accuracy(K_train, y_tr_svm)
    val_acc = svm.accuracy(K_val, y_va_svm)

    if verbose:
        print(f"    Support vectors: {n_sv}/{len(y_train)}")
        print(f"    Train accuracy: {train_acc:.4f}")
        print(f"    Val accuracy:   {val_acc:.4f}")
        print(f"    Time: gram_train={t_gram_train:.1f}s, "
              f"gram_val={t_gram_val:.1f}s, svm={t_svm:.1f}s")

    return {
        "feature_map": fmap.name,
        "kernel_type": "fidelity",
        "train_acc": train_acc,
        "val_acc": val_acc,
        "n_support_vectors": n_sv,
        "C": C,
        "time_gram_train": t_gram_train,
        "time_gram_val": t_gram_val,
        "time_svm": t_svm,
        "svm": svm,
    }


def train_pqk_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_map_name: str = "ZZFeatureMap",
    gamma: float = 1.0,
    C: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Full PQK pipeline: encode → project ⟨Z⟩ → RBF kernel → SVM → evaluate.
    Much faster than FQK (linear in N, not quadratic for the quantum part).
    """
    import time

    fmap = FEATURE_MAPS[feature_map_name]()

    y_tr_svm = 2 * y_train.astype(float) - 1
    y_va_svm = 2 * y_val.astype(float) - 1

    if verbose:
        print(f"\n  PQK [{fmap.name}] (gamma={gamma}):")

    # Project train
    t0 = time.time()
    proj_train = compute_projected_features(X_train, fmap, verbose=verbose)
    t_proj_train = time.time() - t0

    # Project val
    t0 = time.time()
    proj_val = compute_projected_features(X_val, fmap, verbose=verbose)
    t_proj_val = time.time() - t0

    # Gram matrices via RBF
    K_train = rbf_kernel(proj_train, proj_train, gamma=gamma)
    K_val = rbf_kernel(proj_val, proj_train, gamma=gamma)

    # Train SVM
    if verbose:
        print(f"    Training SVM (C={C})...", flush=True)
    t0 = time.time()
    svm = SimpleSVM(C=C, max_iter=2000)
    n_sv = svm.fit(K_train, y_tr_svm)
    t_svm = time.time() - t0

    # Evaluate
    train_acc = svm.accuracy(K_train, y_tr_svm)
    val_acc = svm.accuracy(K_val, y_va_svm)

    if verbose:
        print(f"    Support vectors: {n_sv}/{len(y_train)}")
        print(f"    Train accuracy: {train_acc:.4f}")
        print(f"    Val accuracy:   {val_acc:.4f}")
        print(f"    Time: proj_train={t_proj_train:.1f}s, "
              f"proj_val={t_proj_val:.1f}s, svm={t_svm:.1f}s")

    return {
        "feature_map": fmap.name,
        "kernel_type": "projected",
        "train_acc": train_acc,
        "val_acc": val_acc,
        "n_support_vectors": n_sv,
        "C": C,
        "gamma": gamma,
        "time_proj_train": t_proj_train,
        "time_proj_val": t_proj_val,
        "time_svm": t_svm,
        "svm": svm,
        "proj_train": proj_train,
        "proj_val": proj_val,
    }
