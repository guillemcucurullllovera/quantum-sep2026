"""
QAOA v2 — Portfolio QUBO Optimizer
SK Warm-Start + Correlation-Based Complete-XY Mixer

Fixes from phase0 QAOA v1 (39% of brute-force Sharpe):
  Fix 1: SK mean-field warm-start angles (sk_warmstart.py)
  Fix 2: Correlation-based complete-XY mixer (mixers.py)
Expected improvement: 39% → 70-80% of brute-force Sharpe.

Pure numpy statevector simulation. Qiskit path optional.
For n≤20 qubits, full statevector (2^n ≤ 1M) is memory-feasible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from algorithms.qaoa_qubo.gen2.sk_warmstart import sk_warmstart_from_qubo, random_angles
from algorithms.qaoa_qubo.gen2.mixers import (
    correlation_xy_pairs,
    ring_xy_pairs,
    apply_mixer,
    initial_dicke_state,
    plus_state,
)


@dataclass
class QAOAConfig:
    p: int = 2                    # QAOA depth (number of cost+mixer layers)
    n_iter: int = 200             # Optimizer iterations
    lr: float = 0.02              # ADAM learning rate
    eps: float = 1e-8             # Optimizer epsilon
    mixer_threshold: float = 0.3  # Correlation threshold for XY mixer edges
    use_dicke_init: bool = True   # True = Dicke state; False = |+⟩^n
    target_k: Optional[int] = None  # Dicke state cardinality (assets to hold)
    seed: Optional[int] = None

    def __post_init__(self):
        if self.target_k is None:
            # Default: n/2 assets
            self.use_dicke_init = False  # Can't know n here; use + state


# ── Core statevector operations ───────────────────────────────────────────────

def build_cost_hamiltonian(J: np.ndarray) -> np.ndarray:
    """
    Build diagonal of cost Hamiltonian H_C = Σ_i J_ii Z_i + Σ_{i<j} J_ij Z_i Z_j.

    Handles both pure Ising (J diagonal = 0) and QUBO-converted Ising (J diagonal = h_i
    linear fields). For real financial QUBO, use qubo_to_ising_matrix(Q) first — zeroing
    the diagonal produces a ferromagnetic Ising model whose ground state (all-aligned) is
    unreachable by the Hamming-weight-preserving XY mixer.

    Args:
        J: Coupling matrix, shape (n, n). Diagonal J[i,i] = h_i (linear Z_i field).

    Returns:
        H_diag: Diagonal of H_C, shape (2**n,), real-valued.
    """
    n = J.shape[0]
    N = 2 ** n
    H_diag = np.zeros(N)
    # Precompute z values for all basis states and all qubits
    basis = np.arange(N, dtype=np.int64)
    z = 1 - 2 * ((basis[:, None] >> np.arange(n)[None, :]) & 1)  # (N, n)
    # Linear terms from diagonal: Σ_i J_ii Z_i
    for i in range(n):
        if abs(J[i, i]) > 1e-12:
            H_diag += J[i, i] * z[:, i]
    # Quadratic terms: Σ_{i<j} J_ij Z_i Z_j
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                H_diag += J[i, j] * z[:, i] * z[:, j]
    return H_diag


def qubo_to_ising_matrix(Q: np.ndarray) -> np.ndarray:
    """
    Convert QUBO matrix Q (binary x ∈ {0,1}) to Ising coupling matrix J (spin s ∈ {-1,+1}).

    Substituting x_i = (1 - s_i)/2:
        J[i,i] = -(Q[i,i] + Σ_{j≠i} Q[i,j]) / 2   (linear h_i field from diagonal + row sums)
        J[i,j] = Q[i,j] / 2                for i≠j  (quadratic ZiZj coupling)

    For K-asset cardinality-constrained QUBO with penalty A*(Σx_i - K)^2:
    Q_ii = -A*(1-2K), Q_ij = 2A for i≠j. After conversion, h_i fields encode the
    cardinality constraint in the Ising formulation so XY mixer can reach feasible states.

    Args:
        Q: QUBO matrix, shape (n, n). Symmetrized internally.

    Returns:
        J: Ising coupling matrix, shape (n, n), diagonal = h_i linear fields.
    """
    Q = (Q + Q.T) / 2  # symmetrize
    n = Q.shape[0]
    J = np.zeros_like(Q, dtype=float)
    row_sums = Q.sum(axis=1)  # includes diagonal
    for i in range(n):
        J[i, i] = -row_sums[i] / 2            # linear field h_i
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j] / 2         # quadratic coupling
    return J


def apply_cost_unitary(
    state: np.ndarray, H_diag: np.ndarray, gamma: float
) -> np.ndarray:
    """Apply U_C(γ) = exp(-i γ H_C) to statevector. H_C is diagonal."""
    return state * np.exp(-1j * gamma * H_diag)


def qaoa_statevector(
    state: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    H_diag: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """
    Compute the QAOA statevector after p layers.

    Args:
        state: Initial state, shape (2**n,).
        gamma: Cost angles, length p.
        beta: Mixer angles, length p.
        H_diag: Cost Hamiltonian diagonal, shape (2**n,).
        pairs: XY mixer qubit pairs.

    Returns:
        state: Final QAOA state after p layers.
    """
    p = len(gamma)
    for layer in range(p):
        state = apply_cost_unitary(state, H_diag, gamma[layer])
        state = apply_mixer(state, pairs, beta[layer])
    return state


def energy_expectation(state: np.ndarray, H_diag: np.ndarray) -> float:
    """E = ⟨ψ|H_C|ψ⟩ = Σ_k |ψ_k|^2 * H_C[k]."""
    return float(np.real(np.dot(np.abs(state) ** 2, H_diag)))


# ── ADAM optimizer (numpy) ────────────────────────────────────────────────────

def _adam_optimize(
    f: callable,
    x0: np.ndarray,
    n_iter: int,
    lr: float = 0.02,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
    finite_diff_eps: float = 0.02,
) -> Tuple[np.ndarray, list[float]]:
    """
    Maximize f(x) using ADAM with finite-difference gradient.

    Maximization: gradient ascent (not descent).

    Args:
        f: Objective function x → scalar (to maximize).
        x0: Initial parameter vector.
        n_iter: Number of update steps.
        lr: ADAM learning rate.
        beta1, beta2: ADAM momentum decay.
        eps_adam: ADAM numerical stability.
        finite_diff_eps: Step size for finite-difference gradient.

    Returns:
        x_best: Parameter vector achieving best f value seen.
        history: List of f values at each step.
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    best_val = -np.inf
    best_x = x.copy()

    for t in range(1, n_iter + 1):
        # Finite difference gradient (central differences)
        grad = np.zeros_like(x)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += finite_diff_eps
            xm = x.copy(); xm[k] -= finite_diff_eps
            grad[k] = (f(xp) - f(xm)) / (2 * finite_diff_eps)

        # ADAM update (ascent: += not -=)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x + lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        val = f(x)
        history.append(val)
        if val > best_val:
            best_val = val
            best_x = x.copy()

    return best_x, history


# ── Main QAOA v2 class ────────────────────────────────────────────────────────

class QAOAv2:
    """
    QAOA v2 for portfolio QUBO optimization.

    Incorporates both phase0 fixes:
    - SK warm-start initial angles (not random)
    - Correlation-based complete-XY mixer (not ring-XY)

    Comparison method: also runs ring-XY + random-init for benchmarking.
    """

    def __init__(self, config: Optional[QAOAConfig] = None):
        self.cfg = config or QAOAConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.best_angles_: Optional[np.ndarray] = None
        self.best_energy_: Optional[float] = None
        self.optimize_history_: list[float] = []
        self.corr_matrix_: Optional[np.ndarray] = None
        self.pairs_: Optional[list] = None
        self.H_diag_: Optional[np.ndarray] = None
        self.n_: Optional[int] = None

    def fit(
        self,
        J: np.ndarray,
        corr_matrix: Optional[np.ndarray] = None,
        warmstart: bool = True,
    ) -> "QAOAv2":
        """
        Optimize QAOA angles for QUBO coupling matrix J.

        Args:
            J: QUBO coupling matrix (n×n, symmetric). Diagonal ignored.
               J[i,j] > 0 means assets i,j should co-appear (correlated returns).
            corr_matrix: Asset correlation matrix for XY mixer (optional).
                         If None, complete graph XY mixer is used.
            warmstart: If True, use SK warm-start. If False, random init.

        Returns:
            self (fitted).
        """
        n = J.shape[0]
        self.n_ = n
        self.H_diag_ = build_cost_hamiltonian(J)

        # Build mixer pairs
        if corr_matrix is not None:
            self.corr_matrix_ = corr_matrix
            self.pairs_ = correlation_xy_pairs(corr_matrix, self.cfg.mixer_threshold)
        else:
            self.pairs_ = ring_xy_pairs(n)  # fallback

        # Initial state
        if self.cfg.use_dicke_init and self.cfg.target_k is not None:
            init_state = initial_dicke_state(n, self.cfg.target_k)
        else:
            init_state = plus_state(n)

        # Initial angles
        if warmstart:
            angles0 = sk_warmstart_from_qubo(J, self.cfg.p)
        else:
            angles0 = random_angles(self.cfg.p, self._rng)

        # Objective function
        p = self.cfg.p

        def objective(angles: np.ndarray) -> float:
            gamma = angles[:p]
            beta = angles[p:]
            state = qaoa_statevector(
                init_state.copy(), gamma, beta, self.H_diag_, self.pairs_
            )
            return energy_expectation(state, self.H_diag_)

        best_angles, history = _adam_optimize(
            objective, angles0, self.cfg.n_iter, self.cfg.lr
        )
        self.best_angles_ = best_angles
        self.best_energy_ = objective(best_angles)
        self.optimize_history_ = history
        return self

    def get_state(self) -> np.ndarray:
        """Return the optimized QAOA statevector."""
        if self.best_angles_ is None:
            raise RuntimeError("Call fit() first.")
        p = self.cfg.p
        gamma = self.best_angles_[:p]
        beta = self.best_angles_[p:]
        if self.cfg.use_dicke_init and self.cfg.target_k is not None:
            init = initial_dicke_state(self.n_, self.cfg.target_k)
        else:
            init = plus_state(self.n_)
        return qaoa_statevector(init, gamma, beta, self.H_diag_, self.pairs_)

    def sample_portfolio(
        self, n_samples: int = 1000, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample bitstrings from the optimized QAOA state.

        Returns the most probable bitstring as the portfolio selection:
        bit k = 1 means asset k is included.

        Args:
            n_samples: Number of bitstring samples.
            rng: Random generator (default: self._rng).

        Returns:
            portfolio: Most likely bitstring, shape (n,), values in {0, 1}.
        """
        if rng is None:
            rng = self._rng
        state = self.get_state()
        probs = np.abs(state) ** 2
        probs /= probs.sum()  # renormalise for numerical safety
        samples = rng.choice(len(state), size=n_samples, p=probs)
        counts = np.bincount(samples, minlength=len(state))
        best_idx = int(np.argmax(counts))
        portfolio = np.array(
            [(best_idx >> i) & 1 for i in range(self.n_)], dtype=np.int8
        )
        return portfolio

    def portfolio_weights(self, n_samples: int = 1000) -> np.ndarray:
        """
        Probability-weighted portfolio allocation (soft assignment).

        Returns the probability of each asset being selected, averaged
        over the sampled distribution. Useful for continuous weight vectors.

        Returns:
            weights: Asset weights, shape (n,), sum ≈ target_k.
        """
        state = self.get_state()
        probs = np.abs(state) ** 2
        n = self.n_
        # P(asset i = 1) = Σ_{k: bit_i(k)=1} |ψ_k|^2
        weights = np.zeros(n)
        for i in range(n):
            mask = (np.arange(len(state)) >> i) & 1
            weights[i] = float(np.dot(probs, mask))
        return weights

    def brute_force_optimum(
        self, k: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Brute-force maximum of the QUBO (exact, exponential time).

        Only feasible for n ≤ 20. Returns (max_energy, best_bitstring).
        Used for benchmarking QAOA approximation ratio.

        Args:
            k: If provided, restrict search to K-hot states (exactly k bits set).
               Required for cardinality-constrained portfolio QUBO where the
               unconstrained maximum is the infeasible all-assets-selected state.
               If None, searches all 2^n states (correct for unconstrained QUBO).
        """
        if self.H_diag_ is None:
            raise RuntimeError("Call fit() first.")
        n = self.n_
        if k is not None:
            # Restrict to K-hot states: exactly k assets selected
            all_idx = np.arange(len(self.H_diag_), dtype=np.int64)
            hw = np.array([bin(int(i)).count('1') for i in all_idx])
            feasible = all_idx[hw == k]
            if len(feasible) == 0:
                raise ValueError(f"No {k}-hot states in {n}-qubit system")
            local_best = int(np.argmax(self.H_diag_[feasible]))
            best_idx = int(feasible[local_best])
        else:
            best_idx = int(np.argmax(self.H_diag_))
        best_energy = float(self.H_diag_[best_idx])
        best_bits = np.array(
            [(best_idx >> i) & 1 for i in range(n)], dtype=np.int8
        )
        return best_energy, best_bits

    def approximation_ratio(self) -> float:
        """
        QAOA approximation ratio = QAOA_energy / brute_force_energy.

        Target: ≥ 0.70 (vs phase0 v1 ratio of ≈ 0.39).

        Uses target_k from config to restrict brute force to K-hot states when
        the QUBO is cardinality-constrained (portfolio optimization). Without
        target_k, all 2^n states are searched (correct for unconstrained QUBO).
        """
        if self.best_energy_ is None:
            raise RuntimeError("Call fit() first.")
        k = self.cfg.target_k  # None for unconstrained, K for portfolio QUBO
        bf_energy, _ = self.brute_force_optimum(k=k)
        if abs(bf_energy) < 1e-10:
            return 1.0
        return self.best_energy_ / bf_energy

    def benchmark_vs_ring_xy(
        self,
        J: np.ndarray,
        corr_matrix: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compare correlation-XY (v2) vs ring-XY (v1) on the same QUBO.

        Returns a dict with both results for the acceptance criteria test:
          test_qaoa_sharpe.py checks: v2_ratio >= v1_ratio + 0.10
        """
        # v2: correlation-XY + SK warm-start (this instance)
        self.fit(J, corr_matrix=corr_matrix, warmstart=True)
        ratio_v2 = self.approximation_ratio()

        # v1: ring-XY + SK warm-start (ablation: mixer only change)
        v1_cfg = QAOAConfig(
            p=self.cfg.p,
            n_iter=self.cfg.n_iter,
            lr=self.cfg.lr,
            seed=self.cfg.seed,
        )
        v1 = QAOAv2(v1_cfg)
        n = J.shape[0]
        v1.n_ = n
        v1.H_diag_ = self.H_diag_
        v1.pairs_ = ring_xy_pairs(n)
        angles0 = sk_warmstart_from_qubo(J, self.cfg.p)
        init = plus_state(n)

        def obj_v1(angles):
            gamma, beta = angles[:self.cfg.p], angles[self.cfg.p:]
            state = qaoa_statevector(init.copy(), gamma, beta, v1.H_diag_, v1.pairs_)
            return energy_expectation(state, v1.H_diag_)

        best_v1, _ = _adam_optimize(obj_v1, angles0, self.cfg.n_iter, self.cfg.lr)
        bf_energy, _ = self.brute_force_optimum()
        ratio_v1 = obj_v1(best_v1) / bf_energy if abs(bf_energy) > 1e-10 else 1.0

        return {
            "ratio_v2_correlation_xy": ratio_v2,
            "ratio_v1_ring_xy": ratio_v1,
            "improvement_pp": (ratio_v2 - ratio_v1) * 100,
            "brute_force_energy": bf_energy,
            "v2_energy": self.best_energy_,
        }
