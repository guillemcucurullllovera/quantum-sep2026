"""
QAOA Portfolio Optimizer — Gen1
================================
Quantum Approximate Optimization Algorithm for constrained portfolio allocation.

Ported into quantum_sep2026 from the original Gen1 implementation
(Gen1, May 2026).
Network-dependent price-fetching helpers were dropped; the algorithm classes
below are unchanged in substance. See ../README.md for why this generation
is kept alongside gen2.

Solves the K-of-N asset selection problem:
  Given N assets with expected returns mu and covariance Sigma,
  select exactly K assets to maximize risk-adjusted returns.

This is NP-hard with discrete/cardinality constraints.
QAOA explores the feasible space via quantum superposition.
XY-mixer preserves the K-hot constraint natively.

Architecture:
  |psi_init> -> {Cost(gamma) . Mixer(beta)}^p -> Measure -> Best portfolio

References:
  - JPMorgan/Quantinuum QAOA portfolio optimization (2025)
  - arXiv:2602.14827 — XY-mixer for constrained combinatorial optimization

Usage:
    from algorithms.qaoa_qubo.gen1 import PortfolioQUBO, QAOACircuit, QAOAOptimizer

    qubo = PortfolioQUBO(expected_returns, cov_matrix, risk_aversion=0.5, cardinality_k=5)
    J, h, offset = qubo.build_ising()
    circuit = QAOACircuit(n_assets=11, J=J, h=h, p_layers=3, mixer='xy', cardinality_k=5)
    optimizer = QAOAOptimizer(circuit, qubo)
    result = optimizer.optimize()
"""

import math
import itertools
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ===========================================================================
# GATE PRIMITIVES — Generalized for N qubits
# ===========================================================================

def _apply_single_gate(state: np.ndarray, gate: np.ndarray,
                       qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 gate to a specific qubit in the statevector."""
    dim = 2 ** n_qubits
    tensor = state.reshape([2] * n_qubits)
    tensor = np.tensordot(gate, tensor, axes=([1], [qubit]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(dim)


def _rx(theta: float) -> np.ndarray:
    """RX rotation gate."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)


def _apply_xy_swap(state: np.ndarray, theta: float,
                   qubit_i: int, qubit_j: int, n_qubits: int) -> np.ndarray:
    """
    Apply XY-exchange gate: exp(-i * theta/2 * (XX + YY) / 2).
    Preserves Hamming weight by only mixing |01> <-> |10> subspace.
    """
    dim = 2 ** n_qubits
    new_state = state.copy()
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)

    visited = set()
    mask_i = 1 << (n_qubits - 1 - qubit_i)
    mask_j = 1 << (n_qubits - 1 - qubit_j)

    for idx in range(dim):
        if idx in visited:
            continue

        bit_i = (idx >> (n_qubits - 1 - qubit_i)) & 1
        bit_j = (idx >> (n_qubits - 1 - qubit_j)) & 1

        if bit_i == bit_j:
            continue

        partner = idx ^ mask_i ^ mask_j

        new_state[idx] = c * state[idx] - 1j * s * state[partner]
        new_state[partner] = -1j * s * state[idx] + c * state[partner]

        visited.add(idx)
        visited.add(partner)

    return new_state


# ===========================================================================
# PORTFOLIO QUBO — Cost function formulation
# ===========================================================================

class PortfolioQUBO:
    """
    Constructs the QUBO cost matrix for portfolio optimization.

    Problem: Select K of N assets to maximize risk-adjusted return.

    Cost function:
        C(x) = q * x^T Sigma x  -  (1-q) * mu^T x  +  lambda * (sum(x_i) - K)^2

    Where x_i in {0, 1} = include asset i.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 0.5,
        cardinality_k: int = 5,
        penalty_lambda: float = 2.0,
        asset_names: Optional[List[str]] = None,
    ):
        self.mu = np.asarray(expected_returns, dtype=float)
        self.sigma = np.asarray(cov_matrix, dtype=float)
        self.q = risk_aversion
        self.k = cardinality_k
        self.lam = penalty_lambda
        self.n = len(self.mu)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n)]

        assert self.sigma.shape == (self.n, self.n), \
            f"Covariance matrix shape {self.sigma.shape} doesn't match N={self.n}"
        assert 1 <= self.k <= self.n, \
            f"Cardinality K={self.k} must be between 1 and N={self.n}"

    def cost(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        risk = self.q * x @ self.sigma @ x
        ret = (1 - self.q) * self.mu @ x
        penalty = self.lam * (np.sum(x) - self.k) ** 2
        return risk - ret + penalty

    def build_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO to Ising Hamiltonian coefficients via x_i = (1 - Z_i) / 2.

        Returns:
            J: (N, N) upper-triangular coupling matrix (Z_i Z_j terms)
            h: (N,) local field vector (Z_i terms)
            offset: constant energy offset
        """
        n = self.n
        q = self.q
        lam = self.lam
        k = self.k
        mu = self.mu
        sigma = self.sigma

        Q = np.zeros((n, n))
        Q += q * sigma

        for i in range(n):
            Q[i, i] += lam * (1 - 2 * k)
            for j in range(n):
                if i != j:
                    Q[i, j] += lam

        for i in range(n):
            Q[i, i] -= (1 - q) * mu[i]

        J = np.zeros((n, n))
        h = np.zeros(n)
        offset = 0.0

        for i in range(n):
            for j in range(n):
                offset += Q[i, j] / 4.0
                h[i] -= Q[i, j] / 4.0
                h[j] -= Q[i, j] / 4.0
                J[i, j] += Q[i, j] / 4.0

        offset += lam * k * k

        J_upper = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                J_upper[i, j] = J[i, j] + J[j, i]

        return J_upper, h, offset

    def classical_bruteforce(self, constrain_k: bool = True) -> Tuple[np.ndarray, float, List[Dict]]:
        """Enumerate portfolios and find the global optimum. Only feasible for N <= 20."""
        assert self.n <= 20, f"Brute force infeasible for N={self.n}"

        best_x = None
        best_cost = float('inf')
        all_results = []

        if constrain_k:
            for combo in itertools.combinations(range(self.n), self.k):
                x = np.zeros(self.n, dtype=float)
                for i in combo:
                    x[i] = 1.0
                c = self.cost(x)
                bits = sum(int(x[i]) << (self.n - 1 - i) for i in range(self.n))
                all_results.append((c, x.copy(), bits))
                if c < best_cost:
                    best_cost = c
                    best_x = x.copy()
        else:
            for bits in range(2 ** self.n):
                x = np.array([(bits >> (self.n - 1 - i)) & 1
                              for i in range(self.n)], dtype=float)
                c = self.cost(x)
                all_results.append((c, x.copy(), bits))
                if c < best_cost:
                    best_cost = c
                    best_x = x.copy()

        all_results.sort(key=lambda t: t[0])
        top_k = []
        for c, x, bits in all_results[:10]:
            selected = [self.asset_names[i] for i in range(self.n) if x[i] > 0.5]
            top_k.append({
                "cost": c,
                "bitstring": "".join(str(int(b)) for b in x),
                "n_assets": int(sum(x)),
                "selected": selected,
                "portfolio": self.evaluate_portfolio(x),
            })

        return best_x, best_cost, top_k

    def decode_bitstring(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x)
        n_selected = max(1, np.sum(x > 0.5))
        weight_per_asset = 1.0 / n_selected
        return {
            self.asset_names[i]: weight_per_asset
            for i in range(self.n)
            if x[i] > 0.5
        }

    def evaluate_portfolio(self, x: np.ndarray, annualize_hours: float = None) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        n_selected = int(np.sum(x > 0.5))

        if n_selected == 0:
            return {"expected_return": 0, "risk": 0, "sharpe": 0,
                    "sharpe_annual": 0, "n_assets": 0, "cost": self.cost(x)}

        w = x / max(n_selected, 1)

        exp_ret = float(self.mu @ w)
        risk = float(np.sqrt(w @ self.sigma @ w))
        sharpe = exp_ret / max(risk, 1e-10)

        if annualize_hours:
            sharpe_annual = sharpe * math.sqrt(annualize_hours)
        else:
            sharpe_annual = sharpe

        return {
            "expected_return": round(exp_ret, 6),
            "risk": round(risk, 6),
            "sharpe": round(sharpe, 4),
            "sharpe_annual": round(sharpe_annual, 4),
            "n_assets": n_selected,
            "cost": round(float(self.cost(x)), 6),
        }


# ===========================================================================
# QAOA CIRCUIT — Quantum optimization circuit
# ===========================================================================

class QAOACircuit:
    """
    QAOA circuit for portfolio optimization.

    With XY-mixer: initial state is the uniform superposition over all K-hot
    bitstrings; the mixer preserves this K-hot constraint throughout.
    With standard mixer: initial state is |+>^N.
    """

    def __init__(
        self,
        n_assets: int,
        J: np.ndarray,
        h: np.ndarray,
        p_layers: int = 3,
        mixer: str = "xy",
        cardinality_k: int = 5,
    ):
        self.n = n_assets
        self.dim = 2 ** n_assets
        self.J = np.asarray(J)
        self.h = np.asarray(h)
        self.p = p_layers
        self.mixer = mixer
        self.k = cardinality_k
        self.n_params = 2 * p_layers

        self._cost_diagonal = self._build_cost_diagonal()
        self._init_state = self._build_initial_state()

    def _build_cost_diagonal(self) -> np.ndarray:
        diag = np.zeros(self.dim)
        for idx in range(self.dim):
            energy = 0.0
            for i in range(self.n):
                zi = 1 - 2 * ((idx >> (self.n - 1 - i)) & 1)
                energy += self.h[i] * zi
                for j in range(i + 1, self.n):
                    zj = 1 - 2 * ((idx >> (self.n - 1 - j)) & 1)
                    energy += self.J[i, j] * zi * zj
            diag[idx] = energy
        return diag

    def _build_initial_state(self) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)

        if self.mixer == "xy":
            count = 0
            for idx in range(self.dim):
                if bin(idx).count('1') == self.k:
                    state[idx] = 1.0
                    count += 1
            if count > 0:
                state /= math.sqrt(count)
        else:
            state[:] = 1.0 / math.sqrt(self.dim)

        return state

    def _apply_cost_layer(self, state: np.ndarray, gamma: float) -> np.ndarray:
        phases = np.exp(-1j * gamma * self._cost_diagonal)
        return state * phases

    def _apply_mixer_layer(self, state: np.ndarray, beta: float) -> np.ndarray:
        if self.mixer == "xy":
            for i in range(self.n):
                j = (i + 1) % self.n
                state = _apply_xy_swap(state, beta, i, j, self.n)
        else:
            gate = _rx(2 * beta)
            for i in range(self.n):
                state = _apply_single_gate(state, gate, i, self.n)
        return state

    def run(self, params: np.ndarray) -> np.ndarray:
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} params, got {len(params)}"

        state = self._init_state.copy()
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            state = self._apply_cost_layer(state, gamma)
            state = self._apply_mixer_layer(state, beta)
        return state

    def expectation(self, params: np.ndarray) -> float:
        state = self.run(params)
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self._cost_diagonal))

    def sample_probabilities(self, params: np.ndarray) -> Dict[str, float]:
        state = self.run(params)
        probs = np.abs(state) ** 2
        result = {}
        for idx in range(self.dim):
            p = probs[idx]
            if p > 1e-6:
                bs = format(idx, f'0{self.n}b')
                result[bs] = float(p)
        return result

    def top_bitstrings(self, params: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        probs = self.sample_probabilities(params)
        sorted_bs = sorted(probs.items(), key=lambda x: -x[1])
        return sorted_bs[:top_k]

    def amplitude_amplification(self, params: np.ndarray,
                                 target_bitstrings: List[str],
                                 n_iterations: int = None) -> np.ndarray:
        """Grover-style amplitude amplification to boost target bitstrings
        after QAOA optimization (JPMorgan QAOA+AA approach, arXiv:2308.02342)."""
        qaoa_state = self.run(params)

        target_indices = set()
        for bs in target_bitstrings:
            target_indices.add(int(bs, 2))

        p_target = sum(abs(qaoa_state[idx]) ** 2 for idx in target_indices)

        if p_target < 1e-15 or p_target > 0.5:
            return qaoa_state

        theta = math.asin(math.sqrt(p_target))
        if n_iterations is None:
            n_iterations = max(1, int(math.pi / (4 * theta) - 0.5))
            n_iterations = min(n_iterations, 10)

        state = qaoa_state.copy()
        for _ in range(n_iterations):
            for idx in target_indices:
                state[idx] = -state[idx]
            overlap = np.vdot(qaoa_state, state)
            state = 2 * overlap * qaoa_state - state
        return state


# ===========================================================================
# QAOA OPTIMIZER — Classical outer loop
# ===========================================================================

class QAOAOptimizer:
    """Classical-quantum hybrid optimizer for QAOA parameters (multi-restart,
    adiabatic-schedule initialization, optional amplitude amplification)."""

    def __init__(
        self,
        circuit: QAOACircuit,
        qubo: PortfolioQUBO,
        method: str = "COBYLA",
        maxiter: int = 300,
        n_restarts: int = 5,
    ):
        self.circuit = circuit
        self.qubo = qubo
        self.method = method
        self.maxiter = maxiter
        self.n_restarts = n_restarts

    def _compute_energy_scale(self) -> Tuple[float, float]:
        k_hot_energies = []
        for idx in range(self.circuit.dim):
            if bin(idx).count('1') == self.circuit.k:
                k_hot_energies.append(self.circuit._cost_diagonal[idx])

        if not k_hot_energies:
            return 0.5, 0.5

        e_min = min(k_hot_energies)
        e_max = max(k_hot_energies)
        energy_spread = max(abs(e_max - e_min), 1e-10)

        gamma_max = math.pi / (2 * energy_spread)
        beta_max = math.pi / 4.0
        return gamma_max, beta_max

    def _generate_init_params(self, restart_idx: int, n_params: int) -> np.ndarray:
        p = n_params // 2
        rng = np.random.RandomState(restart_idx * 17 + 42)
        gamma_max, beta_max = self._compute_energy_scale()

        if restart_idx == 0:
            gammas = np.array([gamma_max * (l + 1) / p for l in range(p)])
            betas = np.array([beta_max * (1 - l / p) for l in range(p)])
        elif restart_idx == 1:
            gammas = np.array([gamma_max * (1 - l / p) for l in range(p)])
            betas = np.array([beta_max * (l + 1) / p for l in range(p)])
        else:
            noise_scale = 0.15
            gammas = np.array([gamma_max * (l + 1) / p for l in range(p)])
            betas = np.array([beta_max * (1 - l / p) for l in range(p)])
            gammas += rng.normal(0, noise_scale * gamma_max, p)
            betas += rng.normal(0, noise_scale * beta_max, p)
            gammas = np.clip(gammas, 0.01, gamma_max * 1.5)
            betas = np.clip(betas, 0.01, beta_max * 1.5)

        params = np.empty(n_params)
        for l in range(p):
            params[2 * l] = gammas[l]
            params[2 * l + 1] = betas[l]
        return params

    def optimize(self, verbose: bool = False) -> Dict[str, Any]:
        n_params = self.circuit.n_params
        best_energy = float('inf')
        best_params_found = None
        all_restarts = []
        all_candidate_bitstrings = []

        for restart in range(self.n_restarts):
            x0 = self._generate_init_params(restart, n_params)
            restart_history = []

            def objective(params):
                e = self.circuit.expectation(params)
                restart_history.append(e)
                return e

            if HAS_SCIPY:
                gamma_max, beta_max = self._compute_energy_scale()
                rhobeg = max(0.01, min(0.1 * max(gamma_max, beta_max), 0.5))
                result = scipy_minimize(
                    objective, x0, method=self.method,
                    options={"maxiter": self.maxiter, "rhobeg": rhobeg},
                )
                energy, result_x, result_success = float(result.fun), result.x, result.success
            else:
                result_x, energy, result_success = _nelder_mead(objective, x0, maxiter=self.maxiter)
                energy = float(energy)

            top_bs = self.circuit.top_bitstrings(result_x, top_k=3)
            for bs_str, prob in top_bs:
                bs_x = np.array([int(b) for b in bs_str], dtype=float)
                cost = self.qubo.cost(bs_x)
                all_candidate_bitstrings.append((cost, bs_str, prob, result_x.copy(), energy))

            all_restarts.append({
                "restart": restart, "energy": energy,
                "converged": bool(result_success),
                "top_bitstring": top_bs[0][0] if top_bs else None,
            })

            if energy < best_energy:
                best_energy = energy
                best_params_found = np.array(result_x)

        all_candidate_bitstrings.sort(key=lambda t: t[0])
        seen = set()
        unique_candidates = []
        for cost, bs_str, prob, params, energy in all_candidate_bitstrings:
            if bs_str not in seen:
                seen.add(bs_str)
                unique_candidates.append((cost, bs_str, prob, params, energy))

        best_cost, best_bs_str, best_prob_pre_aa, _, _ = unique_candidates[0]

        target_bitstrings = [bs for _, bs, _, _, _ in unique_candidates[:3]]
        aa_state = self.circuit.amplitude_amplification(best_params_found, target_bitstrings)
        aa_probs = np.abs(aa_state) ** 2

        best_aa_prob = 0
        best_aa_bs = best_bs_str
        for idx in range(self.circuit.dim):
            if bin(idx).count('1') == self.circuit.k:
                p_aa = float(aa_probs[idx])
                if p_aa > best_aa_prob:
                    best_aa_prob = p_aa
                    best_aa_bs = format(idx, f'0{self.circuit.n}b')

        best_aa_x = np.array([int(b) for b in best_aa_bs], dtype=float)
        best_aa_cost = self.qubo.cost(best_aa_x)

        if best_aa_cost <= best_cost:
            final_bs_str, final_cost = best_aa_bs, best_aa_cost
        else:
            final_bs_str, final_cost = best_bs_str, best_cost

        best_x = np.array([int(b) for b in final_bs_str], dtype=float)
        portfolio = self.qubo.evaluate_portfolio(best_x)
        allocation = self.qubo.decode_bitstring(best_x)

        approx_ratio = None
        if self.circuit.n <= 16:
            _, bf_cost, _ = self.qubo.classical_bruteforce(constrain_k=True)
            if abs(bf_cost) > 1e-10:
                approx_ratio = float(final_cost / bf_cost)

        return {
            "best_params": best_params_found.tolist(),
            "best_energy": float(best_energy),
            "best_bitstring": final_bs_str,
            "best_qubo_cost": float(final_cost),
            "portfolio": portfolio,
            "allocation": allocation,
            "approximation_ratio": approx_ratio,
            "n_restarts": self.n_restarts,
            "all_restarts": all_restarts,
        }


def _nelder_mead(func, x0, maxiter=300, tol=1e-8,
                 alpha=1.0, gamma_nm=2.0, rho=0.5, sigma_nm=0.5):
    """Nelder-Mead simplex optimization (no scipy dependency)."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        point[i] += 0.5
        simplex.append(point)

    f_values = [func(s) for s in simplex]

    for _ in range(maxiter):
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]

        if abs(f_values[-1] - f_values[0]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1], f_values[-1] = xr, fr
        elif fr < f_values[0]:
            xe = centroid + gamma_nm * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1], f_values[-1] = xe, fe
            else:
                simplex[-1], f_values[-1] = xr, fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < f_values[-1]:
                simplex[-1], f_values[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                    f_values[i] = func(simplex[i])

    best_idx = np.argmin(f_values)
    return simplex[best_idx], f_values[best_idx], True
