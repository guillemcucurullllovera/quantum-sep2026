"""Constraint-suite QUBO builders for portfolio optimization.

Neither Gen1 nor Gen2 supported anything beyond a bare cardinality
constraint. This module is the net-new platform layer: a penalty-method
QUBO builder that composes cardinality, turnover, transaction cost, sector
exposure, liquidity, and (via `LongShortQUBOBuilder`) long/short + leverage
constraints — the actual constraint surface a real portfolio-construction
problem needs.

Penalty method: every constraint g(x) = 0 (or inequality, converted to
equality via slack) is added to the objective as lambda * g(x)^2. This is
the standard QUBO constraint-encoding approach (Glover, Kochenberger &
Du 2019, "A Tutorial on Formulating and Using QUBO Models") — each
constraint here documents its own penalty term derivation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.errors import ValidationError


@dataclass
class SectorLimit:
    """Constrain the number of selected assets in `asset_indices`.

    When max_count is set, it is the enforced bound (equality-pull penalty
    toward exactly max_count — see PortfolioQUBOBuilder.build for why this
    takes priority over min_count rather than trying to enforce both at
    once). When only min_count is set, that is enforced instead."""
    name: str
    asset_indices: list[int]
    min_count: int = 0
    max_count: int | None = None


@dataclass
class PortfolioConstraints:
    cardinality_k: int | None = None
    cardinality_penalty: float = 2.0

    previous_holdings: np.ndarray | None = None
    turnover_limit: int | None = None
    turnover_penalty: float = 1.0

    transaction_cost_bps: float = 0.0
    transaction_cost_penalty: float = 1.0

    sector_limits: list[SectorLimit] = field(default_factory=list)
    sector_penalty: float = 1.5

    liquidity_scores: np.ndarray | None = None
    liquidity_penalty: float = 0.5
    liquidity_floor: float = 0.0  # penalize any selected asset below this score


class PortfolioQUBOBuilder:
    """Builds a QUBO matrix Q (x^T Q x, x in {0,1}^n, long-only "hold asset i")
    for portfolio selection under a composable constraint set.

    Base objective (mean-variance):
        C0(x) = risk_aversion * x^T Sigma x - (1 - risk_aversion) * mu^T x
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 0.5,
    ) -> None:
        self.mu = np.asarray(expected_returns, dtype=float)
        self.sigma = np.asarray(cov_matrix, dtype=float)
        self.q = risk_aversion
        self.n = len(self.mu)
        if self.sigma.shape != (self.n, self.n):
            raise ValidationError(
                f"cov_matrix shape {self.sigma.shape} does not match "
                f"expected_returns length {self.n}"
            )

    def build(self, constraints: PortfolioConstraints) -> np.ndarray:
        n = self.n
        Q = np.zeros((n, n))

        # --- base mean-variance objective ---
        Q += self.q * self.sigma
        for i in range(n):
            Q[i, i] -= (1 - self.q) * self.mu[i]

        # --- cardinality: lambda * (sum x_i - K)^2 ---
        if constraints.cardinality_k is not None:
            k = constraints.cardinality_k
            lam = constraints.cardinality_penalty
            for i in range(n):
                Q[i, i] += lam * (1 - 2 * k)
                for j in range(n):
                    if i != j:
                        Q[i, j] += lam

        # --- turnover: lambda * (sum_i |x_i - x_prev_i| - L)^2, linearized
        # for binary x via |x_i - p_i| = x_i + p_i - 2*x_i*p_i (p_i in {0,1}):
        # this expands to a quadratic in x with p treated as constants. ---
        if constraints.previous_holdings is not None and constraints.turnover_limit is not None:
            p = np.asarray(constraints.previous_holdings, dtype=float)
            if len(p) != n:
                raise ValidationError("previous_holdings length must match n_assets")
            lam = constraints.turnover_penalty
            L = constraints.turnover_limit
            # trades(x) = sum_i (x_i + p_i - 2 x_i p_i); expand (trades - L)^2
            # linear-in-x part of trades: c_i = 1 - 2 p_i ; constant part: sum(p)
            c = 1 - 2 * p
            const_p = float(np.sum(p))
            # (sum c_i x_i + const_p - L)^2 = sum_ij c_i c_j x_i x_j
            #   + 2*(const_p - L)*sum_i c_i x_i + (const_p - L)^2
            shift = const_p - L
            for i in range(n):
                Q[i, i] += lam * (c[i] * c[i] + 2 * shift * c[i])
                for j in range(n):
                    if i != j:
                        Q[i, j] += lam * c[i] * c[j]
            # constant term lam * shift**2 is dropped (doesn't affect argmin)

        # --- transaction cost: linear cost proportional to trading (only
        # charged on *new* positions relative to previous_holdings; closing
        # an existing position is not penalized here, matching a typical
        # "cost to enter" convention) ---
        if constraints.previous_holdings is not None and constraints.transaction_cost_bps > 0:
            p = np.asarray(constraints.previous_holdings, dtype=float)
            cost_per_new_trade = constraints.transaction_cost_bps * 1e-4 * constraints.transaction_cost_penalty
            for i in range(n):
                if p[i] < 0.5:  # not currently held -> entering costs
                    Q[i, i] += cost_per_new_trade

        # --- sector/exposure limits: equality-pull penalty toward a single
        # target count, lambda * (sum_{i in idx} x_i - target)^2. When BOTH
        # min_count and max_count are given, only max_count is enforced this
        # way — applying separate equality-pulls toward both bounds at once
        # (an earlier draft of this method) doesn't approximate a [min, max]
        # RANGE constraint, it creates two competing equality targets that
        # partially cancel and can leave the true max violated (caught by
        # tests/applications/test_portfolio.py). A real two-sided range
        # constraint without this weakness needs slack-variable encoding
        # (log2(max-min) extra qubits per bucket) — not yet built; see
        # ../README.md "what's still missing". min_count alone (no max_count)
        # is still enforced honestly as its own equality-pull. ---
        for lim in constraints.sector_limits:
            idx = lim.asset_indices
            lam = constraints.sector_penalty
            target = lim.max_count if lim.max_count is not None else (lim.min_count if lim.min_count > 0 else None)
            if target is not None:
                for i in idx:
                    Q[i, i] += lam * (1 - 2 * target)
                    for j in idx:
                        if i != j:
                            Q[i, j] += lam

        # --- liquidity penalty: additive cost for selecting illiquid assets ---
        if constraints.liquidity_scores is not None:
            liq = np.asarray(constraints.liquidity_scores, dtype=float)
            if len(liq) != n:
                raise ValidationError("liquidity_scores length must match n_assets")
            lam = constraints.liquidity_penalty
            floor = constraints.liquidity_floor
            for i in range(n):
                shortfall = max(0.0, floor - liq[i])
                Q[i, i] += lam * shortfall

        return Q

    def cost(self, x: np.ndarray, constraints: PortfolioConstraints) -> float:
        Q = self.build(constraints)
        x = np.asarray(x, dtype=float)
        return float(x @ Q @ x)


@dataclass
class LongShortConstraints:
    """Two binary variables per asset: y_long[i], y_short[i], with a mutual-
    exclusion penalty enforcing y_long[i] + y_short[i] <= 1 (an asset cannot
    be simultaneously long and short in this simple encoding)."""
    leverage_limit: float = 1.5
    leverage_penalty: float = 2.0
    exclusion_penalty: float = 5.0
    net_exposure_target: float | None = None  # target (n_long - n_short), soft
    net_exposure_penalty: float = 1.0


class LongShortQUBOBuilder:
    """Builds a QUBO over 2n binary variables [y_long_0..n, y_short_0..n]
    supporting long/short selection with a leverage-limit penalty. This is
    the constraint type neither generation implemented at all (both were
    long-only, cardinality-K selection)."""

    def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_aversion: float = 0.5) -> None:
        self.mu = np.asarray(expected_returns, dtype=float)
        self.sigma = np.asarray(cov_matrix, dtype=float)
        self.q = risk_aversion
        self.n = len(self.mu)

    def build(self, constraints: LongShortConstraints) -> np.ndarray:
        n = self.n
        dim = 2 * n  # [long_0..n-1, short_0..n-1]
        Q = np.zeros((dim, dim))

        # objective: risk on (long - short) position vector, return favors
        # long on positive-mu assets and short on negative-mu assets
        for i in range(n):
            Q[i, i] -= (1 - self.q) * self.mu[i]        # long benefits from +mu
            Q[n + i, n + i] += (1 - self.q) * self.mu[i]  # short benefits from -mu
            for j in range(n):
                sign_ll = 1
                Q[i, j] += self.q * self.sigma[i, j] * sign_ll
                Q[n + i, n + j] += self.q * self.sigma[i, j]
                Q[i, n + j] -= self.q * self.sigma[i, j]
                Q[n + i, j] -= self.q * self.sigma[i, j]

        # mutual exclusion: lambda * y_long_i * y_short_i (can't hold both)
        lam_excl = constraints.exclusion_penalty
        for i in range(n):
            Q[i, n + i] += lam_excl / 2
            Q[n + i, i] += lam_excl / 2

        # leverage: lambda * (sum(y_long) + sum(y_short) - L)^2
        lam_lev = constraints.leverage_penalty
        L = constraints.leverage_limit
        for i in range(dim):
            Q[i, i] += lam_lev * (1 - 2 * L)
            for j in range(dim):
                if i != j:
                    Q[i, j] += lam_lev

        # net exposure target: lambda * (sum(y_long) - sum(y_short) - target)^2
        if constraints.net_exposure_target is not None:
            target = constraints.net_exposure_target
            lam_net = constraints.net_exposure_penalty
            c = np.array([1.0] * n + [-1.0] * n)
            shift = -target
            for i in range(dim):
                Q[i, i] += lam_net * (c[i] * c[i] + 2 * shift * c[i])
                for j in range(dim):
                    if i != j:
                        Q[i, j] += lam_net * c[i] * c[j]

        return Q

    def decode(self, x: np.ndarray) -> dict:
        n = self.n
        long_mask = x[:n] > 0.5
        short_mask = x[n:] > 0.5
        return {
            "long": [i for i in range(n) if long_mask[i]],
            "short": [i for i in range(n) if short_mask[i]],
            "both_flagged": [i for i in range(n) if long_mask[i] and short_mask[i]],
            "n_long": int(long_mask.sum()),
            "n_short": int(short_mask.sum()),
        }


def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Shared QUBO -> Ising conversion (x_i = (1 - Z_i)/2), used by both the
    NumPy and Qiskit solvers so there is exactly one place this math lives.

    Exact identity: for every bitstring x, offset + h.z + sum_{i<j} J[i,j]*z_i*z_j
    equals x^T Q x precisely (z = 1 - 2x) — verified in
    tests/parity/test_qaoa_parity.py::test_qubo_to_ising_conversion_is_self_consistent.
    Binary x_i^2 = x_i means the diagonal must be handled as a linear
    contribution, not folded into the same quadratic substitution as the
    off-diagonal terms (see algorithms_extended/qubo_constraints_qiskit_optimization/
    for the equivalent SparsePauliOp version — dropping this distinction was
    the exact bug caught by that module's parity test during this build).
    """
    n = Q.shape[0]
    Qs = (Q + Q.T) / 2
    h = -Qs.diagonal().copy() / 2
    offset = float(Qs.diagonal().sum()) / 2

    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair_coeff = Qs[i, j]  # x_i*x_j's true coefficient is 2*Qs[i,j]; halved below by the (1-Zi)(1-Zj)/4 substitution
            offset += pair_coeff / 2
            h[i] -= pair_coeff / 2
            h[j] -= pair_coeff / 2
            J[i, j] = pair_coeff / 2

    return J, h, offset
