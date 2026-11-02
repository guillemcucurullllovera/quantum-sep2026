"""Reusable discretized-distribution uncertainty models.

`algorithms.qae_iqae.finance_integrands` already discretizes a log-normal
density onto a grid inline, separately, in `build_crash_barrier` and
`build_call_spread_integrand` (and a histogram-based empirical variant in
`build_return_distribution_barrier`). This module factors that repeated
pattern into one reusable, independently-testable class rather than a third
copy of the same ~10 lines — used by `barrier_payoff.py` and
`basket_payoff.py` in this package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from algorithms.qae_iqae.finance_integrands import bs_lognormal_pdf


@dataclass(frozen=True)
class DiscretizedLogNormal:
    """A log-normal density for GBM terminal price S_T, discretized onto
    2**num_qubits equally-spaced midpoint bins over [low, high]."""

    s0: float
    r: float
    sigma: float
    t: float
    num_qubits: int
    low: float
    high: float

    @classmethod
    def from_sigma_window(
        cls, *, s0: float, r: float, sigma: float, t: float, num_qubits: int, sigmas: float = 3.5
    ) -> "DiscretizedLogNormal":
        """Auto-size [low, high] to +/- `sigmas` standard deviations of log(S_T)."""
        s = sigma * math.sqrt(t)
        drift = (r - 0.5 * sigma**2) * t
        low = max(1e-9, s0 * math.exp(drift - sigmas * s))
        high = s0 * math.exp(drift + sigmas * s)
        return cls(s0=s0, r=r, sigma=sigma, t=t, num_qubits=num_qubits, low=low, high=high)

    @property
    def num_grid_points(self) -> int:
        return 2**self.num_qubits

    def grid(self) -> list[float]:
        m = self.num_grid_points
        span = self.high - self.low
        return [self.low + span * (j + 0.5) / m for j in range(m)]

    def probabilities(self) -> list[float]:
        pts = self.grid()
        pdf = [bs_lognormal_pdf(s, self.s0, self.r, self.sigma, self.t) for s in pts]
        total = sum(pdf)
        if total <= 0:
            raise ValueError("degenerate grid (zero total density) — widen [low, high]")
        return [p / total for p in pdf]

    def state_preparation(self):
        """Return a Qiskit StatePreparation *gate* (not wrapped in a named
        sub-circuit) loading sqrt(p_j) amplitudes on the num_qubits-wide
        index register. Returning the gate directly, rather than an opaque
        named QuantumCircuit wrapping it, matters: Aer's SamplerV2 can
        decompose/simulate a standard StatePreparation gate but errors on an
        unrecognized custom-named composite instruction (found via the
        execution-wiring smoke test, not caught by Statevector-only tests,
        which decompose custom instructions automatically)."""
        from qiskit.circuit.library import StatePreparation

        probs = self.probabilities()
        amps = [math.sqrt(p) for p in probs]
        norm = math.sqrt(sum(a * a for a in amps)) or 1.0
        amps = [a / norm for a in amps]
        return StatePreparation(amps)


@dataclass(frozen=True)
class DiscretizedBivariateLogNormal:
    """Joint discretized density for two correlated GBM terminal prices,
    over a (num_qubits_a + num_qubits_b)-qubit joint index register.
    Used by basket_payoff.py. Models correlation via a bivariate log-normal
    closed form (correlated Brownian drivers), not two independent marginals.
    """

    s0_a: float
    s0_b: float
    r: float
    sigma_a: float
    sigma_b: float
    rho: float
    t: float
    num_qubits_a: int
    num_qubits_b: int
    low_a: float
    high_a: float
    low_b: float
    high_b: float

    @classmethod
    def from_sigma_window(
        cls, *, s0_a, s0_b, r, sigma_a, sigma_b, rho, t, num_qubits_a, num_qubits_b, sigmas=3.5
    ) -> "DiscretizedBivariateLogNormal":
        def window(s0, sigma):
            s = sigma * math.sqrt(t)
            drift = (r - 0.5 * sigma**2) * t
            return max(1e-9, s0 * math.exp(drift - sigmas * s)), s0 * math.exp(drift + sigmas * s)

        low_a, high_a = window(s0_a, sigma_a)
        low_b, high_b = window(s0_b, sigma_b)
        return cls(
            s0_a=s0_a, s0_b=s0_b, r=r, sigma_a=sigma_a, sigma_b=sigma_b, rho=rho, t=t,
            num_qubits_a=num_qubits_a, num_qubits_b=num_qubits_b,
            low_a=low_a, high_a=high_a, low_b=low_b, high_b=high_b,
        )

    def _grid(self, low, high, n_qubits):
        m = 2**n_qubits
        span = high - low
        return [low + span * (j + 0.5) / m for j in range(m)]

    def joint_grid(self):
        """Returns (grid_a, grid_b, joint_index -> (a_idx, b_idx) via row-major)."""
        return self._grid(self.low_a, self.high_a, self.num_qubits_a), self._grid(
            self.low_b, self.high_b, self.num_qubits_b
        )

    def joint_probabilities(self) -> list[float]:
        """Bivariate log-normal density on the joint grid via the standard
        correlated-Brownian-driver closed form: log S_a, log S_b are jointly
        normal with correlation rho."""
        ga, gb = self.joint_grid()
        drift_a = (self.r - 0.5 * self.sigma_a**2) * self.t
        drift_b = (self.r - 0.5 * self.sigma_b**2) * self.t
        s_a = self.sigma_a * math.sqrt(self.t)
        s_b = self.sigma_b * math.sqrt(self.t)
        rho = self.rho
        denom = 2 * math.pi * s_a * s_b * math.sqrt(1 - rho**2)
        dens = []
        for sb in gb:
            for sa in ga:  # row-major: index = ib * 2**n_a + ia
                za = (math.log(sa / self.s0_a) - drift_a) / s_a
                zb = (math.log(sb / self.s0_b) - drift_b) / s_b
                quad = (za**2 - 2 * rho * za * zb + zb**2) / (2 * (1 - rho**2))
                dens.append(math.exp(-quad) / (denom * sa * sb))
        total = sum(dens)
        if total <= 0:
            raise ValueError("degenerate joint grid — widen windows")
        return [d / total for d in dens]

    def state_preparation(self):
        """See DiscretizedLogNormal.state_preparation's docstring — returns the
        StatePreparation gate directly, not wrapped in a named sub-circuit."""
        from qiskit.circuit.library import StatePreparation

        probs = self.joint_probabilities()
        amps = [math.sqrt(p) for p in probs]
        norm = math.sqrt(sum(a * a for a in amps)) or 1.0
        amps = [a / norm for a in amps]
        return StatePreparation(amps)
