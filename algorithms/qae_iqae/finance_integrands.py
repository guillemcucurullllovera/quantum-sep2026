"""Finance integrands for the QAE numerical-integration encoding.

Two constructions, both Asymmetrica-original (see THIRD_PARTY_NOTICES.md):

* `build_call_spread_integrand` — a Black-Scholes European bull-call-spread expected
  payoff, folded onto the ancilla amplitude (MLAE demo).
* `build_crash_barrier` — the risk-neutral P(S_T <= barrier) crash probability, the
  Woerner-Egger VaR building block (IQAE demo, the platform's primary finance use case).

Every integrand exposes g(x) in [0,1] for x in [0,1], because the ancilla amplitude is
a = sin^2(theta/2) with theta = 2*arcsin(sqrt(g)). The QAE estimates a = mean_i g(x_i)
over the grid.

HONESTY NOTE: at n=2 the grid has 4 points; at n=3, 8. A coarse quadrature of a peaked
lognormal x kinked payoff is a *structural demonstration*, not an accurate price. Three
quantities are always reported separately and never conflated:
  1. exact_grid_a   - the amplitude a perfect QAE returns at this grid (isolates quantum error);
  2. price/prob-from-a - the coarse price/probability implied by an estimated a (isolates discretisation);
  3. true analytic value - the closed-form Black-Scholes target the grid approximates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from algorithms_extended.qae_spinqit.quadrature import Rule, grid_points
from algorithms_extended.qae_spinqit.toy_integrands import Integrand, _clip01


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a European call. The classical ground-truth baseline
    every QAE option-pricing result in this platform is validated against."""
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a European put, via put-call parity."""
    call = bs_call_price(S0, K, r, sigma, T)
    return call - S0 + K * math.exp(-r * T)


def bs_digital_call_price(S0: float, K: float, r: float, sigma: float, T: float, payout: float = 1.0) -> float:
    """Black-Scholes price of a cash-or-nothing digital call: payout * e^{-rT} * P(S_T > K)."""
    if T <= 0 or sigma <= 0:
        return payout * math.exp(-r * T) if S0 > K else 0.0
    d2 = (math.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return payout * math.exp(-r * T) * _norm_cdf(d2)


def bs_lognormal_pdf(S: float, S0: float, r: float, sigma: float, T: float) -> float:
    """Risk-neutral density of S_T at S (S_T lognormal under the BS measure)."""
    if S <= 0:
        return 0.0
    mu = math.log(S0) + (r - 0.5 * sigma ** 2) * T
    s = sigma * math.sqrt(T)
    z = (math.log(S) - mu) / s
    return math.exp(-0.5 * z * z) / (S * s * math.sqrt(2.0 * math.pi))


def bs_cdf_below(barrier: float, S0: float, r: float, sigma: float, T: float) -> float:
    """Risk-neutral P(S_T <= barrier) under Black-Scholes (lognormal S_T)."""
    if barrier <= 0:
        return 0.0
    d = (math.log(barrier / S0) - (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d)


def call_spread_price_from_amplitude(meta: Dict[str, object], a: float) -> float:
    """Map an estimated QAE amplitude back to the coarse grid-implied spread price:
    price ~ span * H * a (change of variables dS = span*dx; a ~ mean_i g_i)."""
    span = float(meta["span"])
    H = float(meta["norm_H"])
    return span * H * float(a)


def build_call_spread_integrand(
    S0: float = 100.0,
    K1: float = 95.0,
    K2: float = 115.0,
    r: float = 0.04,
    sigma: float = 0.20,
    T: float = 0.25,
    S_low: float = 80.0,
    S_high: float = 125.0,
    *,
    n: int = 2,
    rule: Rule = "midpoint",
) -> Integrand:
    """Build a normalised bull-call-spread expected-payoff integrand for a fixed grid."""
    span = S_high - S_low
    disc = math.exp(-r * T)
    cap = K2 - K1

    def h_of_x(x: float) -> float:
        S = S_low + span * x
        payoff = min(max(S - K1, 0.0), cap)
        return disc * bs_lognormal_pdf(S, S0, r, sigma, T) * payoff

    grid = grid_points(y=1.0, n=n, rule=rule)
    h_nodes = [h_of_x(x) for x in grid.points]
    H = max(h_nodes)
    if H <= 0:
        raise ValueError("degenerate integrand (all grid payoffs zero) - widen [S_low,S_high]")

    def g(x: float) -> float:
        return _clip01(h_of_x(x) / H)

    exact_grid_a = sum(g(x) for x in grid.points) / len(grid.points)
    true_bs_spread_price = bs_call_price(S0, K1, r, sigma, T) - bs_call_price(S0, K2, r, sigma, T)

    meta: Dict[str, object] = {
        "S0": S0, "K1": K1, "K2": K2, "r": r, "sigma": sigma, "T": T,
        "S_low": S_low, "S_high": S_high, "span": span, "cap": cap,
        "norm_H": H, "n": n, "rule": rule,
        "grid_S": [S_low + span * x for x in grid.points],
        "grid_g": [g(x) for x in grid.points],
        "exact_grid_a": exact_grid_a,
        "true_bs_spread_price": true_bs_spread_price,
        "grid_price_estimate": span * H * exact_grid_a,
        "note": f"n={n} ({2**n}-point) DEMO - structural pipeline test, not an accurate price.",
    }
    return Integrand(
        name="finance_call_spread", kind="finance",
        label=f"BS bull call spread K1={K1:g}/K2={K2:g} T={T:g}",
        g=g, exact_integral=lambda y: None, meta=meta,
    )


@dataclass(frozen=True)
class BarrierIntegrand:
    name: str
    kind: str
    label: str
    probs: tuple
    barrier_mask: tuple
    exact_grid_a: float
    exact_analytic: float
    meta: Dict[str, object] = field(default_factory=dict)


def build_crash_barrier(
    S0: float = 100.0,
    r: float = 0.04,
    sigma: float = 0.20,
    T: float = 0.25,
    crash_pct: float = 0.05,
    S_low: Optional[float] = None,
    S_high: Optional[float] = None,
    *,
    n: int = 3,
    sigmas: float = 3.5,
) -> BarrierIntegrand:
    """A downside/crash barrier: the risk-neutral probability the index is at or below
    barrier = S0*(1-crash_pct) at maturity — the CDF/comparator object Woerner-Egger's
    VaR construction uses (pure comparator, no rescaling)."""
    barrier = S0 * (1.0 - crash_pct)
    s = sigma * math.sqrt(T)
    drift = (r - 0.5 * sigma ** 2) * T
    if S_low is None:
        S_low = max(1e-9, S0 * math.exp(drift - sigmas * s))
    if S_high is None:
        S_high = S0 * math.exp(drift + sigmas * s)
    span = S_high - S_low
    m = 2 ** n
    S = [S_low + span * (j + 0.5) / m for j in range(m)]
    pdf = [bs_lognormal_pdf(sj, S0, r, sigma, T) for sj in S]
    tot = sum(pdf)
    if tot <= 0:
        raise ValueError("degenerate grid (zero total density) - widen the window")
    probs = tuple(p / tot for p in pdf)
    mask = tuple(sj <= barrier for sj in S)
    exact_grid_a = sum(p for p, mk in zip(probs, mask) if mk)
    exact_analytic = bs_cdf_below(barrier, S0, r, sigma, T)
    meta: Dict[str, object] = {
        "S0": S0, "r": r, "sigma": sigma, "T": T, "crash_pct": crash_pct, "barrier": barrier,
        "S_low": S_low, "S_high": S_high, "n": n, "grid_S": S, "grid_probs": list(probs),
        "n_below_barrier": sum(mask),
        "exact_grid_a": exact_grid_a, "exact_analytic": exact_analytic,
        "discretisation_gap": abs(exact_grid_a - exact_analytic),
        "note": f"n={n} ({2**n}-point) DEMO - coarse CDF, not an accurate probability; no advantage claim.",
    }
    return BarrierIntegrand(
        name="crash_barrier", kind="finance-barrier",
        label=f"P(S_T <= {barrier:.1f}) = P(>= {100*crash_pct:.0f}% drop), T={T:g}",
        probs=probs, barrier_mask=mask, exact_grid_a=exact_grid_a,
        exact_analytic=exact_analytic, meta=meta,
    )


def build_return_distribution_barrier(
    pnl_samples,
    loss_threshold: float,
    *,
    n: int = 3,
) -> BarrierIntegrand:
    """Build a BarrierIntegrand directly from an empirical P&L sample (rather than a
    closed-form lognormal), by histogramming into 2**n bins — the bridge used by
    applications.tail_risk to run QAE-based VaR/CVaR-adjacent probability estimation
    against an arbitrary return distribution, not just Black-Scholes."""
    import numpy as np

    pnl = np.asarray(pnl_samples, dtype=float)
    m = 2 ** n
    lo, hi = float(pnl.min()), float(pnl.max())
    if hi <= lo:
        raise ValueError("pnl_samples has zero range")
    counts, edges = np.histogram(pnl, bins=m, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        raise ValueError("no samples fell in histogram range")
    probs = tuple(float(c) / float(total) for c in counts)
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(m)]
    mask = tuple(c <= loss_threshold for c in centers)
    exact_grid_a = sum(p for p, mk in zip(probs, mask) if mk)
    empirical_prob = float(np.mean(pnl <= loss_threshold))
    meta = {
        "loss_threshold": loss_threshold, "n": n, "grid_centers": centers,
        "grid_probs": list(probs), "n_samples": int(total),
        "exact_grid_a": exact_grid_a, "empirical_prob": empirical_prob,
        "discretisation_gap": abs(exact_grid_a - empirical_prob),
        "note": f"empirical n={n} ({m}-bin) histogram DEMO of P&L <= {loss_threshold}.",
    }
    return BarrierIntegrand(
        name="return_distribution_barrier", kind="finance-barrier",
        label=f"P(PnL <= {loss_threshold:g})",
        probs=probs, barrier_mask=mask, exact_grid_a=exact_grid_a,
        exact_analytic=empirical_prob, meta=meta,
    )
