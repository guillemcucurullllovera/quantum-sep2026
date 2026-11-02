"""Finite-difference Greeks (delta, vega) for any `OptionSpec`, computed by
bumping the underlying `pricing_methods.price_option` call -- works
identically whether the pricer underneath is closed-form Black-Scholes,
Monte Carlo, or a quantum amplitude estimator, since it only ever touches the
normalized `PriceResult.price` field.

For the classical Black-Scholes payoffs (european_call/put), the finite-
difference numbers are cross-checked against the textbook closed-form
formulas (`analytic_bs_delta`/`analytic_bs_vega`) as a correctness check on
the bump-and-reprice machinery itself before trusting it on a payoff that has
no closed form (barrier/basket), where finite-difference is the *only* way
to get a sensitivity number.

Honesty note: bumping a quantum-estimated price re-runs the pricer with a
fixed seed/shots at S0+-h and sigma+-h; the resulting "quantum Greek" carries
both discretisation error (grid resolution) and, if `use_noisy`/shot-sampled,
estimation noise, on top of the finite-difference truncation error itself --
this is not attempting a parameter-shift-rule-style exact quantum gradient,
which the payoff circuits in this codebase are not built for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from applications.options.domain import OptionSpec
from applications.options.pricing_methods import PriceResult, PricingMethod, price_option


def analytic_bs_delta(spec: OptionSpec) -> float:
    """Closed-form Black-Scholes delta -- N(d1) for a call, N(d1)-1 for a put."""
    d1 = (math.log(spec.spot / spec.strike) + (spec.rate + 0.5 * spec.vol**2) * spec.maturity) / (
        spec.vol * math.sqrt(spec.maturity)
    )
    n_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return n_d1 if spec.payoff.value == "european_call" else n_d1 - 1.0


def analytic_bs_vega(spec: OptionSpec) -> float:
    """Closed-form Black-Scholes vega (same for call and put): S0*phi(d1)*sqrt(T)."""
    d1 = (math.log(spec.spot / spec.strike) + (spec.rate + 0.5 * spec.vol**2) * spec.maturity) / (
        spec.vol * math.sqrt(spec.maturity)
    )
    phi_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return spec.spot * phi_d1 * math.sqrt(spec.maturity)


@dataclass(frozen=True)
class GreeksResult:
    delta: float
    vega: float
    price_at_spec: float
    bump_fraction: float
    method: PricingMethod
    price_up_spot: float
    price_down_spot: float
    price_up_vol: float
    price_down_vol: float
    analytic_delta: float | None = None
    analytic_vega: float | None = None


def finite_difference_greeks(
    spec: OptionSpec,
    *,
    method: PricingMethod = PricingMethod.AUTO,
    bump_fraction: float = 0.01,
    shots: int = 4096,
    seed: int = 42,
    n_qubits: int | None = None,
) -> GreeksResult:
    """Central finite-difference delta (d price / d spot) and vega (d price /
    d vol), by bumping `spec.spot`/`spec.vol` by `bump_fraction` and calling
    `price_option` at each bumped spec with the SAME method/shots/seed so the
    only thing that varies between the two prices is the bumped parameter
    (holding quantum estimation noise as similar as fixed-seed allows).
    """
    spec = spec.validate_feasible()
    h_spot = spec.spot * bump_fraction
    h_vol = spec.vol * bump_fraction

    def _price(bumped: OptionSpec) -> float:
        return price_option(
            bumped, method, shots=shots, seed=seed, n_qubits=n_qubits,
        ).price

    base_result: PriceResult = price_option(spec, method, shots=shots, seed=seed, n_qubits=n_qubits)

    price_up_spot = _price(spec.model_copy(update={"spot": spec.spot + h_spot}))
    price_down_spot = _price(spec.model_copy(update={"spot": spec.spot - h_spot}))
    delta = (price_up_spot - price_down_spot) / (2.0 * h_spot)

    price_up_vol = _price(spec.model_copy(update={"vol": spec.vol + h_vol}))
    price_down_vol = _price(spec.model_copy(update={"vol": spec.vol - h_vol}))
    vega = (price_up_vol - price_down_vol) / (2.0 * h_vol)

    analytic_delta = analytic_vega = None
    if spec.payoff.value in ("european_call", "european_put"):
        analytic_delta = analytic_bs_delta(spec)
        analytic_vega = analytic_bs_vega(spec)

    return GreeksResult(
        delta=delta, vega=vega, price_at_spec=base_result.price, bump_fraction=bump_fraction,
        method=base_result.method, price_up_spot=price_up_spot, price_down_spot=price_down_spot,
        price_up_vol=price_up_vol, price_down_vol=price_down_vol,
        analytic_delta=analytic_delta, analytic_vega=analytic_vega,
    )


__all__ = ["GreeksResult", "finite_difference_greeks", "analytic_bs_delta", "analytic_bs_vega"]
