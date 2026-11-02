"""Finite-difference Greeks must agree with the closed-form Black-Scholes
formulas where one exists (validating the bump-and-reprice machinery
itself), and must produce a finite, sane sensitivity for a payoff that has
no closed form (barrier), where finite-difference is the only option."""
from __future__ import annotations

import pytest

from applications.options.domain import OptionSpec, PayoffKind
from applications.options.greeks import finite_difference_greeks
from applications.options.pricing_methods import PricingMethod


def test_finite_difference_delta_matches_analytic_black_scholes_call():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    g = finite_difference_greeks(spec, method=PricingMethod.BLACK_SCHOLES)
    assert g.analytic_delta is not None
    assert g.delta == pytest.approx(g.analytic_delta, abs=1e-3)


def test_finite_difference_vega_matches_analytic_black_scholes_call():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    g = finite_difference_greeks(spec, method=PricingMethod.BLACK_SCHOLES)
    assert g.analytic_vega is not None
    assert g.vega == pytest.approx(g.analytic_vega, abs=1e-2)


def test_finite_difference_delta_matches_analytic_black_scholes_put():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_PUT, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    g = finite_difference_greeks(spec, method=PricingMethod.BLACK_SCHOLES)
    assert -1.0 <= g.analytic_delta <= 0.0
    assert g.delta == pytest.approx(g.analytic_delta, abs=1e-3)


def test_call_delta_is_between_zero_and_one():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    g = finite_difference_greeks(spec, method=PricingMethod.MONTE_CARLO, seed=1)
    assert -0.05 <= g.delta <= 1.05  # small MC-noise slack around the [0,1] theoretical bound


def test_barrier_finite_difference_delta_is_finite_and_reasonable():
    """No closed-form barrier Greek exists -- finite difference against the
    QAE_IQAE pricer is the only way to get a sensitivity number, so just
    check it comes back finite and within a sane order of magnitude (a
    barrier call's delta should not exceed a vanilla call's, roughly)."""
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    g = finite_difference_greeks(spec, method=PricingMethod.QAE_IQAE, n_qubits=4)
    assert g.analytic_delta is None  # no closed form for a barrier -- must not fabricate one
    assert abs(g.delta) < 5.0  # sane order of magnitude, not a finite-difference blow-up
