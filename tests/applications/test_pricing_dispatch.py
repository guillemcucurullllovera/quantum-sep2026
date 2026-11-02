"""Cross-method pricing dispatch: every supported (payoff, method) pair
should agree with a classical ground truth within a stated tolerance, and
every unsupported pair should raise ValidationError with a clear message."""
from __future__ import annotations

import pytest

from applications.options.domain import OptionSpec, PayoffKind
from applications.options.pricing_methods import PricingMethod, price_option
from core.errors import ValidationError


def test_black_scholes_call_matches_classical_reference_exactly():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.BLACK_SCHOLES)
    assert result.abs_error == pytest.approx(0.0, abs=1e-9)
    assert result.price > 0


def test_black_scholes_put_matches_classical_reference_exactly():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_PUT, spot=100.0, strike=105.0, rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.BLACK_SCHOLES)
    assert result.abs_error == pytest.approx(0.0, abs=1e-9)


def test_monte_carlo_european_call_agrees_with_black_scholes():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.MONTE_CARLO, mc_paths=200_000, seed=3)
    assert result.abs_error < 6 * result.stderr  # ~6-sigma MC tolerance, very unlikely to flake


def test_call_spread_black_scholes_and_monte_carlo_agree():
    spec = OptionSpec(payoff=PayoffKind.CALL_SPREAD, spot=100.0, strike=95.0, strike2=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    bs = price_option(spec, PricingMethod.BLACK_SCHOLES)
    mc = price_option(spec, PricingMethod.MONTE_CARLO, mc_paths=200_000, seed=5)
    assert abs(bs.price - mc.price) < 6 * mc.stderr


def test_call_spread_qae_mlae_tracks_black_scholes():
    spec = OptionSpec(payoff=PayoffKind.CALL_SPREAD, spot=100.0, strike=95.0, strike2=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.QAE_MLAE, shots=4096, seed=1)
    assert result.abs_error < 5.0  # coarse n=2 grid, same honest bound as test_options.py


def test_digital_call_qae_iqae_tracks_black_scholes():
    spec = OptionSpec(payoff=PayoffKind.DIGITAL_CALL, spot=100.0, strike=100.0, rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.QAE_IQAE, n_qubits=4, shots=2048)
    assert result.abs_error < 0.15  # matches test_options.py's honest coarse-grid bound


def test_barrier_up_and_out_qae_iqae_never_exceeds_vanilla_upper_bound():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.QAE_IQAE, n_qubits=4)
    assert result.raw["vanilla_upper_bound"] >= result.price - 1e-9


def test_barrier_terminal_only_parity_price_is_close_to_quantum_price():
    """The circuit and monte_carlo_terminal_only both discretize the barrier
    check to maturity only -- apples-to-apples, should be close (matching
    test_barrier_basket_options.py's existing tolerance)."""
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=120.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    result = price_option(spec, PricingMethod.QAE_IQAE, n_qubits=5)
    assert abs(result.price - result.raw["monte_carlo_terminal_only"]) < 0.5


def test_basket_call_qae_iqae_agrees_with_monte_carlo_within_grid_tolerance():
    spec = OptionSpec(payoff=PayoffKind.BASKET_CALL, spot=100.0, spot2=100.0, strike=100.0,
                       vol=0.2, vol2=0.25, correlation=0.3, rate=0.04, maturity=0.25)
    result = price_option(spec, PricingMethod.QAE_IQAE, n_qubits=2)
    # same 20% floor used and justified in applications/options/barrier_pricer.py's
    # BasketOptionPricer.price at n_a=n_b=2
    assert result.abs_error < 0.20 * result.classical_reference_price + 3 * (result.stderr or 0)


def test_vanilla_call_has_no_quantum_pricer_and_says_so():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0)
    with pytest.raises(ValidationError, match="does not support method"):
        price_option(spec, PricingMethod.QAE_IQAE)


def test_barrier_has_no_closed_form_and_says_so():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0)
    with pytest.raises(ValidationError, match="does not support method"):
        price_option(spec, PricingMethod.BLACK_SCHOLES)


def test_auto_method_selects_a_supported_method_for_every_payoff():
    specs = [
        OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0),
        OptionSpec(payoff=PayoffKind.DIGITAL_CALL, spot=100.0, strike=100.0),
        OptionSpec(payoff=PayoffKind.CALL_SPREAD, spot=100.0, strike=95.0, strike2=115.0),
        OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0),
        OptionSpec(payoff=PayoffKind.BASKET_CALL, spot=100.0, spot2=100.0, strike=100.0, vol2=0.25, correlation=0.2),
    ]
    for spec in specs:
        result = price_option(spec, PricingMethod.AUTO, shots=1024, n_qubits=3)
        assert result.method is not PricingMethod.AUTO
        assert result.price >= 0.0


def test_price_option_can_log_to_experiment_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_SEP2026_REGISTRY_DB", str(tmp_path / "registry.sqlite3"))
    from core.config import get_settings
    get_settings(refresh=True)

    from core.experiment_registry import ExperimentRegistry

    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0)
    before = ExperimentRegistry().count()
    price_option(spec, PricingMethod.BLACK_SCHOLES, log_to_registry=True)
    after = ExperimentRegistry().count()
    assert after == before + 1
