"""European, digital, barrier, and basket option pricing applications,
composing algorithms.qae_iqae + algorithms_extended.{qae_options_qiskit_finance,
qiskit_finance_extensions} with classical (Black-Scholes / Monte Carlo)
baselines for validation on every call.

This module adds a payoff-agnostic dispatch layer on top of these payoff-specific
pricer classes -- `domain.OptionSpec` + `pricing_methods.price_option` is now
the recommended entry point for new callers (scenario grids, resource
estimation, hedging, the CLI all go through it); the classes below remain
for direct use / backward compatibility."""
from applications.options.barrier_pricer import BarrierOptionPricer, BasketOptionPricer
from applications.options.domain import OptionSpec, PayoffKind
from applications.options.option_pricer import DigitalOptionPricer, EuropeanOptionPricer
from applications.options.pricing_methods import PriceResult, PricingMethod, price_option

__all__ = [
    "EuropeanOptionPricer", "DigitalOptionPricer",
    "BarrierOptionPricer", "BasketOptionPricer",
    "OptionSpec", "PayoffKind", "PricingMethod", "PriceResult", "price_option",
]
