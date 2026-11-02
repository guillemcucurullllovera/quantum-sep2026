"""Classical Monte Carlo / closed-form option pricing, re-exported here so
the quantum-vs-classical generations sit side by side under
``applications/options/``. Implementation lives in ``classical.option_pricing``.
"""
from classical.option_pricing.black_scholes import black_scholes_price
from classical.option_pricing.monte_carlo_pricer import monte_carlo_european_price

__all__ = ["black_scholes_price", "monte_carlo_european_price"]
