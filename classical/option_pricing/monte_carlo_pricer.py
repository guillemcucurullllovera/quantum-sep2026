"""Monte Carlo European option pricer, built on classical.monte_carlo's GBM
engine — used as an independent cross-check against black_scholes_price, and
as the classical comparator for QAE-based option pricing.
"""
from __future__ import annotations

import numpy as np

from classical.monte_carlo.gbm import gbm_terminal_prices


def monte_carlo_european_price(
    *,
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 200_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return (price, standard_error) for a European option under
    risk-neutral GBM dynamics (drift = r).
    """
    terminal = gbm_terminal_prices(s0=s0, mu=r, sigma=sigma, t=t, n_paths=n_paths, seed=seed)
    if option_type == "call":
        payoffs = np.maximum(terminal - k, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(k - terminal, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    discounted = np.exp(-r * t) * payoffs
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
    return price, stderr
