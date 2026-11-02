"""Closed-form Black-Scholes European option pricing — the ground-truth
reference every QAE-based option-pricing amplitude-estimation implementation
in algorithms_extended/ must be validated against.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_price(
    *, s0: float, k: float, t: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    if t <= 0:
        return float(max(s0 - k, 0.0) if option_type == "call" else max(k - s0, 0.0))

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == "call":
        price = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        price = k * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return float(price)
