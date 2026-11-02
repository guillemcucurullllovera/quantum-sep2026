"""
Volatility-based feature computations for quantum signal generation.

Computes realized volatility from returns, detects volatility regime transitions,
and normalizes Average True Range relative to price levels.
"""

import math


def compute_realized_vol(
    returns: list[float], annualization_factor: float = 252
) -> float:
    """
    Compute realized (historical) volatility from returns.

    Calculates the standard deviation of returns and annualizes it using
    the provided annualization factor (typically 252 for daily data).

    Args:
        returns: List of return values (e.g., daily log returns)
        annualization_factor: Scaling factor for annualization (default: 252 for daily)

    Returns:
        Annualized realized volatility as float >= 0.
        Returns 0.0 if less than 2 data points.

    Examples:
        >>> compute_realized_vol([0.01, -0.01, 0.02])
        0.025
        >>> compute_realized_vol([0.01])
        0.0
    """
    if len(returns) < 2:
        return 0.0

    # Calculate mean
    mean_return = sum(returns) / len(returns)

    # Calculate variance
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

    # Standard deviation
    std_dev = math.sqrt(variance)

    # Annualize
    return std_dev * math.sqrt(annualization_factor)


def compute_vol_regime(
    realized_vol: float,
    vol_history: list[float],
    low_pct: float = 20,
    high_pct: float = 80,
) -> float:
    """
    Determine volatility regime based on percentile rank of current vol vs history.

    Classifies current volatility as LOW (0.0), NORMAL (0.5), or HIGH (1.0)
    based on its percentile position within historical volatility range.

    Args:
        realized_vol: Current realized volatility value
        vol_history: Historical volatility values for percentile calculation
        low_pct: Percentile threshold for LOW regime (default: 20)
        high_pct: Percentile threshold for HIGH regime (default: 80)

    Returns:
        Regime indicator: 0.0 (LOW), 0.5 (NORMAL), or 1.0 (HIGH).
        Returns 0.5 if history has fewer than 20 points.

    Examples:
        >>> compute_vol_regime(0.05, [0.01, 0.02, 0.03, 0.04] * 6)
        0.5
        >>> compute_vol_regime(0.001, list(range(50)))
        0.0
    """
    if len(vol_history) < 20:
        return 0.5

    # Sort history for percentile calculation
    sorted_vol = sorted(vol_history)

    # Calculate percentile position of current vol
    count_below = sum(1 for v in vol_history if v < realized_vol)
    percentile = (count_below / len(vol_history)) * 100.0

    # Classify regime
    if percentile <= low_pct:
        return 0.0
    elif percentile >= high_pct:
        return 1.0
    else:
        return 0.5


def compute_atr_normalized(atr: float, price: float) -> float:
    """
    Compute ATR (Average True Range) normalized by price level.

    Expresses ATR as a percentage of price, providing a scale-independent
    measure of volatility useful for cross-asset comparison.

    Args:
        atr: Average True Range value (non-negative)
        price: Current price level (must be > 0)

    Returns:
        Normalized ATR (ATR/price) in range [0, infinity).
        Returns 0.0 if price <= 0.

    Examples:
        >>> compute_atr_normalized(10.0, 100.0)
        0.1
        >>> compute_atr_normalized(5.0, 50.0)
        0.1
    """
    if price <= 0:
        return 0.0

    return atr / price
