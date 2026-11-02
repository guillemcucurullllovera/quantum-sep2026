"""
Cross-asset feature computations for quantum signal generation.

Computes macroeconomic cross-asset indicators including DXY momentum
and risk sentiment based on VIX levels.
"""


def compute_dxy_momentum(dxy_prices_5d: list[float]) -> float:
    """
    Compute 5-day DXY (Dollar Index) momentum as percentage return.

    Measures the recent directional movement in USD strength. Positive values
    indicate USD strengthening, negative indicates weakening.

    Args:
        dxy_prices_5d: List of DXY prices over 5 days (oldest to newest)

    Returns:
        5-day percentage return clipped to reasonable bounds.
        Returns 0.0 if fewer than 2 prices provided.

    Examples:
        >>> compute_dxy_momentum([100.0, 101.0])
        0.01
        >>> compute_dxy_momentum([100.0, 99.0])
        -0.01
        >>> compute_dxy_momentum([100.0])
        0.0
    """
    if len(dxy_prices_5d) < 2:
        return 0.0

    first_price = dxy_prices_5d[0]
    last_price = dxy_prices_5d[-1]

    # Avoid division by zero
    if first_price == 0.0:
        return 0.0

    # Calculate 5-day percentage return
    return (last_price - first_price) / first_price


def compute_risk_sentiment(vix_level: float, vix_history: list[float]) -> float:
    """
    Compute risk sentiment as percentile rank of current VIX vs historical data.

    Higher VIX indicates elevated market fear/volatility, lower VIX indicates
    complacency. Percentile rank shows whether current VIX is elevated relative
    to recent history.

    Args:
        vix_level: Current VIX level (typically 10-80 range)
        vix_history: Historical VIX values for percentile calculation

    Returns:
        Risk sentiment in range [0, 1].
        0.0 = VIX at historical lows (risk-on sentiment)
        1.0 = VIX at historical highs (risk-off sentiment)
        Returns 0.5 if history too short (< 20 points).

    Examples:
        >>> compute_risk_sentiment(15.0, [10, 12, 14, 16, 18, 20] * 5)
        0.333
        >>> compute_risk_sentiment(30.0, [10, 12, 14, 16, 18, 20] * 5)
        1.0
    """
    if len(vix_history) < 20:
        return 0.5

    # Calculate percentile position of current VIX
    count_below = sum(1 for v in vix_history if v < vix_level)
    percentile = count_below / len(vix_history)

    return max(0.0, min(1.0, percentile))
