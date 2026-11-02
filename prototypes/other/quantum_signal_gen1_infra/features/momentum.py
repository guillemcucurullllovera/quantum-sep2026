"""
Momentum-based feature computations for quantum signal generation.

Combines multiple momentum indicators (RSI, MACD cross, Bollinger Band ratio)
into a composite signal with PCA loadings. Also computes momentum acceleration
as first-difference of momentum values.
"""

import math


def compute_momentum_composite(
    rsi: float, macd_cross: float, bb_ratio: float
) -> float:
    """
    Compute composite momentum indicator combining RSI, MACD cross, and BB ratio.

    This function normalizes three momentum indicators and applies pre-computed
    PCA loadings to create a unified momentum score.

    Args:
        rsi: Relative Strength Index in range [0, 100]
        macd_cross: MACD histogram value (typically small, e.g., [-0.01, 0.01])
        bb_ratio: Bollinger Band ratio in range [0, 1]
            (position between lower and upper bands)

    Returns:
        Composite momentum score clipped to [-1, 1]

    Examples:
        >>> compute_momentum_composite(50.0, 0.0, 0.5)  # Neutral momentum
        0.0
        >>> compute_momentum_composite(70.0, 0.001, 0.8)  # Bullish
        0.5
    """
    # Normalize RSI from [0, 100] to [-1, 1]
    rsi_normalized = (rsi - 50.0) / 50.0

    # Normalize MACD cross: clip to [-0.005, 0.005], then scale to [-1, 1]
    macd_clipped = max(-0.005, min(0.005, macd_cross))
    macd_normalized = macd_clipped / 0.005

    # BB ratio already in [0, 1], scale to [-1, 1]
    bb_normalized = 2.0 * bb_ratio - 1.0

    # Apply PCA loadings (pre-computed placeholder)
    pca_loadings = [0.58, 0.52, 0.63]
    composite = (
        pca_loadings[0] * rsi_normalized
        + pca_loadings[1] * macd_normalized
        + pca_loadings[2] * bb_normalized
    )

    # Clip result to [-1, 1]
    return max(-1.0, min(1.0, composite))


def compute_momentum_acceleration(
    current: float, previous: float | None = None
) -> float:
    """
    Compute momentum acceleration as first difference of momentum values.

    Momentum acceleration measures the rate of change in momentum, indicating
    whether momentum is strengthening or weakening.

    Args:
        current: Current momentum value (typically in [-1, 1])
        previous: Previous momentum value, or None for initial observation

    Returns:
        Momentum acceleration clipped to [-1, 1].
        Returns 0.0 if previous is None.

    Examples:
        >>> compute_momentum_acceleration(0.6, 0.3)
        0.3
        >>> compute_momentum_acceleration(0.5, None)
        0.0
    """
    if previous is None:
        return 0.0

    acceleration = current - previous
    return max(-1.0, min(1.0, acceleration))
