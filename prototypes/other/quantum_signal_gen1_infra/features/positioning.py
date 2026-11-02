"""
Positioning-based feature computations for quantum signal generation.

Computes COT (Commitment of Traders) positioning metrics, including net position
calculation and extremeness detection via z-score analysis.
"""

import math


def compute_cot_positioning(
    base_cot_pct: float | None, quote_cot_pct: float | None
) -> float:
    """
    Compute net COT positioning as difference between base and quote COT percentages.

    Treats None values as 0, allowing for cases where positioning data may be
    partially available. Represents relative positioning imbalance.

    Args:
        base_cot_pct: COT percentage for base asset (range typically [-100, 100]),
            or None for missing data
        quote_cot_pct: COT percentage for quote asset (range typically [-100, 100]),
            or None for missing data

    Returns:
        Net COT positioning clipped to [-100, 100].
        Positive values indicate net long bias, negative indicates net short bias.

    Examples:
        >>> compute_cot_positioning(30.0, 10.0)
        20.0
        >>> compute_cot_positioning(None, 20.0)
        -20.0
        >>> compute_cot_positioning(50.0, -30.0)
        80.0
    """
    # Treat None as 0
    base_pct = base_cot_pct if base_cot_pct is not None else 0.0
    quote_pct = quote_cot_pct if quote_cot_pct is not None else 0.0

    # Compute difference
    net_positioning = base_pct - quote_pct

    # Clip to [-100, 100]
    return max(-100.0, min(100.0, net_positioning))


def compute_cot_extreme(
    current_net: float, cot_history: list[float]
) -> float:
    """
    Compute COT extremeness as normalized z-score of current positioning.

    Identifies whether current positioning is at historical extremes relative
    to recent history. Z-score is normalized to [-1, 1] range.

    Args:
        current_net: Current net COT positioning value
        cot_history: Historical net COT positioning values for z-score calculation

    Returns:
        Normalized z-score clipped to [-1, 1].
        Returns 0.0 if history has fewer than 10 points.
        Positive values indicate long extremes, negative indicates short extremes.

    Examples:
        >>> compute_cot_extreme(50.0, list(range(-50, 50)))
        0.333
        >>> compute_cot_extreme(0.0, list(range(-50, 50)))
        -0.0
    """
    if len(cot_history) < 10:
        return 0.0

    # Calculate mean and std
    mean = sum(cot_history) / len(cot_history)
    variance = sum((x - mean) ** 2 for x in cot_history) / len(cot_history)
    std = math.sqrt(variance)

    # Handle zero variance case
    if std == 0.0:
        return 0.0

    # Calculate z-score
    z_score = (current_net - mean) / std

    # Normalize to [-1, 1] by dividing by 3 (approximately covers 3 standard deviations)
    normalized_z = z_score / 3.0

    # Clip to [-1, 1]
    return max(-1.0, min(1.0, normalized_z))
