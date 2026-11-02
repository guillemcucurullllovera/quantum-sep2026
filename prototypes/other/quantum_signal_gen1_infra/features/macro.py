"""
Macro-based feature computations for quantum signal generation.

Computes event risk scoring based on time-to-event and impact classification,
and calculates yield curve slope from 2-year and 10-year yields.
"""


def compute_event_risk_score(
    hours_to_event: float | None, event_impact: str = "HIGH"
) -> float:
    """
    Compute event risk score based on proximity and impact classification.

    Uses a continuous decay function (1/(1+hours)) weighted by impact category.
    Risk decays as hours increase, and is zeroed out beyond 48 hours.

    Args:
        hours_to_event: Hours until the event, or None for no upcoming event
        event_impact: Event impact level - "HIGH" (1.0), "MEDIUM" (0.5), "LOW" (0.2)
            (default: "HIGH")

    Returns:
        Event risk score in range [0, 1], clipped to [0, 1].
        Returns 0.0 if hours_to_event is None or > 48.

    Examples:
        >>> compute_event_risk_score(12.0, "HIGH")
        0.0769
        >>> compute_event_risk_score(12.0, "LOW")
        0.0154
        >>> compute_event_risk_score(60.0, "HIGH")
        0.0
    """
    if hours_to_event is None or hours_to_event > 48:
        return 0.0

    # Map impact to weight
    impact_weights = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.2}
    impact_weight = impact_weights.get(event_impact, 1.0)

    # Compute risk: 1/(1+hours) * impact_weight
    risk = (1.0 / (1.0 + hours_to_event)) * impact_weight

    return max(0.0, min(1.0, risk))


def compute_yield_curve_slope(
    yield_10y: float | None, yield_2y: float | None
) -> float | None:
    """
    Compute yield curve slope as difference between 10-year and 2-year yields.

    Simple spread calculation used as a macro indicator. Positive slope indicates
    normal upward-sloping curve (economic expansion), negative indicates inversion.

    Args:
        yield_10y: 10-year yield as decimal (e.g., 0.03 for 3%)
        yield_2y: 2-year yield as decimal (e.g., 0.02 for 2%)

    Returns:
        Yield curve slope (10y - 2y) as float.
        Returns None if either input is None.

    Examples:
        >>> compute_yield_curve_slope(0.03, 0.02)
        0.01
        >>> compute_yield_curve_slope(0.02, 0.025)
        -0.005
        >>> compute_yield_curve_slope(0.03, None)
        None
    """
    if yield_10y is None or yield_2y is None:
        return None

    return yield_10y - yield_2y
