"""
Derived feature modules for quantum signal generation.

Exposes core feature computation functions across multiple categories:
- Momentum: Composite momentum indicators and acceleration
- Volatility: Realized volatility, regime detection, and normalized ATR
- Macro: Event risk scoring and yield curve analysis
- Sentiment: Net sentiment and momentum calculation
- Positioning: COT positioning and extremeness detection
- Cross-asset: DXY momentum and risk sentiment ranking
"""

from .momentum import compute_momentum_composite, compute_momentum_acceleration
from .volatility import (
    compute_realized_vol,
    compute_vol_regime,
    compute_atr_normalized,
)
from .macro import compute_event_risk_score, compute_yield_curve_slope
from .sentiment import compute_net_sentiment, compute_sentiment_momentum
from .positioning import compute_cot_positioning, compute_cot_extreme
from .cross_asset import compute_dxy_momentum, compute_risk_sentiment

__all__ = [
    # Momentum
    "compute_momentum_composite",
    "compute_momentum_acceleration",
    # Volatility
    "compute_realized_vol",
    "compute_vol_regime",
    "compute_atr_normalized",
    # Macro
    "compute_event_risk_score",
    "compute_yield_curve_slope",
    # Sentiment
    "compute_net_sentiment",
    "compute_sentiment_momentum",
    # Positioning
    "compute_cot_positioning",
    "compute_cot_extreme",
    # Cross-asset
    "compute_dxy_momentum",
    "compute_risk_sentiment",
]
