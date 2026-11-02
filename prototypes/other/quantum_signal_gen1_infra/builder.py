"""
Quantum Input Vector Builder
============================
Main orchestrator that transforms raw agent signals into a validated
QuantumInputVector. This is the primary public API of the package.

Usage:
    from prototypes.other.quantum_signal_gen1_infra.builder import build_quantum_vector

    vector = build_quantum_vector(
        technical={"rsi": 65.2, "macd_cross": 0.002, ...},
        macro={"carry_differential": 1.35, "hours_to_next_event": 4.5, ...},
        news={"base_bull_count": 12, "base_bear_count": 5, ...},
        fundamental={"base_cot_net_pct": 15.3, "quote_cot_net_pct": -8.2, ...},
        risk_sentiment={"vix_level": 22.5, ...},
    )

    print(vector.to_json())
    print(vector.to_array())  # 14 floats for qubit encoding
"""

from datetime import datetime, timezone
from typing import Dict, Optional, List

from prototypes.other.quantum_signal_gen1_infra.models import QuantumInputVector
from prototypes.other.quantum_signal_gen1_infra.normalizer import Normalizer, RollingStats
from prototypes.other.quantum_signal_gen1_infra.features.momentum import (
    compute_momentum_composite,
    compute_momentum_acceleration,
)
from prototypes.other.quantum_signal_gen1_infra.features.volatility import (
    compute_realized_vol,
    compute_vol_regime,
    compute_atr_normalized,
)
from prototypes.other.quantum_signal_gen1_infra.features.macro import (
    compute_event_risk_score,
    compute_yield_curve_slope,
)
from prototypes.other.quantum_signal_gen1_infra.features.sentiment import (
    compute_net_sentiment,
    compute_sentiment_momentum,
)
from prototypes.other.quantum_signal_gen1_infra.features.positioning import (
    compute_cot_positioning,
    compute_cot_extreme,
)
from prototypes.other.quantum_signal_gen1_infra.features.cross_asset import (
    compute_dxy_momentum,
    compute_risk_sentiment,
)


class QuantumVectorBuilder:
    """
    Stateful builder that maintains rolling statistics for proper
    normalization across sequential calls.

    For one-shot use, call the module-level `build_quantum_vector()` function.
    For streaming use (processing sequential signals), instantiate this class
    and call `build()` repeatedly — it maintains normalization state.
    """

    def __init__(self, window_size: int = 252):
        """
        Args:
            window_size: Rolling window for normalization statistics.
                         Default 252 (one trading year).
        """
        self._window = window_size

        # Rolling stats for features that need historical context
        self._vol_history = RollingStats(max_size=window_size)
        self._vix_history = RollingStats(max_size=window_size)
        self._cot_history = RollingStats(max_size=window_size)
        self._carry_history = RollingStats(max_size=window_size)
        self._dxy_history = RollingStats(max_size=window_size)

        # State for sequential features
        self._prev_momentum: Optional[float] = None
        self._prev_sentiment: Optional[float] = None
        self._prev_sentiment_3d: List[float] = []  # last 3 sentiment values

    def build(
        self,
        technical: Dict,
        macro: Dict,
        news: Dict,
        fundamental: Dict,
        risk_sentiment: Optional[Dict] = None,
        commodity: Optional[Dict] = None,
        price_returns_20d: Optional[List[float]] = None,
        dxy_prices_5d: Optional[List[float]] = None,
        asset: str = "UNKNOWN",
        timestamp: Optional[str] = None,
    ) -> QuantumInputVector:
        """
        Build a complete QuantumInputVector from raw agent outputs.

        Args:
            technical: Technical agent signal dict (rsi, macd_cross, bb_position_ratio, atr, price)
            macro: Macro agent signal dict (carry_differential, hours_to_next_event, etc.)
            news: News agent signal dict (base_bull_count, base_bear_count, etc.)
            fundamental: Fundamental agent signal dict (base_cot_net_pct, quote_cot_net_pct, etc.)
            risk_sentiment: Risk sentiment signal dict (vix_level, vix_change). Optional.
            commodity: Commodity signal dict (oil_price, copper_price). Optional.
            price_returns_20d: List of 20 daily log returns for realized vol. Optional.
            dxy_prices_5d: List of 5 daily DXY prices. Optional.
            asset: Asset identifier (e.g., "EURUSD").
            timestamp: ISO 8601 timestamp. Defaults to now.

        Returns:
            Validated QuantumInputVector.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        risk_sentiment = risk_sentiment or {}
        commodity = commodity or {}

        # --- MOMENTUM BLOCK ---
        momentum = compute_momentum_composite(
            rsi=technical.get("rsi", 50.0),
            macd_cross=technical.get("macd_cross", 0.0),
            bb_ratio=technical.get("bb_position_ratio", 0.5),
        )
        momentum_accel = compute_momentum_acceleration(
            current=momentum,
            previous=self._prev_momentum,
        )
        self._prev_momentum = momentum

        # --- VOLATILITY BLOCK ---
        if price_returns_20d and len(price_returns_20d) >= 2:
            realized_vol_raw = compute_realized_vol(price_returns_20d)
        else:
            # Fallback: approximate from ATR
            price = technical.get("price", 1.0)
            atr = technical.get("atr", 0.0)
            realized_vol_raw = (atr / max(price, 1e-8)) if price > 0 else 0.0

        self._vol_history.add(realized_vol_raw)
        realized_vol = Normalizer.percentile_rank(
            realized_vol_raw, list(self._vol_history.history)
        )

        vol_regime = compute_vol_regime(
            realized_vol_raw, list(self._vol_history.history)
        )

        atr_norm = compute_atr_normalized(
            atr=technical.get("atr", 0.0),
            price=technical.get("price", 1.0),
        )
        # Normalize ATR to [0, 1] — typical range 0.001 to 0.05
        atr_norm = Normalizer.clip_and_scale(
            atr_norm, input_range=(0.0, 0.05), output_range=(0.0, 1.0)
        )

        # --- MACRO BLOCK ---
        carry_raw = macro.get("carry_differential", 0.0)
        self._carry_history.add(carry_raw)
        carry = Normalizer.standard_scale(
            carry_raw,
            mean=self._carry_history.mean(),
            std=self._carry_history.std(),
            clip_range=(-1.0, 1.0),
        )

        # Yield curve: available from macro equity signals or external
        yield_10y = macro.get("us10y_yield", None)
        yield_2y = macro.get("us2y_yield", None)
        yield_slope_raw = compute_yield_curve_slope(yield_10y, yield_2y)
        if yield_slope_raw is not None:
            yield_slope = Normalizer.clip_and_scale(
                yield_slope_raw, input_range=(-3.0, 3.0), output_range=(-1.0, 1.0)
            )
        else:
            yield_slope = 0.0  # Neutral default when unavailable

        event_risk = compute_event_risk_score(
            hours_to_event=macro.get("hours_to_next_event"),
            event_impact=macro.get("next_event_impact", "HIGH"),
        )

        # --- SENTIMENT BLOCK ---
        # Handle both FX and equity schemas
        if "base_bull_count" in news:
            net_sent = compute_net_sentiment(
                base_bull=news.get("base_bull_count", 0),
                base_bear=news.get("base_bear_count", 0),
                base_total=news.get("base_article_count", 1),
                quote_bull=news.get("quote_bull_count", 0),
                quote_bear=news.get("quote_bear_count", 0),
                quote_total=news.get("quote_article_count", 1),
            )
        elif "focus_bull_count" in news:
            # Equity schema
            net_sent = compute_net_sentiment(
                base_bull=news.get("focus_bull_count", 0),
                base_bear=news.get("focus_bear_count", 0),
                base_total=news.get("focus_article_count", 1),
                quote_bull=news.get("market_bull_count", 0),
                quote_bear=news.get("market_bear_count", 0),
                quote_total=news.get("market_article_count", 1),
            )
        else:
            net_sent = 0.0

        # Track sentiment for momentum calculation
        self._prev_sentiment_3d.append(net_sent)
        if len(self._prev_sentiment_3d) > 4:
            self._prev_sentiment_3d.pop(0)

        sent_3d_ago = self._prev_sentiment_3d[0] if len(self._prev_sentiment_3d) >= 4 else None
        sent_momentum = compute_sentiment_momentum(net_sent, sent_3d_ago)

        # --- POSITIONING BLOCK ---
        cot_pos = compute_cot_positioning(
            base_cot_pct=fundamental.get("base_cot_net_pct"),
            quote_cot_pct=fundamental.get("quote_cot_net_pct"),
        )
        # Normalize COT positioning to [-1, 1]
        cot_pos_norm = Normalizer.clip_and_scale(
            cot_pos, input_range=(-100.0, 100.0), output_range=(-1.0, 1.0)
        )

        cot_net = fundamental.get("base_cot_net", 0)
        if cot_net is not None:
            self._cot_history.add(float(cot_net))
        cot_ext = compute_cot_extreme(
            current_net=float(cot_net) if cot_net is not None else 0.0,
            cot_history=list(self._cot_history.history),
        )

        # --- CROSS-ASSET BLOCK ---
        if dxy_prices_5d and len(dxy_prices_5d) >= 2:
            dxy_mom = compute_dxy_momentum(dxy_prices_5d)
        else:
            dxy_mom = 0.0
        # Normalize DXY momentum to [-1, 1]
        dxy_mom_norm = Normalizer.clip_and_scale(
            dxy_mom, input_range=(-0.05, 0.05), output_range=(-1.0, 1.0)
        )

        vix = risk_sentiment.get("vix_level", 20.0)
        if vix is not None:
            self._vix_history.add(vix)
        risk_sent = compute_risk_sentiment(
            vix_level=vix if vix is not None else 20.0,
            vix_history=list(self._vix_history.history),
        )

        # --- ASSEMBLE VECTOR ---
        vector = QuantumInputVector(
            momentum_composite=momentum,
            momentum_acceleration=momentum_accel,
            realized_vol_20d=realized_vol,
            vol_regime=vol_regime,
            atr_normalized=atr_norm,
            carry_differential=carry,
            yield_curve_slope=yield_slope,
            event_risk_score=event_risk,
            net_sentiment=net_sent,
            sentiment_momentum=sent_momentum,
            cot_positioning=cot_pos_norm,
            cot_extreme=cot_ext,
            dxy_momentum=dxy_mom_norm,
            risk_sentiment=risk_sent,
            asset=asset,
            timestamp=timestamp,
        )

        return vector

    def reset(self):
        """Clear all rolling state. Use when switching assets or contexts."""
        self._vol_history = RollingStats(max_size=self._window)
        self._vix_history = RollingStats(max_size=self._window)
        self._cot_history = RollingStats(max_size=self._window)
        self._carry_history = RollingStats(max_size=self._window)
        self._dxy_history = RollingStats(max_size=self._window)
        self._prev_momentum = None
        self._prev_sentiment = None
        self._prev_sentiment_3d = []


# Module-level convenience function (stateless, single-shot)
_default_builder = QuantumVectorBuilder()


def build_quantum_vector(
    technical: Dict,
    macro: Dict,
    news: Dict,
    fundamental: Dict,
    risk_sentiment: Optional[Dict] = None,
    commodity: Optional[Dict] = None,
    price_returns_20d: Optional[List[float]] = None,
    dxy_prices_5d: Optional[List[float]] = None,
    asset: str = "UNKNOWN",
    timestamp: Optional[str] = None,
) -> QuantumInputVector:
    """
    Build a QuantumInputVector from raw agent signals (stateless convenience function).

    For streaming use with proper normalization, instantiate QuantumVectorBuilder directly.

    See QuantumVectorBuilder.build() for parameter documentation.
    """
    return _default_builder.build(
        technical=technical,
        macro=macro,
        news=news,
        fundamental=fundamental,
        risk_sentiment=risk_sentiment,
        commodity=commodity,
        price_returns_20d=price_returns_20d,
        dxy_prices_5d=dxy_prices_5d,
        asset=asset,
        timestamp=timestamp,
    )
