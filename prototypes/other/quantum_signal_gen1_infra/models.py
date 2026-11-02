"""
Data models for quantum signal processing.

Defines the QuantumInputVector dataclass and related structures for representing
market signals in a quantum-compatible format.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any


@dataclass
class QuantumInputVector:
    """
    A 14-dimensional quantum signal vector combining technical, macro, and sentiment indicators.

    Each field is constrained to specific ranges for proper quantum state preparation.
    All ranges are normalized to support downstream quantum circuit operations.

    Features (by qubit index):
    - 0: momentum_composite [-1, 1]
    - 1: momentum_acceleration [-1, 1]
    - 2: realized_vol_20d [0, 1]
    - 3: vol_regime {0, 0.5, 1.0}
    - 4: atr_normalized [0, 1]
    - 5: carry_differential [-1, 1]
    - 6: yield_curve_slope [-1, 1]
    - 7: event_risk_score [0, 1]
    - 8: net_sentiment [-1, 1]
    - 9: sentiment_momentum [-1, 1]
    - 10: cot_positioning [-1, 1]
    - 11: cot_extreme [-1, 1]
    - 12: dxy_momentum [-1, 1]
    - 13: risk_sentiment [0, 1]
    """

    # Core features (14 fields)
    momentum_composite: float
    momentum_acceleration: float
    realized_vol_20d: float
    vol_regime: float
    atr_normalized: float
    carry_differential: float
    yield_curve_slope: float
    event_risk_score: float
    net_sentiment: float
    sentiment_momentum: float
    cot_positioning: float
    cot_extreme: float
    dxy_momentum: float
    risk_sentiment: float

    # Metadata
    asset: str
    timestamp: str
    version: str = "1.0"

    def validate(self) -> bool:
        """
        Validate that all features are within their expected ranges.

        Returns:
            bool: True if all features pass range validation.

        Raises:
            ValueError: If any feature is out of range.
        """
        constraints = {
            "momentum_composite": (-1, 1),
            "momentum_acceleration": (-1, 1),
            "realized_vol_20d": (0, 1),
            "vol_regime": None,  # Special handling: discrete values
            "atr_normalized": (0, 1),
            "carry_differential": (-1, 1),
            "yield_curve_slope": (-1, 1),
            "event_risk_score": (0, 1),
            "net_sentiment": (-1, 1),
            "sentiment_momentum": (-1, 1),
            "cot_positioning": (-1, 1),
            "cot_extreme": (-1, 1),
            "dxy_momentum": (-1, 1),
            "risk_sentiment": (0, 1),
        }

        errors = []

        for field_name, bounds in constraints.items():
            value = getattr(self, field_name)

            if field_name == "vol_regime":
                # vol_regime must be one of {0, 0.5, 1.0}
                if value not in (0, 0.5, 1.0):
                    errors.append(
                        f"{field_name} must be 0, 0.5, or 1.0, got {value}"
                    )
            else:
                min_val, max_val = bounds
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"{field_name} out of range [{min_val}, {max_val}], got {value}"
                    )

        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(errors))

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary representation.

        Returns:
            dict: All fields as key-value pairs.
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Serialize to JSON string.

        Returns:
            str: JSON representation of the vector.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_array(self) -> List[float]:
        """
        Extract the 14 feature values as an ordered list.

        Order matches qubit indices:
        [momentum_composite, momentum_acceleration, realized_vol_20d, vol_regime,
         atr_normalized, carry_differential, yield_curve_slope, event_risk_score,
         net_sentiment, sentiment_momentum, cot_positioning, cot_extreme,
         dxy_momentum, risk_sentiment]

        Returns:
            list[float]: The 14 feature values in qubit order.
        """
        return [
            self.momentum_composite,
            self.momentum_acceleration,
            self.realized_vol_20d,
            self.vol_regime,
            self.atr_normalized,
            self.carry_differential,
            self.yield_curve_slope,
            self.event_risk_score,
            self.net_sentiment,
            self.sentiment_momentum,
            self.cot_positioning,
            self.cot_extreme,
            self.dxy_momentum,
            self.risk_sentiment,
        ]

    def to_reduced_8q(self) -> List[float]:
        """
        Extract 8 selected features for reduced-qubit circuits.

        v2.0 — Data-driven selection (Fix 2, April 2026):
          Replaced 3 dead/redundant features:
            vol_regime (3 unique values, r=0.86 with realized_vol) → momentum_acceleration
            carry_differential (0.09 std, -0.02 label corr)       → atr_normalized
            cot_positioning (3 unique values)                      → sentiment_momentum

        Selected indices: 0, 2, 1, 4, 7, 8, 9, 13
        Features: [momentum_composite, realized_vol_20d, momentum_acceleration,
                   atr_normalized, event_risk_score, net_sentiment,
                   sentiment_momentum, risk_sentiment]

        Returns:
            list[float]: 8 selected feature values.
        """
        features = self.to_array()
        return [features[i] for i in [0, 2, 1, 4, 7, 8, 9, 13]]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantumInputVector":
        """
        Construct a QuantumInputVector from a dictionary.

        Args:
            d: Dictionary with required keys for all fields.

        Returns:
            QuantumInputVector: Validated instance.

        Raises:
            ValueError: If required fields are missing.
            TypeError: If field types cannot be coerced to float/str.
        """
        required_fields = [
            "momentum_composite", "momentum_acceleration", "realized_vol_20d",
            "vol_regime", "atr_normalized", "carry_differential",
            "yield_curve_slope", "event_risk_score", "net_sentiment",
            "sentiment_momentum", "cot_positioning", "cot_extreme",
            "dxy_momentum", "risk_sentiment", "asset", "timestamp"
        ]

        missing = [f for f in required_fields if f not in d]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        try:
            vector = cls(
                momentum_composite=float(d["momentum_composite"]),
                momentum_acceleration=float(d["momentum_acceleration"]),
                realized_vol_20d=float(d["realized_vol_20d"]),
                vol_regime=float(d["vol_regime"]),
                atr_normalized=float(d["atr_normalized"]),
                carry_differential=float(d["carry_differential"]),
                yield_curve_slope=float(d["yield_curve_slope"]),
                event_risk_score=float(d["event_risk_score"]),
                net_sentiment=float(d["net_sentiment"]),
                sentiment_momentum=float(d["sentiment_momentum"]),
                cot_positioning=float(d["cot_positioning"]),
                cot_extreme=float(d["cot_extreme"]),
                dxy_momentum=float(d["dxy_momentum"]),
                risk_sentiment=float(d["risk_sentiment"]),
                asset=str(d["asset"]),
                timestamp=str(d["timestamp"]),
                version=str(d.get("version", "1.0")),
            )
            vector.validate()
            return vector
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot coerce dictionary values to QuantumInputVector: {e}")
