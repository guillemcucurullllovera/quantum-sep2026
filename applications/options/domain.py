"""Typed domain models for the options/derivatives/hedging application layer.

`OptionSpec` is the single payoff-agnostic description every pricer in this
package (classical Black-Scholes/Monte Carlo, the QAE/MLAE call-spread
and digital-call pricers, and the qiskit-finance-extensions barrier/
basket pricers) is dispatched from -- see `applications.options.pricing_methods`.

Validation follows the codebase's existing pattern (`applications.portfolio.
domain.PortfolioSpec.validate_shape`): pydantic handles field-level shape/type
checks; `validate_feasible()` is called explicitly by the dispatcher for
economic feasibility checks (e.g. a barrier on the wrong side of spot) so the
raised exception is always `core.errors.ValidationError`, not pydantic's own
`pydantic.ValidationError` wrapped around a `model_validator`.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from core.errors import ValidationError


class PayoffKind(str, Enum):
    EUROPEAN_CALL = "european_call"
    EUROPEAN_PUT = "european_put"
    DIGITAL_CALL = "digital_call"
    CALL_SPREAD = "call_spread"
    UP_AND_OUT_CALL = "up_and_out_call"
    DOWN_AND_OUT_CALL = "down_and_out_call"
    BASKET_CALL = "basket_call"


BARRIER_KINDS = (PayoffKind.UP_AND_OUT_CALL, PayoffKind.DOWN_AND_OUT_CALL)


class OptionSpec(BaseModel):
    """A payoff-agnostic option/derivative specification.

    Field naming intentionally generic (`spot`, `strike`, `vol`, `rate`,
    `maturity`) rather than the `S0`/`K`/`sigma`/`r`/`T` vocabulary the
    underlying algorithm modules use -- `pricing_methods.py` is the one place
    that translates between the two, so every pricer call site in this
    package speaks the same names.
    """

    payoff: PayoffKind
    spot: float = Field(gt=0, description="S0 of the primary/only underlying")
    strike: float = Field(gt=0, description="K (or K1 for a call spread's lower strike)")
    rate: float = 0.04
    vol: float = Field(gt=0, default=0.20, description="sigma of the primary underlying")
    maturity: float = Field(gt=0, default=0.25, description="T, in years")

    # digital
    payout: float = Field(default=1.0, gt=0)

    # call spread
    strike2: float | None = Field(default=None, gt=0, description="K2, must be > strike")

    # barrier (up_and_out_call / down_and_out_call)
    barrier: float | None = Field(default=None, gt=0)

    # basket (second asset)
    spot2: float | None = Field(default=None, gt=0)
    vol2: float | None = Field(default=None, gt=0)
    weight1: float = 0.5
    weight2: float = 0.5
    correlation: float | None = Field(default=None, ge=-1.0, le=1.0)

    def validate_feasible(self) -> "OptionSpec":
        """Economic-feasibility checks the dispatcher must run before pricing
        anything. Raises `core.errors.ValidationError` (never a bare
        `ValueError`/pydantic error) so callers can catch one exception type
        across the whole platform. Returns self for chaining."""
        if self.payoff in BARRIER_KINDS:
            if self.barrier is None:
                raise ValidationError(f"{self.payoff.value} requires a `barrier` level")
            if self.payoff is PayoffKind.UP_AND_OUT_CALL and self.barrier <= self.spot:
                raise ValidationError(
                    f"up_and_out_call requires barrier ({self.barrier}) above spot "
                    f"({self.spot}) -- an up barrier at or below spot would already be "
                    "knocked out at inception, which is not a priceable option."
                )
            if self.payoff is PayoffKind.DOWN_AND_OUT_CALL and self.barrier >= self.spot:
                raise ValidationError(
                    f"down_and_out_call requires barrier ({self.barrier}) below spot "
                    f"({self.spot}) -- a down barrier at or above spot would already be "
                    "knocked out at inception, which is not a priceable option."
                )

        if self.payoff is PayoffKind.CALL_SPREAD:
            if self.strike2 is None:
                raise ValidationError("call_spread requires `strike2` (K2)")
            if self.strike2 <= self.strike:
                raise ValidationError(
                    f"call_spread requires strike2 ({self.strike2}) > strike ({self.strike}) "
                    "(K1 < K2 for a bull call spread)"
                )

        if self.payoff is PayoffKind.BASKET_CALL:
            missing = [
                name for name, val in (
                    ("spot2", self.spot2), ("vol2", self.vol2), ("correlation", self.correlation),
                ) if val is None
            ]
            if missing:
                raise ValidationError(f"basket_call requires {missing} to be set")
            if abs(self.weight1 + self.weight2 - 1.0) > 1e-6:
                raise ValidationError(
                    f"basket_call weights must sum to 1.0, got weight1={self.weight1} "
                    f"weight2={self.weight2} (sum={self.weight1 + self.weight2})"
                )

        if self.maturity <= 0:
            raise ValidationError(f"maturity must be > 0, got {self.maturity}")
        return self

    # -- convenience translators into the underlying algorithm modules' own
    # parameter vocabularies, kept here (not duplicated per call site) --

    def barrier_kind_str(self) -> str:
        if self.payoff not in BARRIER_KINDS:
            raise ValidationError(f"{self.payoff.value} is not a barrier payoff")
        return "up_and_out" if self.payoff is PayoffKind.UP_AND_OUT_CALL else "down_and_out"


__all__ = ["OptionSpec", "PayoffKind", "BARRIER_KINDS"]
