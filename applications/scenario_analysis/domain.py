"""Typed domain models for scenario analysis: parametrized shocks applied to
a base return/P&L distribution (or multi-asset returns matrix), composed
into named scenarios and grids, applicable across any downstream risk
pipeline -- tail-risk, options, portfolio -- rather than being specific to
one of them.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ShockKind(str, Enum):
    VOL_SPIKE = "vol_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    MEAN_SHIFT = "mean_shift"
    JUMP = "jump"


class ShockSpec(BaseModel):
    """One parametrized shock. `params` is shock-kind-specific:
      VOL_SPIKE:             {"multiplier": float}
      MEAN_SHIFT:            {"shift": float}
      JUMP:                  {"jump_prob": float, "jump_size": float, "seed": int | None}
      CORRELATION_BREAKDOWN: {"target_corr": float, "seed": int | None}  (2-D input only)
    """

    kind: ShockKind
    params: dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.kind.value}({parts})" if parts else self.kind.value


class ScenarioSpec(BaseModel):
    """A named scenario: a sequence of shocks applied in order to a base
    series. An empty `shocks` list is the legal identity/baseline scenario.
    """

    name: str
    shocks: list[ShockSpec] = Field(default_factory=list)

    def describe(self) -> str:
        if not self.shocks:
            return f"{self.name} (baseline, no shocks)"
        return f"{self.name}: " + " + ".join(s.describe() for s in self.shocks)
