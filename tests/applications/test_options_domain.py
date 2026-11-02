"""Domain-model validation: infeasible option specs must raise
core.errors.ValidationError, not silently produce a nonsense price."""
from __future__ import annotations

import pytest

from applications.options.domain import OptionSpec, PayoffKind
from core.errors import ValidationError


def test_up_and_out_requires_barrier_above_spot():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=90.0)
    with pytest.raises(ValidationError, match="above spot"):
        spec.validate_feasible()


def test_down_and_out_requires_barrier_below_spot():
    spec = OptionSpec(payoff=PayoffKind.DOWN_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=110.0)
    with pytest.raises(ValidationError, match="below spot"):
        spec.validate_feasible()


def test_barrier_payoff_without_barrier_level_rejected():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0)
    with pytest.raises(ValidationError, match="requires a `barrier`"):
        spec.validate_feasible()


def test_valid_barrier_direction_passes():
    up = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0)
    up.validate_feasible()  # must not raise
    down = OptionSpec(payoff=PayoffKind.DOWN_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=85.0)
    down.validate_feasible()  # must not raise


def test_call_spread_requires_strike2_above_strike():
    spec = OptionSpec(payoff=PayoffKind.CALL_SPREAD, spot=100.0, strike=100.0, strike2=90.0)
    with pytest.raises(ValidationError, match="K1 < K2"):
        spec.validate_feasible()


def test_call_spread_without_strike2_rejected():
    spec = OptionSpec(payoff=PayoffKind.CALL_SPREAD, spot=100.0, strike=95.0)
    with pytest.raises(ValidationError, match="requires `strike2`"):
        spec.validate_feasible()


def test_basket_call_requires_second_asset_fields():
    spec = OptionSpec(payoff=PayoffKind.BASKET_CALL, spot=100.0, strike=100.0)
    with pytest.raises(ValidationError, match="requires"):
        spec.validate_feasible()


def test_basket_call_weights_must_sum_to_one():
    spec = OptionSpec(
        payoff=PayoffKind.BASKET_CALL, spot=100.0, strike=100.0, spot2=100.0, vol2=0.25,
        correlation=0.3, weight1=0.7, weight2=0.7,
    )
    with pytest.raises(ValidationError, match="sum to 1.0"):
        spec.validate_feasible()


def test_pydantic_rejects_nonpositive_spot():
    with pytest.raises(Exception):  # pydantic.ValidationError, field-level
        OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=-1.0, strike=100.0)
