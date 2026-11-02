"""QAOA parameter initialization strategies (INTERP, linear-ramp, random
baseline). See THIRD_PARTY_NOTICES.md for what was reviewed and why nothing was
adapted directly from QAOAKit or the End-to-End-Protocol repo."""
from algorithms_extended.qaoa_initialization.interp_strategy import (
    interpolate_to_next_depth,
    random_initialization,
)
from algorithms_extended.qaoa_initialization.linear_ramp import linear_ramp_initialization
from algorithms_extended.qaoa_initialization.strategies import INIT_STRATEGY_REGISTRY

__all__ = [
    "interpolate_to_next_depth",
    "random_initialization",
    "linear_ramp_initialization",
    "INIT_STRATEGY_REGISTRY",
]
