"""Pluggable initialization-strategy interface, registered in
core.registry.ALGORITHM_REGISTRY so algorithms/qaoa_qubo/ can select a
strategy by name from ExperimentConfig.params rather than importing a
specific function.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from algorithms_extended.qaoa_initialization.interp_strategy import (
    interpolate_to_next_depth,
    random_initialization,
)
from algorithms_extended.qaoa_initialization.linear_ramp import linear_ramp_initialization
from core.registry import Registry

InitFn = Callable[..., tuple[np.ndarray, np.ndarray]]

INIT_STRATEGY_REGISTRY: Registry = Registry("qaoa_initialization_strategy")


@INIT_STRATEGY_REGISTRY.register("linear_ramp")
def _linear_ramp(p: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    return linear_ramp_initialization(p, **kwargs)


@INIT_STRATEGY_REGISTRY.register("random")
def _random(p: int, seed: int | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    return random_initialization(p, seed=seed)


@INIT_STRATEGY_REGISTRY.register("interp_from_previous")
def _interp(gammas: np.ndarray, betas: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Note the different call signature (needs a previous-depth solution,
    not just p) — callers must check which strategy they're invoking; this
    is intentional since INTERP is only meaningful as a depth-p -> depth-p+1
    step, not a from-scratch initializer."""
    return interpolate_to_next_depth(gammas, betas)
