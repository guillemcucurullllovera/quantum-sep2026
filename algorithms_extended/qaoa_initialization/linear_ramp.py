"""Linear-ramp (adiabatic-inspired) QAOA parameter initialization.

Standard QAOA-literature heuristic (see THIRD_PARTY_NOTICES.md): a discretized
approximation of an adiabatic schedule, with gamma ramping up and beta
ramping down across the circuit depth.
"""
from __future__ import annotations

import numpy as np

from core.errors import ValidationError


def linear_ramp_initialization(
    p: int, gamma_max: float = np.pi / 2, beta_max: float = np.pi / 4
) -> tuple[np.ndarray, np.ndarray]:
    """gamma linearly increases from ~0 to gamma_max across p layers; beta
    linearly decreases from beta_max to ~0 — mirroring a discretized
    adiabatic anneal from the mixer Hamiltonian toward the cost Hamiltonian.
    """
    if p < 1:
        raise ValidationError("p must be >= 1")
    layer_index = np.arange(1, p + 1)
    gammas = gamma_max * layer_index / p
    betas = beta_max * (1 - layer_index / p)
    return gammas, betas
