"""Gradient-based optimizers for FinancialQNN, operating on the flat
(theta, W, b) parameter tuple. Two families, for the honest optimizer
comparison this platform's convention calls for: Adam (the genuinely new
capability -- v1-v5's hand-rolled trainers never had access to exact
end-to-end gradients, only the ridge-closed-form + gradient-free outer loop)
and COBYLA (reused via scipy, as a gradient-free baseline for comparison
against Adam on the exact same QNN loss surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import optimize as sp_optimize


@dataclass
class AdamState:
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    t: int = 0


class Adam:
    """Standard Adam (Kingma & Ba 2015) -- a general published optimization
    algorithm, implemented from its formula, not adapted from any library."""

    def __init__(self, lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.state = AdamState()

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.state.t += 1
        t = self.state.t
        out = {}
        for key, p in params.items():
            g = grads[key]
            m = self.state.m.get(key, np.zeros_like(p))
            v = self.state.v.get(key, np.zeros_like(p))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g**2)
            self.state.m[key], self.state.v[key] = m, v
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            out[key] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return out


def cobyla_minimize(
    loss_fn: Callable[[np.ndarray], float], x0: np.ndarray, max_iter: int = 200
) -> tuple[np.ndarray, list[float]]:
    """Gradient-free baseline optimizer over a flattened parameter vector,
    for direct comparison against Adam on the identical QNN loss surface."""
    history: list[float] = []

    def tracked(x: np.ndarray) -> float:
        loss = loss_fn(x)
        history.append(loss)
        return loss

    result = sp_optimize.minimize(tracked, x0, method="COBYLA", options={"maxiter": max_iter})
    return result.x, history
