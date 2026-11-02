"""Echo State Network — classical reservoir-computing baseline.

The fair classical control for QRC, decoupled from any platform-specific
validation harness (trial registries, an import-firewall CI rule) since
none of that is quantum-specific. This implements the actual
ESN algorithm (standard Jaeger 2001 / Lukoseviciute 2012 formulation: a
random sparse reservoir rescaled to a target spectral radius, leaky-integrator
state update, ridge-regression readout as the only trained layer), rebuilt as
a standalone class with the same call shape as
`algorithms.qrc.reservoir.QuantumReservoir` so benchmark code can swap one
for the other without special-casing either.

References:
    Jaeger, H. (2001). "The echo state approach to analysing and training
    recurrent neural networks."
    Lukosevicius, M. (2012). "A practical guide to applying echo state
    networks."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors import ValidationError
from core.registry import ALGORITHM_REGISTRY


@dataclass
class ESNConfig:
    reservoir_size: int = 100
    spectral_radius: float = 0.9   # must be < 1 for the echo-state property
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    ridge_lambda: float = 1e-3
    density: float = 0.1           # fraction of non-zero reservoir connections
    seed: int = 42

    def validate(self) -> None:
        if self.reservoir_size < 1:
            raise ValidationError(f"reservoir_size={self.reservoir_size} must be >= 1")
        if not (0.0 < self.spectral_radius < 1.0):
            raise ValidationError(f"spectral_radius={self.spectral_radius} must be in (0,1)")
        if not (0.0 < self.leak_rate <= 1.0):
            raise ValidationError(f"leak_rate={self.leak_rate} must be in (0,1]")
        if self.input_scaling <= 0.0:
            raise ValidationError(f"input_scaling={self.input_scaling} must be > 0")
        if not (0.0 < self.density <= 1.0):
            raise ValidationError(f"density={self.density} must be in (0,1]")


class EchoStateNetwork:
    """Standard leaky-integrator ESN with a ridge-regression readout.

    State update:  x_t = (1-a) x_{t-1} + a * tanh(W_in @ [1; u_t] + W @ x_{t-1})
    Readout:       y_t = W_out @ [1; x_t]   (the only trained layer)
    """

    def __init__(self, config: ESNConfig | None = None) -> None:
        self.config = config or ESNConfig()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.seed)
        self.W: np.ndarray | None = None
        self.W_in: np.ndarray | None = None
        self.W_out: np.ndarray | None = None
        self._built = False

    def build(self, n_inputs: int) -> "EchoStateNetwork":
        cfg = self.config
        N = cfg.reservoir_size

        # sparse random reservoir, rescaled to target spectral radius
        W = self._rng.standard_normal((N, N))
        mask = self._rng.random((N, N)) < cfg.density
        W = W * mask
        eigvals = np.linalg.eigvals(W)
        radius = np.max(np.abs(eigvals))
        if radius < 1e-12:
            raise ValidationError("Reservoir matrix has near-zero spectral radius; increase density")
        self.W = W / radius * cfg.spectral_radius

        self.W_in = cfg.input_scaling * self._rng.uniform(-1, 1, size=(N, n_inputs + 1))
        self._built = True
        return self

    def _require_built(self) -> None:
        if not self._built:
            raise ValidationError("Call .build(n_inputs) before compute_states()")

    def compute_states(self, U: np.ndarray) -> np.ndarray:
        """U: (T, n_inputs) -> (T, reservoir_size) reservoir state trajectory.

        This is the unsupervised nonlinear feature map — safe to compute once
        and reuse across walk-forward folds (only the readout is refit per
        fold), matching the fair-comparison discipline QRC's own walk-forward
        harness follows.
        """
        self._require_built()
        cfg = self.config
        T = U.shape[0]
        N = cfg.reservoir_size
        states = np.zeros((T, N))
        x = np.zeros(N)
        for t in range(T):
            u_bias = np.concatenate([[1.0], U[t]])
            pre = self.W_in @ u_bias + self.W @ x
            x = (1 - cfg.leak_rate) * x + cfg.leak_rate * np.tanh(pre)
            states[t] = x
        return states

    def fit_readout(self, states: np.ndarray, y: np.ndarray, washout: int = 0) -> "EchoStateNetwork":
        S = states[washout:]
        target = y[washout:]
        S_bias = np.hstack([np.ones((S.shape[0], 1)), S])
        A = S_bias.T @ S_bias + self.config.ridge_lambda * np.eye(S_bias.shape[1])
        b = S_bias.T @ target
        self.W_out = np.linalg.solve(A, b)
        return self

    def predict(self, states: np.ndarray) -> np.ndarray:
        if self.W_out is None:
            raise ValidationError("Call .fit_readout() before predict()")
        S_bias = np.hstack([np.ones((states.shape[0], 1)), states])
        return S_bias @ self.W_out


@ALGORITHM_REGISTRY.register("esn_baseline")
class ESNBaselineAlgorithm:
    config_cls = ESNConfig
    model_cls = EchoStateNetwork


__all__ = ["ESNConfig", "EchoStateNetwork", "ESNBaselineAlgorithm"]
