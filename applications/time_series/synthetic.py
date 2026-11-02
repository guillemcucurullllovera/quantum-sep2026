"""Synthetic financial time-series generators, used for self-contained
testing of QRC/ESN/Mamba without requiring real market data.

Two processes are provided:
  - `generate_garch_like_series`: a single fixed-parameter GARCH(1,1)
    process (the original original generator).
  - `generate_regime_switching_garch_series`: a Markov-switching-variance
    process -- 3 GARCH(1,1) regimes ("calm", "turbulent", "crisis") with a
    persistent Markov chain selecting the active regime at each step. This
    is a materially harder, more realistic forecasting target than a single
    fixed-parameter process: real realized-volatility series visibly change
    character across calm and crisis periods, and a method that only ever
    sees one variance regime can look artificially good.

Both stand in for the paper's {RV, MKT, DP, IP, RV_q, STR, DEF} feature set
shape via the same lagged-feature construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.dataset_interfaces import FinancialDataset
from core.schemas import DatasetDescriptor


def generate_garch_like_series(
    n_steps: int, omega: float = 0.05, alpha: float = 0.15, beta: float = 0.75, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (returns, realized_volatility) of length n_steps from a
    GARCH(1,1) process: sigma2_t = omega + alpha*r_{t-1}^2 + beta*sigma2_{t-1}.
    alpha+beta < 1 for stationarity (0.15+0.75=0.9 here).
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n_steps)
    returns = np.empty(n_steps)
    sigma2[0] = omega / (1 - alpha - beta)
    returns[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n_steps):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    realized_vol = np.sqrt(sigma2)
    return returns, realized_vol


def _lagged_feature_matrix(returns: np.ndarray) -> np.ndarray:
    """[returns_lag1, |returns|_lag1, rolling_mean_5, rolling_std_5,
    momentum_10], each using only information up to t-1 (no look-ahead).
    Shared by every synthetic dataset in this module so the feature
    construction is identical regardless of which variance process
    generated `returns`.
    """
    n = len(returns)
    window = 5
    roll_mean = np.array([returns[max(0, t - window) : t].mean() if t > 0 else 0.0 for t in range(n)])
    roll_std = np.array([returns[max(0, t - window) : t].std() if t > 1 else 0.0 for t in range(n)])
    momentum = np.array([returns[max(0, t - 10) : t].sum() if t > 0 else 0.0 for t in range(n)])

    X = np.stack(
        [
            np.roll(returns, 1),
            np.roll(np.abs(returns), 1),
            np.roll(roll_mean, 1),
            np.roll(roll_std, 1),
            np.roll(momentum, 1),
        ],
        axis=1,
    )
    X[0] = 0.0
    return X


class SyntheticVolatilityDataset(FinancialDataset):
    """Feature set: [returns_lag1, |returns|_lag1, rolling_mean_5, rolling_std_5,
    momentum_10] -> target: next-step realized volatility. Shape mirrors the
    paper's multi-feature setup without claiming to be real market data.
    """

    def __init__(self, n_steps: int = 600, seed: int = 42) -> None:
        returns, rv = generate_garch_like_series(n_steps, seed=seed)
        X = _lagged_feature_matrix(returns)
        self._X = X[5:]  # drop warm-up rows before rolling windows are valid
        self._y = rv[5:]

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_garch_volatility",
            n_samples=len(self._y),
            n_features=self._X.shape[1],
            domain="realized_volatility",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


@dataclass
class RegimeParams:
    name: str
    omega: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not (0 <= self.alpha and 0 <= self.beta and self.alpha + self.beta < 1):
            raise ValueError(
                f"Regime {self.name!r}: alpha+beta must be < 1 for stationarity "
                f"(got alpha={self.alpha}, beta={self.beta})"
            )

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.alpha - self.beta)


def default_regimes() -> list[RegimeParams]:
    """Three regimes with the same persistence (alpha+beta=0.9) but
    increasingly severe unconditional variance: calm=0.1, turbulent=0.8,
    crisis=2.0 -- an 8x and 20x spread, chosen to be large enough that a
    correct implementation's per-regime realized-vol statistics are
    trivially, robustly distinguishable (see
    tests/applications/test_regime_switching_synthetic.py).
    """
    return [
        RegimeParams("calm", omega=0.01, alpha=0.05, beta=0.85),
        RegimeParams("turbulent", omega=0.08, alpha=0.20, beta=0.70),
        RegimeParams("crisis", omega=0.20, alpha=0.35, beta=0.55),
    ]


def _default_transition_matrix(k: int, p_stay: float = 0.97) -> np.ndarray:
    """Persistent Markov chain: regimes last ~1/(1-p_stay) steps on average
    (33 steps at p_stay=0.97), switching uniformly among the other regimes.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    p_switch = (1 - p_stay) / (k - 1) if k > 1 else 0.0
    P = np.full((k, k), p_switch)
    np.fill_diagonal(P, p_stay)
    return P


def generate_regime_switching_garch_series(
    n_steps: int,
    regimes: list[RegimeParams] | None = None,
    transition_matrix: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Markov-switching-variance ("regime-switching GARCH-like") process.

    At each step, the active regime (drawn from a persistent Markov chain --
    default self-transition probability 0.97, so regimes last ~30 steps on
    average) selects which (omega, alpha, beta) GARCH(1,1) triple drives
        sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}.
    `sigma2` itself carries over continuously across regime switches (only
    the coefficients change), so switches show up as visible level/variance
    shifts in realized volatility rather than resets to zero.

    Returns (returns, realized_vol, regime_labels), each length n_steps.
    `regime_labels` is int in [0, len(regimes)) and is provided for
    diagnostics/sanity-checking only -- it is never a model input (using it
    as one would leak the answer).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}")
    regimes = regimes or default_regimes()
    k = len(regimes)
    if transition_matrix is None:
        transition_matrix = _default_transition_matrix(k)
    if transition_matrix.shape != (k, k):
        raise ValueError(f"transition_matrix must be ({k}, {k}); got {transition_matrix.shape}")

    rng = np.random.default_rng(seed)
    regime_idx = np.empty(n_steps, dtype=int)
    regime_idx[0] = rng.integers(k)
    for t in range(1, n_steps):
        regime_idx[t] = rng.choice(k, p=transition_matrix[regime_idx[t - 1]])

    sigma2 = np.empty(n_steps)
    returns = np.empty(n_steps)
    r0 = regimes[regime_idx[0]]
    sigma2[0] = r0.unconditional_variance
    returns[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n_steps):
        r = regimes[regime_idx[t]]
        sigma2[t] = r.omega + r.alpha * returns[t - 1] ** 2 + r.beta * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    realized_vol = np.sqrt(sigma2)
    return returns, realized_vol, regime_idx


class RegimeSwitchingVolatilityDataset(FinancialDataset):
    """Same lagged-feature shape as SyntheticVolatilityDataset, but the
    target realized-volatility series comes from
    generate_regime_switching_garch_series -- a genuinely non-stationary,
    multi-regime target instead of one fixed-parameter process.

    `regime_labels` (aligned to `.targets()`, i.e. with the same 5-row
    warm-up dropped) is exposed for diagnostics only, never as a feature.
    """

    def __init__(
        self,
        n_steps: int = 800,
        regimes: list[RegimeParams] | None = None,
        transition_matrix: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        returns, rv, regime_idx = generate_regime_switching_garch_series(
            n_steps, regimes=regimes, transition_matrix=transition_matrix, seed=seed
        )
        X = _lagged_feature_matrix(returns)
        self._X = X[5:]
        self._y = rv[5:]
        self.regime_labels = regime_idx[5:]

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            name="synthetic_regime_switching_volatility",
            n_samples=len(self._y),
            n_features=self._X.shape[1],
            domain="realized_volatility",
        )

    def features(self) -> np.ndarray:
        return self._X

    def targets(self) -> np.ndarray:
        return self._y


__all__ = [
    "generate_garch_like_series",
    "SyntheticVolatilityDataset",
    "RegimeParams",
    "default_regimes",
    "generate_regime_switching_garch_series",
    "RegimeSwitchingVolatilityDataset",
]
