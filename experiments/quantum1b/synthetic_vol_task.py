"""Synthetic realized-volatility-*like* proxy series generator for Track C
(QRC vs ESN fair comparison, `EXP-C1`).

**This module produces SYNTHETIC DATA ONLY.** It is a GARCH(1,1)-style
stochastic process plus small i.i.d. observation noise, chosen because it
produces a strictly-positive, volatility-clustered series that is
*structurally* shaped like a realized-volatility target -- nothing more.
It is never fit to, seeded from, replayed from, or validated against any
real market series. Nothing in this module reads, touches, or references
any file under `research-corpus/model-lab` or any other real OHLCV source.

Why synthetic-only: Model Lab's own data-readiness gate
(`research-corpus/model-lab/work/readiness/READINESS_BRIEF.md`, D4 ruling)
records an operator-accepted HOLD on activating any questionable-license
free-tier data source. There is currently no daily OHLCV for the AMC's six
underlyings (SPY, QQQ, TLT, IEF, GLD, USO) held anywhere in the estate
with confirmed research rights. `QUANTUM1B_PREREGISTRATION.md`'s EXP-C1
anticipates exactly this and calls for a clearly-labeled synthetic
validation task with real-financial-task execution explicitly BLOCKED --
see `experiments.quantum1b.qrc_esn_harness.RealFinancialExecutionBlocked`.

Every array and variable name in this module carries the
`synthetic_realized_vol_proxy` label deliberately, so nothing downstream
can be mistaken for, or silently relabeled as, real market evidence.

Process (fixed, documented parameterization -- a standard illustrative
GARCH(1,1), not tuned to or derived from any dataset):

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}   (variance recursion)
    eps_t    = sigma_t * z_t,                  z_t ~ N(0, 1) i.i.d.
    proxy_t  = max(sigma_t * |z_t| + noise_t, floor_eps),  noise_t ~ N(0, obs_noise_std)

`proxy_t` is the *observed* series (`synthetic_realized_vol_proxy`) that
the forecasting harness sees; `sigma2_t` (the latent variance path) is
exposed only for diagnostics, never as a harness input (a real forecaster
would not have access to the latent variance either).

Default parameters:
    omega=0.05, alpha=0.10, beta=0.85 (alpha+beta=0.95 < 1 for
    stationarity; a standard, moderately-persistent textbook GARCH(1,1)
    parameterization -- not fit to any real series),
    sigma2_0 = omega / (1 - alpha - beta)  (unconditional variance, the
    natural deterministic warm start),
    obs_noise_std=0.02, floor_eps=1e-6, burn_in=200 (discarded so the
    returned series starts from the stationary regime, not the
    deterministic warm start).

Fully deterministic given `seed`: uses a private `numpy.random.Generator`
(`np.random.default_rng(seed)`), never global RNG state, so
`generate_synthetic_realized_vol_proxy(seed=k)` is byte-identical across
calls and processes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

GARCH_OMEGA = 0.05
GARCH_ALPHA = 0.10
GARCH_BETA = 0.85
OBS_NOISE_STD = 0.02
FLOOR_EPS = 1e-6
BURN_IN = 200


@dataclass
class SyntheticVolSeries:
    """Container for one `synthetic_realized_vol_proxy` draw. Every field
    here is synthetic data (see module docstring) -- never real market
    data, never derived from any real market data."""

    proxy: np.ndarray  # the observed synthetic_realized_vol_proxy series (harness input/target)
    sigma2: np.ndarray  # latent synthetic GARCH(1,1) variance path (diagnostics only, not a harness input)
    seed: int
    params: dict[str, Any] = field(default_factory=dict)


def generate_synthetic_realized_vol_proxy(
    n: int = 2000,
    seed: int = 0,
    omega: float = GARCH_OMEGA,
    alpha: float = GARCH_ALPHA,
    beta: float = GARCH_BETA,
    obs_noise_std: float = OBS_NOISE_STD,
    floor_eps: float = FLOOR_EPS,
    burn_in: int = BURN_IN,
) -> SyntheticVolSeries:
    """Generate one deterministic draw of the synthetic realized-vol proxy.

    Args:
        n: number of observations to return (post burn-in).
        seed: RNG seed; fully determines the draw.
        omega, alpha, beta: GARCH(1,1) recursion parameters (see module
            docstring). Must satisfy alpha + beta < 1 (stationarity).
        obs_noise_std: std of the additive observation-noise term.
        floor_eps: minimum value the returned proxy can take (keeps it
            strictly positive, required for QLIKE to be well-defined).
        burn_in: number of initial steps simulated and discarded so the
            returned series starts in the stationary regime.

    Returns:
        SyntheticVolSeries with `.proxy` of length `n`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (alpha + beta < 1.0):
        raise ValueError(f"alpha+beta={alpha + beta} must be < 1 for GARCH(1,1) stationarity")
    if omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")

    rng = np.random.default_rng(seed)
    total = n + burn_in
    sigma2 = np.empty(total, dtype=np.float64)
    eps = np.empty(total, dtype=np.float64)
    proxy = np.empty(total, dtype=np.float64)

    sigma2[0] = omega / (1.0 - alpha - beta)
    z0 = rng.normal()
    eps[0] = np.sqrt(sigma2[0]) * z0
    proxy[0] = max(np.sqrt(sigma2[0]) * abs(z0) + rng.normal(0.0, obs_noise_std), floor_eps)

    for t in range(1, total):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        z = rng.normal()
        eps[t] = np.sqrt(sigma2[t]) * z
        proxy[t] = max(np.sqrt(sigma2[t]) * abs(z) + rng.normal(0.0, obs_noise_std), floor_eps)

    proxy = proxy[burn_in:]
    sigma2 = sigma2[burn_in:]

    params = dict(
        omega=omega, alpha=alpha, beta=beta, obs_noise_std=obs_noise_std,
        floor_eps=floor_eps, burn_in=burn_in, n=n,
        process="synthetic_garch11_realized_vol_proxy",
    )
    return SyntheticVolSeries(proxy=proxy, sigma2=sigma2, seed=seed, params=params)


__all__ = [
    "SyntheticVolSeries",
    "generate_synthetic_realized_vol_proxy",
    "GARCH_OMEGA",
    "GARCH_ALPHA",
    "GARCH_BETA",
    "OBS_NOISE_STD",
    "FLOOR_EPS",
    "BURN_IN",
]
