"""Maximum-likelihood amplitude estimation (MLAE, QAE-without-QPE) + honest error bars.

Model: p_k(a) = sin^2((2k+1)*arcsin(sqrt(a))), m_k ~ Binomial(N_k, p_k(a)).
Estimator: a_hat = argmax_a sum_k [ m_k*log p_k + (N_k-m_k)*log(1-p_k) ] (grid + golden-section).

Error bars (both reported - never a single number):
  * Fisher / Cramer-Rao asymptotic SE.
  * Parametric bootstrap CI.

MIXED PROVENANCE (see THIRD_PARTY_NOTICES.md): the point-estimator
and grid/golden-section refinement pattern is adapted (MIT) from
spinqit-qae-triangulum `postprocess.py`. The Fisher SE and bootstrap CI in `estimate()`
below are Asymmetrica-original, kept in this same function because they are one coupled
estimator, not because the provenance boundary is unclear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def p_k(a: float, k: int) -> float:
    a = min(max(a, 0.0), 1.0)
    theta = math.asin(math.sqrt(a))
    return math.sin((2 * k + 1) * theta) ** 2


def _nll(a: float, ks: Sequence[int], m: Sequence[int], N: Sequence[int]) -> float:
    val = 0.0
    for k, mk, Nk in zip(ks, m, N):
        pk = min(max(p_k(a, int(k)), _EPS), 1.0 - _EPS)
        val -= mk * math.log(pk) + (Nk - mk) * math.log(1.0 - pk)
    return val


@lru_cache(maxsize=32)
def _log_likelihood_grid(ks: tuple[int, ...], grid_size: int):
    """Release modification, 2026-11-02: cache vectorized original grid terms.

    Same grid, clipping, binomial likelihood and subsequent golden-section
    refinement as the MIT-adapted scalar implementation; no sampling/trial
    counts or estimator model are changed. See RELEASE_CHANGES.md.
    """
    grid = np.linspace(0.0, 1.0, grid_size)
    powers = 2 * np.asarray(ks) + 1
    probs = np.clip(np.sin(powers[:, None] * np.arcsin(np.sqrt(grid))) ** 2,
                    _EPS, 1.0 - _EPS)
    log_p, log_one_minus_p = np.log(probs), np.log1p(-probs)
    for array in (grid, log_p, log_one_minus_p):
        array.setflags(write=False)
    return grid, log_p, log_one_minus_p


def _mle_point(ks: Sequence[int], m: Sequence[int], N: Sequence[int], grid_size: int = 20001) -> float:
    A, log_p, log_one_minus_p = _log_likelihood_grid(tuple(int(k) for k in ks), grid_size)
    nlls = -(np.asarray(m) @ log_p + (np.asarray(N) - m) @ log_one_minus_p)
    a0 = float(A[int(np.argmin(nlls))])
    d = 5.0 / grid_size
    lo, hi = max(0.0, a0 - d), min(1.0, a0 + d)
    invphi = 2.0 / (1.0 + 5.0 ** 0.5)
    x1, x2 = hi - (hi - lo) * invphi, lo + (hi - lo) * invphi
    f1, f2 = _nll(x1, ks, m, N), _nll(x2, ks, m, N)
    for _ in range(80):
        if f1 > f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + (hi - lo) * invphi
            f2 = _nll(x2, ks, m, N)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - (hi - lo) * invphi
            f1 = _nll(x1, ks, m, N)
        if abs(hi - lo) < 1e-13:
            break
    return float((lo + hi) / 2.0)


def _dpk_da(a: float, k: int) -> float:
    a = min(max(a, _EPS), 1.0 - _EPS)
    theta = math.asin(math.sqrt(a))
    return (2 * k + 1) * math.sin(2 * (2 * k + 1) * theta) / (2.0 * math.sqrt(a * (1.0 - a)))


def fisher_information(a: float, ks: Sequence[int], N: Sequence[int]) -> float:
    info = 0.0
    for k, Nk in zip(ks, N):
        pk = min(max(p_k(a, int(k)), _EPS), 1.0 - _EPS)
        info += Nk * _dpk_da(a, int(k)) ** 2 / (pk * (1.0 - pk))
    return info


@dataclass(frozen=True)
class MLAEResult:
    a_hat: float
    theta_hat: float
    ks: Tuple[int, ...]
    shots: Tuple[int, ...]
    successes: Tuple[int, ...]
    p_hat: Tuple[float, ...]
    se_fisher: float
    ci_fisher: Tuple[float, float]
    ci_bootstrap: Tuple[float, float]
    z: float
    n_bootstrap: int
    nll: float
    meta: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        d = {
            "a_hat": self.a_hat, "theta_hat": self.theta_hat,
            "ks": list(self.ks), "shots": list(self.shots),
            "successes": list(self.successes), "p_hat": list(self.p_hat),
            "se_fisher": self.se_fisher, "ci_fisher": list(self.ci_fisher),
            "ci_bootstrap": list(self.ci_bootstrap), "z": self.z,
            "n_bootstrap": self.n_bootstrap, "nll": self.nll,
        }
        d.update(self.meta)
        return d


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate(
    ks: Sequence[int],
    successes: Sequence[int],
    shots: Sequence[int],
    *,
    z: float = 1.96,
    n_bootstrap: int = 2000,
    seed: int = 12345,
    meta: Optional[Dict[str, object]] = None,
) -> MLAEResult:
    ks = [int(k) for k in ks]
    m = [int(v) for v in successes]
    N = [int(v) for v in shots]
    a_hat = _mle_point(ks, m, N)

    info = fisher_information(a_hat, ks, N)
    se = float("inf") if info <= 0 else 1.0 / math.sqrt(info)
    if math.isfinite(se):
        ci_f = (max(0.0, a_hat - z * se), min(1.0, a_hat + z * se))
    else:
        ci_f = (0.0, 1.0)

    rng = np.random.default_rng(seed)
    p_obs = [mk / Nk if Nk > 0 else 0.0 for mk, Nk in zip(m, N)]
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        m_star = [int(rng.binomial(Nk, pk)) for Nk, pk in zip(N, p_obs)]
        boot[b] = _mle_point(ks, m_star, N, grid_size=4001)
    alpha = 2.0 * (1.0 - _norm_cdf(z))
    ci_b = (float(np.quantile(boot, alpha / 2.0)), float(np.quantile(boot, 1.0 - alpha / 2.0)))

    return MLAEResult(
        a_hat=a_hat, theta_hat=math.asin(math.sqrt(min(max(a_hat, 0.0), 1.0))),
        ks=tuple(ks), shots=tuple(N), successes=tuple(m),
        p_hat=tuple(p_obs), se_fisher=se, ci_fisher=ci_f, ci_bootstrap=ci_b,
        z=z, n_bootstrap=n_bootstrap, nll=_nll(a_hat, ks, m, N), meta=dict(meta or {}),
    )
