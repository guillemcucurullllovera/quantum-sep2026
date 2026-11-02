"""Noise-aware amplitude estimation — joint (theta, lambda) MLE with a depth-damped likelihood.

WHY THIS EXISTS (the certification gap, measured in our own results):
the canonical rigorous-CI estimator (IQAE, Clopper-Pearson) can return a confidence
interval that does not contain the truth once device noise is present, because that
CI bounds shot noise only, under a noiseless-oracle assumption. Device noise adds a
depth-dependent systematic bias. A hardware AE number without a noise-aware likelihood
is therefore *uncertified*. (Validated against a real fake-backend run:
on `deliverables/results/qae_iqae_barrier_crash5_20260723T071738.json`
(`modes.kingston_noisy`), the plain estimator returned a_hat=0.4367,
CI [0.4355, 0.4379] against a true value of 0.3880 — outside its own
"rigorous" interval; this module's damped-MLE CI is designed to cover cases
like that. A separate, independent FakeKingston validation of this module
itself lives in `deliverables/results/noise_aware_fakekingston_validation.json`
— its numbers, e.g. a_true=0.4012, are from a different run and should not
be conflated with the crash5 figures above.)

MODEL: under depolarizing-type noise that grows with circuit length, the Grover-power
success probability damps toward 1/2:

    p_k(theta, lam) = 1/2 - 1/2 * exp(-lam * (2k+1)) * cos(2 * (2k+1) * theta),   a = sin^2(theta)

lam >= 0 is the damping per oracle application (Q^k A contains (2k+1) applications of A / A^dag;
reflections are absorbed into lam). lam = 0 reduces EXACTLY to the ideal MLAE likelihood.
Readout error is handled separately (M3, `algorithms.qae_iqae.mitigation`); the certified
pipeline is M3 (readout) -> damped MLE (depth noise).

TECHNIQUE LABEL: REPRODUCTION — this is the known noisy-MLAE line (Tanaka, Suzuki et al.,
Quantum Inf. Process. 2021: amplitude estimation on noisy devices). Our contribution is wiring
it into a certified pipeline and (pre-registered) hardware validation. NO advantage claim.

CI: profile likelihood on theta (lam profiled out), Wilks chi^2_1 cutoff, reported on
a = sin^2(theta). Flags: `overdamped` (deepest circuit's interference below ~5% -> weak
identification) and `multimodal` (other theta regions also clear the cutoff).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-9
_WILKS_95 = 1.9207  # chi^2_1 0.95 quantile / 2


def damped_pk(theta: float, lam: float, k: int) -> float:
    """P(ancilla=1) for Q^k A under the depth-damped model. lam=0 -> ideal sin^2((2k+1)theta)."""
    return 0.5 - 0.5 * math.exp(-lam * (2 * k + 1)) * math.cos(2 * (2 * k + 1) * theta)


def _loglik_surface(ks, m, N, thetas: np.ndarray, lams: np.ndarray) -> np.ndarray:
    """Vectorised binomial log-likelihood L[i_lam, i_theta]."""
    L = np.zeros((lams.size, thetas.size))
    for k, mk, Nk in zip(ks, m, N):
        damp = np.exp(-lams * (2 * k + 1))[:, None]                  # (lam, 1)
        osc = np.cos(2 * (2 * k + 1) * thetas)[None, :]              # (1, theta)
        p = np.clip(0.5 - 0.5 * damp * osc, _EPS, 1.0 - _EPS)
        L += mk * np.log(p) + (Nk - mk) * np.log(1.0 - p)
    return L


@dataclass(frozen=True)
class NoiseAwareResult:
    a_hat: float
    theta_hat: float
    lam_hat: float
    ci_a: Tuple[float, float]           # profile-likelihood 95% CI on a
    damping_per_k: Dict[int, float]     # e^{-lam*(2k+1)} at the fit
    p_obs: Dict[int, float]
    p_fit: Dict[int, float]
    overdamped: bool
    multimodal: bool
    nll: float
    ks: Tuple[int, ...]
    shots: Tuple[int, ...]
    successes: Tuple[int, ...]
    meta: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "a_hat": self.a_hat, "theta_hat": self.theta_hat, "lam_hat": self.lam_hat,
            "ci_a": list(self.ci_a),
            "damping_per_k": {str(k): v for k, v in self.damping_per_k.items()},
            "p_obs": {str(k): v for k, v in self.p_obs.items()},
            "p_fit": {str(k): v for k, v in self.p_fit.items()},
            "overdamped": self.overdamped, "multimodal": self.multimodal,
            "nll": self.nll, "ks": list(self.ks), "shots": list(self.shots),
            "successes": list(self.successes), **self.meta,
        }


def estimate_noise_aware(
    ks: Sequence[int],
    successes: Sequence[int],
    shots: Sequence[int],
    *,
    theta_grid: int = 4001,
    lam_grid: int = 241,
    lam_max: float = 1.2,
    refine: bool = True,
    meta: Optional[Dict[str, object]] = None,
) -> NoiseAwareResult:
    """Jointly estimate (theta, lam) by MLE on a grid (+ optional local polish);
    profile-likelihood 95% CI on a = sin^2(theta)."""
    ks = [int(k) for k in ks]
    m = [int(v) for v in successes]
    N = [int(v) for v in shots]
    if len(ks) < 3:
        raise ValueError("need >= 3 Grover powers to identify (theta, lam)")

    thetas = np.linspace(0.0, math.pi / 2, theta_grid)
    lams = np.linspace(0.0, lam_max, lam_grid)
    L = _loglik_surface(ks, m, N, thetas, lams)

    i_lam, i_th = np.unravel_index(int(np.argmax(L)), L.shape)
    theta_hat, lam_hat = float(thetas[i_th]), float(lams[i_lam])
    best_ll = float(L[i_lam, i_th])

    if refine:
        try:
            from scipy.optimize import minimize

            def nll(x):
                th, la = x
                val = 0.0
                for k, mk, Nk in zip(ks, m, N):
                    p = min(max(damped_pk(th, la, k), _EPS), 1.0 - _EPS)
                    val -= mk * math.log(p) + (Nk - mk) * math.log(1.0 - p)
                return val

            res = minimize(nll, x0=[theta_hat, lam_hat], method="L-BFGS-B",
                           bounds=[(0.0, math.pi / 2), (0.0, lam_max)])
            if res.success and -res.fun >= best_ll - 1e-9:
                theta_hat, lam_hat = float(res.x[0]), float(res.x[1])
                best_ll = float(-res.fun)
        except Exception:
            pass  # grid estimate stands

    # Profile-likelihood CI on theta (lam profiled out), Wilks 95% cutoff — computed by
    # ROOT-FINDING on the continuous profile, not read off the grid: with large shot counts
    # the peak is narrower than any fixed grid spacing and a grid CI degenerates.
    def _nll_at(th: float, la: float) -> float:
        val = 0.0
        for k, mk, Nk in zip(ks, m, N):
            p = min(max(damped_pk(th, la, k), _EPS), 1.0 - _EPS)
            val -= mk * math.log(p) + (Nk - mk) * math.log(1.0 - p)
        return val

    def _profile_nll(th: float) -> float:
        """min over lam of nll(th, lam): vectorised lam-grid seed + bounded scalar polish."""
        vals = np.zeros_like(lams)
        for k, mk, Nk in zip(ks, m, N):
            pk = np.clip(0.5 - 0.5 * np.exp(-lams * (2 * k + 1)) * math.cos(2 * (2 * k + 1) * th),
                         _EPS, 1.0 - _EPS)
            vals -= mk * np.log(pk) + (Nk - mk) * np.log(1.0 - pk)
        best = float(vals.min())
        try:
            from scipy.optimize import minimize_scalar
            r = minimize_scalar(lambda la: _nll_at(th, la), bounds=(0.0, lam_max),
                                method="bounded", options={"xatol": 1e-9})
            best = min(best, float(r.fun))
        except Exception:
            pass
        return best

    best_nll_val = _nll_at(theta_hat, lam_hat)
    target = best_nll_val + _WILKS_95

    def _find_edge(direction: int) -> float:
        """Nearest theta (going ±) where profile nll crosses the Wilks target; boundary if none."""
        bound = math.pi / 2 if direction > 0 else 0.0
        step = 1e-6
        prev = theta_hat
        while True:
            cand = theta_hat + direction * step
            if (direction > 0 and cand >= bound) or (direction < 0 and cand <= bound):
                if _profile_nll(bound) <= target:
                    return bound                      # CI extends to the boundary
                cand = bound
            if _profile_nll(cand) > target:
                try:
                    from scipy.optimize import brentq
                    return float(brentq(lambda t: _profile_nll(t) - target,
                                        min(prev, cand), max(prev, cand), xtol=1e-10))
                except Exception:
                    return cand
            if cand == bound:
                return bound
            prev = cand
            step *= 2.0

    th_lo, th_hi = _find_edge(-1), _find_edge(+1)
    ci_a = (float(math.sin(th_lo) ** 2), float(math.sin(th_hi) ** 2))

    # Multimodality flag from the coarse global scan: any OTHER theta region whose grid profile
    # clears the cutoff outside the CI (half-grid-step slack) suggests aliased modes.
    Lp = L.max(axis=0)
    slack = (thetas[1] - thetas[0]) if thetas.size > 1 else 0.0
    ok = Lp >= (-target)
    outside = ok & ((thetas < th_lo - slack) | (thetas > th_hi + slack))
    multimodal = bool(outside.any())

    k_max = max(ks)
    damping = {k: float(math.exp(-lam_hat * (2 * k + 1))) for k in ks}
    result = NoiseAwareResult(
        a_hat=float(math.sin(theta_hat) ** 2), theta_hat=theta_hat, lam_hat=lam_hat,
        ci_a=ci_a, damping_per_k=damping,
        p_obs={k: (mk / Nk if Nk else 0.0) for k, mk, Nk in zip(ks, m, N)},
        p_fit={k: damped_pk(theta_hat, lam_hat, k) for k in ks},
        overdamped=bool(lam_hat * (2 * k_max + 1) > 3.0),
        multimodal=multimodal, nll=float(-best_ll),
        ks=tuple(ks), shots=tuple(N), successes=tuple(m), meta=dict(meta or {}),
    )
    return result
