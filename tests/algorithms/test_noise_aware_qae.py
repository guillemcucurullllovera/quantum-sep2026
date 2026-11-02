"""Tests for the noise-aware (theta, lambda) MLE — the certified-error-bars estimator.

Pure numpy/scipy (no qiskit needed): these run in any venv.
"""
import math

import numpy as np
import pytest

from algorithms_extended.qae_spinqit import mlae as M
from algorithms.qae_iqae.hardware.noise_aware import damped_pk, estimate_noise_aware

A_TRUE = 0.40120  # the folded-spread grid amplitude used across the QAE results
KS = [0, 1, 2, 3]


def _draw(a, lam, N, ks, seed):
    rng = np.random.default_rng(seed)
    theta = math.asin(math.sqrt(a))
    return [int(rng.binomial(N, damped_pk(theta, lam, k))) for k in ks]


def test_reduces_to_ideal_at_zero_noise():
    a = 0.37
    N = 100000
    theta = math.asin(math.sqrt(a))
    succ = [round(damped_pk(theta, 0.0, k) * N) for k in KS]  # exact ideal counts
    res = estimate_noise_aware(KS, succ, [N] * len(KS))
    assert abs(res.a_hat - a) < 5e-3
    assert res.lam_hat < 0.01
    assert res.ci_a[0] <= a <= res.ci_a[1]
    assert not res.overdamped


def test_recovers_amplitude_and_damping_from_damped_counts():
    lam = 0.08
    N = 4096
    succ = _draw(A_TRUE, lam, N, KS, seed=42)
    res = estimate_noise_aware(KS, succ, [N] * len(KS))
    assert res.ci_a[0] <= A_TRUE <= res.ci_a[1]      # truth inside the profile CI
    assert abs(res.lam_hat - lam) < 0.05             # damping recovered
    assert abs(res.a_hat - A_TRUE) < 0.03


def test_plain_mlae_misses_but_noise_aware_covers():
    """The certification point: under depth damping the plain estimator's CI is invalid;
    the noise-aware CI contains the truth."""
    lam = 0.08
    N = 4096
    succ = _draw(A_TRUE, lam, N, KS, seed=7)
    plain = M.estimate(KS, succ, [N] * len(KS), n_bootstrap=300, seed=7)
    aware = estimate_noise_aware(KS, succ, [N] * len(KS))
    lo, hi = plain.ci_bootstrap
    assert not (lo <= A_TRUE <= hi)                  # plain CI misses the truth
    assert aware.ci_a[0] <= A_TRUE <= aware.ci_a[1]  # noise-aware CI covers it


def test_coverage_synthetic():
    """Coverage over seeded trials at (a=0.35, lam=0.10): noise-aware ~nominal, plain badly below."""
    a, lam, N, trials = 0.35, 0.10, 2048, 60
    hits_aware = hits_plain = 0
    for s in range(trials):
        succ = _draw(a, lam, N, KS, seed=1000 + s)
        aware = estimate_noise_aware(KS, succ, [N] * len(KS),
                                     theta_grid=1501, lam_grid=101, refine=False)
        if aware.ci_a[0] <= a <= aware.ci_a[1]:
            hits_aware += 1
        plain = M.estimate(KS, succ, [N] * len(KS), n_bootstrap=120, seed=s)
        lo, hi = plain.ci_bootstrap
        if lo <= a <= hi:
            hits_plain += 1
    assert hits_aware / trials >= 0.85               # near-nominal coverage
    assert hits_plain / trials <= 0.5                # plain CI is invalid under damping


def test_overdamped_flag():
    succ = _draw(0.4, 1.0, 4096, KS, seed=3)
    res = estimate_noise_aware(KS, succ, [4096] * len(KS))
    assert res.overdamped
