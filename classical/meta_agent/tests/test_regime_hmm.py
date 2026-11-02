"""
Smoke tests for RegimeHMM (C1.1).

All tests run on synthetic data — no FRED, no Yahoo, no external API. They
gate the git push: if any of these fail, the module is broken regardless
of how it performs on real macro data.

Tests cover:
    1. NBER prior helper produces a valid (3, 3) row-stochastic-ish matrix
       with all entries > 0, and respects the empirical-vs-default toggle.
    2. Fit on synthetic 3-regime data recovers the transition matrix
       diagonal within tolerance.
    3. predict_proba returns a valid probability simplex per timestep.
    4. predict returns valid integer state labels in {0, 1, 2}.
    5. Post-hoc relabeling: state 0 has the smallest VIX-mean, state 2 the
       largest, regardless of EM's random initialization.
    6. Save/load round-trip preserves all model state.
    7. NBER prior with synthetic USREC matches the analytic stickiness
       within rounding.
    8. summary() returns sensible stationary distribution + dwell times.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from classical.meta_agent.regime_hmm import (
    RegimeHMM,
    build_nber_prior,
    NBER_MEDIAN_EXPANSION_MONTHS,
    NBER_MEDIAN_RECESSION_MONTHS,
)


# ── synthetic-data generator ──

def _generate_synthetic_3regime_panel(
    T: int = 600,
    seed: int = 0,
    n_features: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic 3-regime macro-like panel:
        state 0 (bull):   feature 0 (VIX-like) has mean = 15
        state 1 (bear):   feature 0 has mean = 25
        state 2 (crisis): feature 0 has mean = 45
    Other features are uncorrelated noise with state-specific means.
    Returns (X, true_states).
    """
    rng = np.random.default_rng(seed)
    # Sticky transition matrix similar to what we'd expect from NBER
    T_mat = np.array([
        [0.97, 0.025, 0.005],   # bull rarely leaves
        [0.10, 0.85, 0.05],     # bear is moderately sticky
        [0.05, 0.50, 0.45],     # crisis de-escalates to bear
    ])
    # Per-state means (feature 0 is VIX-like; others are arbitrary)
    means = np.array([
        [15.0,  2.0, 0.0, 100.0,  0.20],   # bull
        [25.0,  1.0, 0.5, 200.0,  0.18],   # bear
        [45.0,  0.0, 2.0, 400.0,  0.15],   # crisis
    ])
    # Per-state std (diag covariance)
    stds = np.array([
        [2.0, 0.3, 0.4, 20.0, 0.02],
        [4.0, 0.5, 0.6, 40.0, 0.02],
        [8.0, 0.8, 1.0, 80.0, 0.03],
    ])
    states = np.zeros(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = rng.choice(3, p=T_mat[states[t - 1]])
    X = np.empty((T, n_features))
    for t in range(T):
        s = states[t]
        X[t] = rng.normal(means[s], stds[s])
    return X, states


# ── tests ──

def test_nber_prior_default_shape_and_positivity():
    """build_nber_prior() with no usrec returns a (3,3) positive matrix."""
    alpha = build_nber_prior()
    assert alpha.shape == (3, 3)
    assert (alpha > 0).all(), "Dirichlet alpha must be all positive"
    # Diagonal should dominate (sticky regimes)
    for i in range(3):
        assert alpha[i, i] > alpha[i, (i + 1) % 3]


def test_nber_prior_matches_median_dwell_times():
    """Default prior's bull-diagonal should reflect ~58-month expansion median."""
    alpha = build_nber_prior(effective_n=100.0)
    # Reconstruct implied transition probs by row-normalizing alpha/effective_n
    row0 = alpha[0] / alpha[0].sum()
    # Default: stay_exp = 1 - 1/58 ≈ 0.9828
    expected_stay = 1.0 - 1.0 / NBER_MEDIAN_EXPANSION_MONTHS
    assert abs(row0[0] - expected_stay) < 0.01


def test_nber_prior_with_empirical_usrec():
    """Passing a synthetic USREC with known stickiness reproduces it."""
    rng = np.random.default_rng(7)
    T = 800
    usrec = np.zeros(T, dtype=int)
    # Hand-construct a series with stay_exp ≈ 0.95, stay_rec ≈ 0.80
    state = 0
    for t in range(1, T):
        if state == 0:
            state = 0 if rng.random() < 0.95 else 1
        else:
            state = 1 if rng.random() < 0.80 else 0
        usrec[t] = state
    alpha = build_nber_prior(usrec=usrec, effective_n=100.0)
    row0 = alpha[0] / alpha[0].sum()
    # bull-diagonal should be near 0.95
    assert abs(row0[0] - 0.95) < 0.03, f"got {row0[0]:.3f}, expected ~0.95"


def test_nber_prior_rejects_non_binary_usrec():
    with pytest.raises(ValueError, match="binary"):
        build_nber_prior(usrec=np.array([0, 1, 2, 0]))


def test_nber_prior_rejects_wrong_n_states():
    with pytest.raises(NotImplementedError):
        build_nber_prior(n_states=4)


def test_regime_hmm_rejects_wrong_n_states():
    with pytest.raises(NotImplementedError):
        RegimeHMM(n_states=2)


def test_fit_recovers_transition_diagonal():
    """Fit on synthetic 3-regime data should recover the true transition diagonal."""
    X, true_states = _generate_synthetic_3regime_panel(T=1200, seed=1)
    prior = build_nber_prior(effective_n=50.0)
    m = RegimeHMM(n_states=3, vix_col=0, random_state=1).fit(X, transmat_prior=prior)
    T_fit = np.array(m.summary().transmat)
    # True diagonal: [0.97, 0.85, 0.45]. Allow ±0.10 tolerance — EM is noisy
    # at the crisis-state (rare) end.
    assert abs(T_fit[0, 0] - 0.97) < 0.10, f"bull-stay {T_fit[0, 0]:.3f} vs 0.97"
    assert abs(T_fit[1, 1] - 0.85) < 0.15, f"bear-stay {T_fit[1, 1]:.3f} vs 0.85"
    # Crisis diagonal tolerance widened: rare state, hard to estimate
    assert abs(T_fit[2, 2] - 0.45) < 0.30, f"crisis-stay {T_fit[2, 2]:.3f} vs 0.45"


def test_predict_proba_is_a_valid_simplex():
    """predict_proba rows should sum to 1 and be non-negative."""
    X, _ = _generate_synthetic_3regime_panel(T=400, seed=2)
    m = RegimeHMM(random_state=2).fit(X)
    P = m.predict_proba(X)
    assert P.shape == (400, 3)
    assert (P >= -1e-9).all(), "negative probabilities returned"
    row_sums = P.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_predict_returns_valid_state_labels():
    X, _ = _generate_synthetic_3regime_panel(T=300, seed=3)
    m = RegimeHMM(random_state=3).fit(X)
    s = m.predict(X)
    assert s.shape == (300,)
    assert set(np.unique(s).tolist()) <= {0, 1, 2}


def test_post_hoc_labeling_orders_states_by_vix():
    """After fit, state 0 must have the smallest VIX-feature mean."""
    X, _ = _generate_synthetic_3regime_panel(T=600, seed=4)
    for rs in [0, 1, 2, 3, 4]:                  # different EM inits
        m = RegimeHMM(random_state=rs).fit(X)
        means = np.array(m.summary().state_means)
        vix_means = means[:, 0]
        assert vix_means[0] < vix_means[1] < vix_means[2], (
            f"random_state={rs}: VIX means not sorted: {vix_means}"
        )


def test_save_load_roundtrip():
    """Save then load should reproduce predict_proba bit-for-bit."""
    X, _ = _generate_synthetic_3regime_panel(T=200, seed=5)
    m = RegimeHMM(random_state=5, feature_names=["vix", "ts", "dxy", "ig", "cgr"]).fit(X)
    P_before = m.predict_proba(X)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.pkl")
        m.save(path)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(td, "model.summary.json"))
        m2 = RegimeHMM.load(path)
        P_after = m2.predict_proba(X)
    np.testing.assert_array_equal(P_before, P_after)
    assert m2.feature_names == ["vix", "ts", "dxy", "ig", "cgr"]


def test_summary_stationary_distribution_sums_to_one():
    X, _ = _generate_synthetic_3regime_panel(T=400, seed=6)
    m = RegimeHMM(random_state=6).fit(X)
    s = m.summary()
    assert abs(sum(s.stationary_distribution) - 1.0) < 1e-6
    assert all(p >= 0 for p in s.stationary_distribution)
    # Bull state should dominate the stationary distribution on this synthetic
    # data (transition diagonal 0.97 in row 0 means bull is by far the most
    # common state)
    assert s.stationary_distribution[0] > 0.5


def test_summary_dwell_times_are_positive():
    X, _ = _generate_synthetic_3regime_panel(T=400, seed=7)
    m = RegimeHMM(random_state=7).fit(X)
    s = m.summary()
    assert all(d > 1.0 for d in s.mean_dwell_months), (
        f"dwell times must all be > 1 month; got {s.mean_dwell_months}"
    )


def test_fit_rejects_nan_input():
    X, _ = _generate_synthetic_3regime_panel(T=100, seed=8)
    X[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RegimeHMM(random_state=8).fit(X)


def test_fit_rejects_wrong_prior_shape():
    X, _ = _generate_synthetic_3regime_panel(T=100, seed=9)
    bad_prior = np.eye(4) * 10.0
    with pytest.raises(ValueError, match="transmat_prior"):
        RegimeHMM(random_state=9).fit(X, transmat_prior=bad_prior)


def test_score_returns_finite_log_likelihood():
    X, _ = _generate_synthetic_3regime_panel(T=200, seed=10)
    m = RegimeHMM(random_state=10).fit(X)
    ll = m.score(X)
    assert np.isfinite(ll)
