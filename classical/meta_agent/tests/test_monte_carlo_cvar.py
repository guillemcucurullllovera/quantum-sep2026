"""
Smoke tests for MonteCarloCVaR (C1.5).

All synthetic data. These gate the git push.

Tests cover:
    1. Construction rejects invalid hyperparameters.
    2. Predict/estimate before fit raises.
    3. Bootstrap CVaR converges to analytical Gaussian CVaR within tolerance.
    4. Regime similarity collapses to uniform when regime probs are uniform.
    5. Regime conditioning works: drawing from crisis-weighted history
       produces fatter left tail than bull-weighted.
    6. Gross exposure scales DOWN when proposed weights have high tail risk.
    7. Gross exposure = max_gross when weights have low tail risk.
    8. CVaR is monotone in alpha (higher α → narrower tail → larger |CVaR|).
    9. apply_to_weights returns scaled = gross * weights.
    10. apply_batch returns valid shapes.
    11. Save/load roundtrip preserves bootstrap output bit-for-bit (with same RNG seed).
    12. Input validation.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest

from classical.meta_agent.monte_carlo_cvar import MonteCarloCVaR


# ── construction / validation ─────────────────────────────────────

def test_construction_rejects_invalid_args():
    with pytest.raises(ValueError, match="alpha"):
        MonteCarloCVaR(alpha=0.4)
    with pytest.raises(ValueError, match="alpha"):
        MonteCarloCVaR(alpha=1.0)
    with pytest.raises(ValueError, match="target_loss"):
        MonteCarloCVaR(target_loss=0.0)
    with pytest.raises(ValueError, match="target_loss"):
        MonteCarloCVaR(target_loss=-0.05)
    with pytest.raises(ValueError, match="n_bootstrap"):
        MonteCarloCVaR(n_bootstrap=50)
    with pytest.raises(ValueError, match="min_gross"):
        MonteCarloCVaR(min_gross=-0.1)
    with pytest.raises(ValueError, match="min_gross"):
        MonteCarloCVaR(min_gross=0.6, max_gross=0.5)


def test_estimate_before_fit_raises():
    cvar = MonteCarloCVaR()
    with pytest.raises(RuntimeError, match="not been fitted"):
        cvar.estimate_cvar(np.array([0.2, 0.2, 0.6]), np.array([1.0, 0.0, 0.0]))


def test_fit_rejects_wrong_shapes():
    cvar = MonteCarloCVaR()
    # 1-D returns
    with pytest.raises(ValueError, match="returns"):
        cvar.fit(np.zeros(50), np.ones((50, 3)) / 3)
    # T mismatch
    with pytest.raises(ValueError, match="T mismatch"):
        cvar.fit(np.zeros((50, 3)), np.ones((49, 3)) / 3)


def test_fit_rejects_nan():
    cvar = MonteCarloCVaR()
    bad_r = np.zeros((50, 3)); bad_r[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|finite"):
        cvar.fit(bad_r, np.ones((50, 3)) / 3)


def test_fit_rejects_negative_regime_probs():
    cvar = MonteCarloCVaR()
    bad_p = np.ones((50, 3)) / 3
    bad_p[5, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        cvar.fit(np.zeros((50, 3)), bad_p)


# ── CVaR math ─────────────────────────────────────────────────────

def test_bootstrap_cvar_approximates_gaussian_truth():
    """
    For Gaussian returns with mean 0, std σ, the analytic CVaR at α=0.95 is:
        CVaR_95 = -σ * φ(z_95) / (1 - 0.95)
    where φ is the standard-normal pdf and z_95 ≈ 1.645.
    For σ=0.05 → analytical CVaR_95 ≈ -0.05 * 0.1031 / 0.05 ≈ -0.1031
    Wait: CVaR_95 = -σ * φ(z) / (1 - 0.95) where z = Φ^{-1}(0.95).
    φ(1.6449) = 0.1031. So CVaR = -0.05 * 0.1031 / 0.05 = -0.1031.
    """
    rng = np.random.default_rng(0)
    T = 5000
    # 1 strategy with N(0, 0.05^2) returns; uniform regime probs
    returns = rng.standard_normal((T, 1)) * 0.05
    regime_probs = np.ones((T, 3)) / 3

    cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=20_000, random_state=0)
    cvar.fit(returns, regime_probs)
    est = cvar.estimate_cvar(np.array([1.0]), np.array([1/3, 1/3, 1/3]))

    analytical = -0.05 * 0.1031 / 0.05   # ≈ -0.1031
    # Bootstrap noise: should be within 0.01 (~10% relative)
    assert abs(est["cvar"] - analytical) < 0.01, (
        f"bootstrap cvar {est['cvar']:.4f} too far from analytical {analytical:.4f}"
    )


def test_cvar_monotone_in_alpha():
    """Higher α (narrower tail) → larger |CVaR| (more negative)."""
    rng = np.random.default_rng(1)
    T = 3000
    returns = rng.standard_normal((T, 1)) * 0.05
    regime_probs = np.ones((T, 3)) / 3

    cvars = []
    for alpha in [0.90, 0.95, 0.99]:
        cvar = MonteCarloCVaR(alpha=alpha, target_loss=0.05, n_bootstrap=15_000, random_state=1)
        cvar.fit(returns, regime_probs)
        est = cvar.estimate_cvar(np.array([1.0]), np.array([1/3, 1/3, 1/3]))
        cvars.append(est["cvar"])
    # Each successive CVaR should be MORE negative (larger magnitude loss)
    assert cvars[0] > cvars[1] > cvars[2], f"non-monotone: {cvars}"


def test_regime_conditioning_widens_left_tail_in_crisis():
    """
    Bootstrap drawing from crisis-weighted history should produce a more
    negative CVaR than drawing from bull-weighted history when crisis
    months had bigger losses.
    """
    rng = np.random.default_rng(2)
    T = 600
    # 1 strategy. Half the months are 'bull' (low vol), half are 'crisis' (high vol + neg mean).
    bull_returns = rng.standard_normal(T // 2) * 0.03
    crisis_returns = rng.standard_normal(T // 2) * 0.10 - 0.05
    returns = np.concatenate([bull_returns, crisis_returns])[:, None]

    regime_probs = np.zeros((T, 3))
    regime_probs[: T // 2, 0] = 1.0   # bull
    regime_probs[T // 2 :, 2] = 1.0   # crisis

    cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=20_000, random_state=2)
    cvar.fit(returns, regime_probs)

    cvar_bull = cvar.estimate_cvar(np.array([1.0]), np.array([1.0, 0.0, 0.0]))["cvar"]
    cvar_crisis = cvar.estimate_cvar(np.array([1.0]), np.array([0.0, 0.0, 1.0]))["cvar"]
    assert cvar_crisis < cvar_bull, (
        f"crisis-conditioned CVaR {cvar_crisis:.4f} not worse than "
        f"bull-conditioned {cvar_bull:.4f}"
    )
    # Crisis CVaR should be meaningfully larger in magnitude
    assert cvar_crisis < cvar_bull - 0.05, (
        "crisis CVaR not at least 5% more negative than bull"
    )


# ── gross exposure ───────────────────────────────────────────────

def test_gross_exposure_full_when_low_risk():
    """If proposed weights have low tail risk (well below target), gross = max."""
    rng = np.random.default_rng(3)
    T = 600
    # Very low-vol returns
    returns = rng.standard_normal((T, 2)) * 0.01
    regime_probs = np.ones((T, 3)) / 3
    cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=10_000, random_state=3)
    cvar.fit(returns, regime_probs)
    gross = cvar.compute_gross_exposure(np.array([0.5, 0.5]), np.array([1/3, 1/3, 1/3]))
    assert gross == 1.0, f"expected gross=1.0 for low-risk weights; got {gross}"


def test_gross_exposure_scales_down_when_high_risk():
    """If proposed weights have CVaR > target, gross should scale down."""
    rng = np.random.default_rng(4)
    T = 600
    # High-vol returns
    returns = rng.standard_normal((T, 2)) * 0.10 - 0.01
    regime_probs = np.ones((T, 3)) / 3
    cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=10_000, random_state=4)
    cvar.fit(returns, regime_probs)
    gross = cvar.compute_gross_exposure(np.array([0.5, 0.5]), np.array([1/3, 1/3, 1/3]))
    assert 0.0 < gross < 1.0, f"expected gross in (0, 1); got {gross}"


def test_apply_to_weights_returns_correct_scaling():
    rng = np.random.default_rng(5)
    T = 400
    returns = rng.standard_normal((T, 3)) * 0.08
    regime_probs = np.ones((T, 3)) / 3
    cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=8_000, random_state=5)
    cvar.fit(returns, regime_probs)
    weights = np.array([0.4, 0.3, 0.3])
    scaled, gross = cvar.apply_to_weights(weights, np.array([1/3, 1/3, 1/3]))
    np.testing.assert_allclose(scaled, weights * gross, rtol=1e-12)
    assert 0.0 < gross <= 1.0


def test_apply_batch_shapes():
    rng = np.random.default_rng(6)
    T_hist = 300
    returns = rng.standard_normal((T_hist, 4)) * 0.05
    regime_probs = np.ones((T_hist, 3)) / 3
    cvar = MonteCarloCVaR(alpha=0.95, target_loss=0.05, n_bootstrap=5_000, random_state=6)
    cvar.fit(returns, regime_probs)
    T_test = 12
    weights_T = rng.dirichlet(np.ones(4), size=T_test)
    regime_T = rng.dirichlet(np.ones(3), size=T_test)
    scaled, gross = cvar.apply_batch(weights_T, regime_T)
    assert scaled.shape == (T_test, 4)
    assert gross.shape == (T_test,)
    # Each row of scaled should equal corresponding weights times gross[t]
    for t in range(T_test):
        np.testing.assert_allclose(scaled[t], weights_T[t] * gross[t], rtol=1e-12)


# ── persistence ──────────────────────────────────────────────────

def test_save_load_roundtrip():
    rng = np.random.default_rng(7)
    T = 300
    returns = rng.standard_normal((T, 3)) * 0.04
    regime_probs = np.ones((T, 3)) / 3
    cvar = MonteCarloCVaR(
        alpha=0.97, target_loss=0.06, n_bootstrap=5_000, random_state=7,
    ).fit(returns, regime_probs, strategy_names=["a", "b", "c"])

    # Sample CVaR before save
    weights = np.array([0.4, 0.3, 0.3])
    regime_t = np.array([1/3, 1/3, 1/3])
    est_before = cvar.estimate_cvar(weights, regime_t)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cvar.pkl")
        cvar.save(path)
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(td, "cvar.summary.json"))
        cvar2 = MonteCarloCVaR.load(path)
        # Re-seed for deterministic comparison
        cvar._rng = np.random.default_rng(7)
        cvar2._rng = np.random.default_rng(7)
        est1 = cvar.estimate_cvar(weights, regime_t)
        est2 = cvar2.estimate_cvar(weights, regime_t)

    assert est1["cvar"] == est2["cvar"]
    assert cvar2.alpha == 0.97
    assert cvar2.target_loss == 0.06


def test_summary_returns_expected_fields():
    rng = np.random.default_rng(8)
    T = 100
    cvar = MonteCarloCVaR(alpha=0.99, target_loss=0.04).fit(
        rng.standard_normal((T, 2)) * 0.05,
        np.ones((T, 3)) / 3,
        strategy_names=["x", "y"],
    )
    s = cvar.summary()
    assert s.alpha == 0.99
    assert s.target_loss == 0.04
    assert s.n_history_months == T
    assert s.n_strategies == 2
    assert s.n_regimes == 3
    assert s.strategy_names == ["x", "y"]
    assert s.regime_names == ["bull", "bear", "crisis"]
