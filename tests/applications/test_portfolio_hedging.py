import numpy as np
import pytest

from applications.derivatives.portfolio_hedging import hedge_tail_risk, simulate_portfolio_pnl
from applications.portfolio.domain import AssetSpec, ConstraintSpec, PortfolioSpec, SolverChoice
from applications.portfolio.multi_solver import solve_portfolio


def _toy_spec_and_weights(high_vol: bool = False):
    rng = np.random.default_rng(7)
    n = 6
    mu = rng.uniform(0.02, 0.15, size=n)
    # High-vol variant deliberately gives sigma >> mu so the simulated P&L
    # distribution has genuine downside tail risk (prob_of_loss > 0),
    # rather than the low-vol default where 1-year drift dominates and the
    # sample is loss-free -- VaR/CVaR sign conventions are only intuitive
    # to reason about ("hedging reduces loss") when there IS a loss tail.
    scale = 0.6 if high_vol else 0.05
    L = rng.normal(scale=scale, size=(n, n))
    cov = L @ L.T + np.eye(n) * (0.25 if high_vol else 0.01)
    spec = PortfolioSpec(
        assets=[AssetSpec(name=f"A{i}", expected_return=float(m)) for i, m in enumerate(mu)],
        cov_matrix=cov.tolist(),
        constraints=[ConstraintSpec(kind="cardinality", params={"k": 3})],
    )
    result = solve_portfolio(spec, SolverChoice.MODELED_REPAIR, seed=1)
    return spec, result.bitstring


def test_simulate_portfolio_pnl_produces_correlated_samples_with_right_shape():
    spec, weights = _toy_spec_and_weights()
    pnl = simulate_portfolio_pnl(spec, weights, n_paths=2048, seed=0)
    assert pnl.shape == (2048,)
    assert np.isfinite(pnl).all()


def test_hedge_scales_var_and_cvar_by_positive_homogeneity():
    # VaR and CVaR (Expected Shortfall) are positively homogeneous risk
    # measures: VaR(s*X) = s*VaR(X) and CVaR(s*X) = s*CVaR(X) for scale
    # factor s > 0, regardless of the sign of either value (that sign
    # depends on whether the sampled tail is a loss or a favorable
    # outcome, and isn't itself a fixed property of scaling). This is the
    # actual mathematical property a linear-exposure hedge has to satisfy;
    # a naive "hedged VaR must be numerically smaller" inequality is wrong
    # whenever the sample has no loss tail (a real bug an early version of
    # this test had, caught by running it).
    spec, weights = _toy_spec_and_weights(high_vol=True)
    comparison = hedge_tail_risk(spec, weights, hedge_scale=0.4, n_paths=4096, seed=0, run_quantum=False)
    assert comparison.unhedged.prob_of_loss > 0, "test setup should produce a real loss tail"
    assert comparison.hedged.var == pytest.approx(0.4 * comparison.unhedged.var, rel=0.15)
    assert comparison.hedged.cvar == pytest.approx(0.4 * comparison.unhedged.cvar, rel=0.15)
    # with a genuine loss tail present, scaling exposure down by 0.4 must
    # shrink the magnitude of both the loss-at-VaR and expected shortfall
    assert abs(comparison.hedged.var) < abs(comparison.unhedged.var)
    assert abs(comparison.hedged.cvar) < abs(comparison.unhedged.cvar)
    assert comparison.cvar_improved
