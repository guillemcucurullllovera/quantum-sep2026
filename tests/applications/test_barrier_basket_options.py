import numpy as np
import pytest

from applications.options.barrier_pricer import BarrierOptionPricer, BasketOptionPricer
from applications.tail_risk.var_cvar import assess_tail_risk


def test_up_and_out_barrier_priced_below_vanilla():
    pricer = BarrierOptionPricer()
    quote = pricer.price("up_and_out", s0=100.0, strike=100.0, barrier=115.0, n=4)
    # a knock-out barrier can only remove payoff, never add it
    assert quote.quantum_price <= quote.vanilla_upper_bound + 1e-9
    assert quote.monte_carlo_terminal_only <= quote.vanilla_upper_bound + 1e-9


def test_barrier_quantum_price_agrees_with_terminal_only_monte_carlo():
    pricer = BarrierOptionPricer()
    quote = pricer.price("up_and_out", s0=100.0, strike=100.0, barrier=120.0, n=5)
    # both are apples-to-apples terminal-only constructions -- should be close
    assert quote.abs_error_vs_terminal_parity < 0.5


def test_down_and_out_knocks_out_when_barrier_far_above_spot_is_invalid_direction():
    # down_and_out means knocked out if S_T <= barrier; barrier far below spot
    # should barely reduce the price relative to vanilla
    pricer = BarrierOptionPricer()
    near_vanilla = pricer.price("down_and_out", s0=100.0, strike=100.0, barrier=1.0, n=4)
    assert near_vanilla.quantum_price == pytest.approx(near_vanilla.vanilla_upper_bound, rel=0.05)


def test_barrier_resource_estimate_present():
    pricer = BarrierOptionPricer()
    quote = pricer.price("up_and_out", barrier=115.0, n=4, target_epsilon=0.02)
    assert quote.resource_estimate.iqae_expected_queries > 0
    assert quote.resource_estimate.circuit_stats is not None


def test_basket_option_agrees_with_monte_carlo_within_grid_tolerance():
    pricer = BasketOptionPricer()
    quote = pricer.price(n_a=2, n_b=2)
    assert quote.within_mc_ci, (
        f"quantum={quote.quantum_price} mc={quote.monte_carlo_price} "
        f"stderr={quote.monte_carlo_stderr} err={quote.abs_error}"
    )


def test_basket_option_rejects_too_many_qubits():
    pricer = BasketOptionPricer()
    with pytest.raises(ValueError):
        pricer.price(n_a=4, n_b=4)


def test_tail_risk_meta_agent_comparator_present_and_negative():
    rng = np.random.default_rng(7)
    pnl = rng.standard_t(df=4, size=2000) * 0.02  # fat-tailed synthetic PnL
    report = assess_tail_risk(pnl, alpha=0.95, run_quantum=False, run_meta_agent_comparator=True)
    assert report.meta_agent_cvar is not None
    assert report.meta_agent_cvar < 0  # a loss, expressed as a negative return
    # all three tail-loss estimates should be in the same ballpark (same distribution)
    assert abs(report.meta_agent_cvar) == pytest.approx(report.cvar, rel=0.6)


def test_tail_risk_meta_agent_comparator_can_be_disabled():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0, 0.01, 500)
    report = assess_tail_risk(pnl, run_quantum=False, run_meta_agent_comparator=False)
    assert report.meta_agent_cvar is None
