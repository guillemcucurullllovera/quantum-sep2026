from __future__ import annotations

import pytest

from applications.derivatives.scenario_grid import run_scenario_grid
from applications.options.domain import OptionSpec, PayoffKind
from applications.options.pricing_methods import PricingMethod
from core.errors import ValidationError


def test_spot_barrier_grid_produces_full_cartesian_product():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    grid = run_scenario_grid(spec, spots=(90.0, 100.0, 110.0), barriers=(115.0, 125.0), mc_paths=5000)
    assert len(grid.cells) == 3 * 2
    table = grid.as_table()
    assert all("classical_price" in row for row in table)


def test_scenario_grid_classical_price_rises_with_spot_for_up_and_out():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=140.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    # a barrier far above spot barely truncates the vanilla payoff -- price
    # should still rise with spot, like the vanilla underneath it
    grid = run_scenario_grid(spec, spots=(90.0, 100.0, 110.0), mc_paths=20_000)
    prices = [cell.classical.price for cell in grid.cells]
    assert prices[0] < prices[1] < prices[2]


def test_scenario_grid_can_cross_validate_selected_cells_with_quantum():
    spec = OptionSpec(payoff=PayoffKind.UP_AND_OUT_CALL, spot=100.0, strike=100.0, barrier=115.0,
                       rate=0.04, vol=0.2, maturity=0.25)
    grid = run_scenario_grid(
        spec, spots=(95.0, 100.0), barriers=(115.0, 120.0),
        quantum_method=PricingMethod.QAE_IQAE, quantum_cell_indices=(0, 3),
        mc_paths=5000, shots=1024, n_qubits=3,
    )
    quantum_cells = [c for c in grid.cells if c.quantum is not None]
    assert len(quantum_cells) == 2
    for cell in quantum_cells:
        assert cell.quantum.price >= 0.0


def test_basket_correlation_sweep_price_increases_with_correlation():
    spec = OptionSpec(payoff=PayoffKind.BASKET_CALL, spot=100.0, spot2=100.0, strike=100.0,
                       vol=0.2, vol2=0.25, correlation=0.0, rate=0.04, maturity=0.25)
    grid = run_scenario_grid(spec, correlations=(-0.5, 0.0, 0.5), mc_paths=20_000)
    prices = [cell.classical.price for cell in grid.cells]
    assert prices[0] < prices[1] < prices[2]  # higher correlation -> higher basket variance -> higher call value


def test_barriers_sweep_rejected_for_non_barrier_payoff():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0)
    with pytest.raises(ValidationError, match="barrier payoffs"):
        run_scenario_grid(spec, barriers=(110.0,))


def test_correlations_sweep_rejected_for_non_basket_payoff():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0)
    with pytest.raises(ValidationError, match="basket_call"):
        run_scenario_grid(spec, correlations=(0.2,))


def test_no_dimensions_rejected():
    spec = OptionSpec(payoff=PayoffKind.EUROPEAN_CALL, spot=100.0, strike=100.0)
    with pytest.raises(ValidationError, match="at least one"):
        run_scenario_grid(spec)
