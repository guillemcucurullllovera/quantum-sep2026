"""Scenario analysis over option pricers -- payoff/price sensitivity to spot and
volatility, reusing applications.options rather than re-deriving pricing logic."""
from applications.derivatives.scenario_analysis import ScenarioSweepResult, run_spot_vol_sweep

__all__ = ["ScenarioSweepResult", "run_spot_vol_sweep"]
