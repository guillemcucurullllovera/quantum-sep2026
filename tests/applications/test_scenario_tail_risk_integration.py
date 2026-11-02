import numpy as np

from applications.scenario_analysis.domain import ScenarioSpec, ShockKind, ShockSpec
from applications.scenario_analysis.tail_risk_adapter import run_tail_risk_scenarios
from applications.tail_risk.domain import TailRiskSpec


def _base_pnl(seed: int = 1, n: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(df=5, size=n) * 0.01


def test_scenario_grid_runs_real_tail_risk_comparison_per_scenario():
    base = _base_pnl()
    scenarios = [
        ScenarioSpec(name="baseline", shocks=[]),
        ScenarioSpec(name="vol_spike", shocks=[ShockSpec(kind=ShockKind.VOL_SPIKE, params={"multiplier": 2.5})]),
        ScenarioSpec(name="down_shift", shocks=[ShockSpec(kind=ShockKind.MEAN_SHIFT, params={"shift": -0.02})]),
    ]
    spec = TailRiskSpec(confidence_levels=[0.95], run_quantum=False, seed=1)

    report = run_tail_risk_scenarios(base, scenarios, spec=spec, seed=1)

    assert report.scenario_names() == ["baseline", "vol_spike", "down_shift"]
    for row in report.rows:
        comparison = row.risk
        assert set(comparison.results) == {
            "historical", "parametric_gaussian", "cornish_fisher", "monte_carlo_gbm", "meta_agent_bootstrap",
        }
        for name, m in comparison.results.items():
            assert m.cvar >= m.var - 1e-9, f"{row.scenario}/{name}: cvar < var"

    baseline_var = report.by_name("baseline").risk.results["historical"].var
    vol_spike_var = report.by_name("vol_spike").risk.results["historical"].var
    down_shift_var = report.by_name("down_shift").risk.results["historical"].var

    # A vol-spike scenario should produce a materially larger VaR than
    # baseline (wider spread -> bigger tail loss at the same confidence).
    assert vol_spike_var > baseline_var * 1.3
    # A persistent downward mean-shift should also increase VaR.
    assert down_shift_var > baseline_var


def test_quantum_scenarios_subset_only_runs_quantum_for_selected_names():
    base = _base_pnl(seed=9, n=600)
    scenarios = [
        ScenarioSpec(name="baseline", shocks=[]),
        ScenarioSpec(name="crash_jump", shocks=[ShockSpec(kind=ShockKind.JUMP, params={"jump_prob": 0.05, "jump_size": -0.2})]),
    ]
    spec = TailRiskSpec(run_quantum=False, n_grid_qubits=2, epsilon=0.1, shots=256, seed=9)

    report = run_tail_risk_scenarios(base, scenarios, spec=spec, quantum_scenarios=("crash_jump",), seed=9)

    baseline_results = report.by_name("baseline").risk.results
    crash_results = report.by_name("crash_jump").risk.results
    assert "qae_amplitude_estimation" not in baseline_results
    assert "qae_amplitude_estimation" in crash_results
