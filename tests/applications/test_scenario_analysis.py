import numpy as np
import pytest

from applications.scenario_analysis.domain import ScenarioSpec, ShockKind, ShockSpec
from applications.scenario_analysis.generator import ScenarioGenerator, default_scenario_grid
from applications.scenario_analysis.runner import run_scenario_grid


def _base_pnl(seed: int = 1, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.01, n)


def test_vol_spike_actually_widens_spread():
    gen = ScenarioGenerator(seed=1)
    base = _base_pnl()
    shocked = gen.vol_spike(base, multiplier=3.0)
    assert not np.array_equal(base, shocked)
    assert shocked.std() == pytest.approx(base.std() * 3.0, rel=1e-6)
    assert shocked.mean() == pytest.approx(base.mean(), abs=1e-9)


def test_mean_shift_actually_shifts_mean():
    gen = ScenarioGenerator(seed=1)
    base = _base_pnl()
    shocked = gen.mean_shift(base, shift=-0.05)
    assert shocked.mean() == pytest.approx(base.mean() - 0.05, abs=1e-9)
    assert shocked.std() == pytest.approx(base.std(), abs=1e-9)


def test_jump_creates_real_structural_break():
    gen = ScenarioGenerator(seed=2)
    base = _base_pnl(n=5000)
    shocked = gen.jump(base, jump_prob=0.05, jump_size=-0.3, seed=2)
    assert not np.array_equal(base, shocked)
    # A real jump should push the minimum well below the base minimum and
    # fatten the left tail (more negative skew).
    assert shocked.min() < base.min() - 0.1
    n_changed = int(np.sum(shocked != base))
    assert n_changed > 0
    # jump_prob=0.05 over 5000 samples -> roughly 250 changed, not all/none
    assert 50 < n_changed < 500


def test_correlation_breakdown_actually_raises_correlation():
    rng = np.random.default_rng(3)
    t, k = 2000, 4
    # Near-independent base returns.
    base = rng.normal(0, 0.01, size=(t, k))
    base_corr = np.corrcoef(base, rowvar=False)
    base_offdiag = base_corr[~np.eye(k, dtype=bool)]

    gen = ScenarioGenerator(seed=3)
    shocked = gen.correlation_breakdown(base, target_corr=0.9, seed=3)
    shocked_corr = np.corrcoef(shocked, rowvar=False)
    shocked_offdiag = shocked_corr[~np.eye(k, dtype=bool)]

    assert shocked.shape == base.shape
    assert not np.array_equal(base, shocked)
    assert shocked_offdiag.mean() > base_offdiag.mean() + 0.3
    assert shocked_offdiag.mean() == pytest.approx(0.9, abs=0.15)


def test_correlation_breakdown_rejects_1d_input():
    gen = ScenarioGenerator(seed=1)
    with pytest.raises(ValueError):
        gen.correlation_breakdown(_base_pnl(), target_corr=0.9)


def test_generate_grid_produces_distinct_scenarios_not_copies():
    gen = ScenarioGenerator(seed=4)
    base = _base_pnl(seed=5)
    scenarios = default_scenario_grid()
    grid = gen.generate_grid(base, scenarios, seed=4)

    assert set(grid) == {s.name for s in scenarios}
    assert np.array_equal(grid["baseline"], base)  # identity scenario == base, exactly

    non_baseline = [name for name in grid if name != "baseline"]
    # Every non-baseline scenario must differ from the base sample...
    for name in non_baseline:
        assert not np.array_equal(grid[name], base), f"{name} is identical to baseline"
    # ...and every pair of non-baseline scenarios must differ from each other
    # (real distinct scenarios, not copies of one shock applied everywhere).
    for i, a in enumerate(non_baseline):
        for b in non_baseline[i + 1:]:
            assert not np.array_equal(grid[a], grid[b]), f"{a} and {b} are identical"

    # Vol-spike scenarios should actually have larger std than baseline.
    assert grid["severe_vol_spike"].std() > grid["mild_vol_spike"].std() > base.std()


def test_run_scenario_grid_wires_arbitrary_risk_fn():
    gen = ScenarioGenerator(seed=6)
    base = _base_pnl(seed=6)
    scenarios = [
        ScenarioSpec(name="baseline", shocks=[]),
        ScenarioSpec(name="down_shift", shocks=[ShockSpec(kind=ShockKind.MEAN_SHIFT, params={"shift": -0.03})]),
    ]

    def risk_fn(shocked: np.ndarray) -> dict:
        return {"mean": float(shocked.mean()), "std": float(shocked.std())}

    report = run_scenario_grid(base, scenarios, risk_fn, generator=gen, seed=6)
    assert report.base_n_samples == len(base)
    assert report.scenario_names() == ["baseline", "down_shift"]
    baseline_row = report.by_name("baseline")
    shifted_row = report.by_name("down_shift")
    assert shifted_row.risk["mean"] < baseline_row.risk["mean"]

    with pytest.raises(KeyError):
        report.by_name("nonexistent")
