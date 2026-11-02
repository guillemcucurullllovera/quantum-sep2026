"""Tests for `experiments.quantum1b.mixer_warmstart_bench` (Track F, EXP-F1)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.classical_exact import solve_exact_milp
from experiments.quantum1b.mixer_warmstart_bench import (
    SEEDS,
    StrategyRunResult,
    run_strategy_comparison,
)


@pytest.fixture(scope="module")
def small_instance_exact_optimum():
    inst = generate_instance("independent", 4, seed=0)
    exact = solve_exact_milp(inst)
    return exact.objective


class TestRunStrategyComparison:
    def test_returns_all_four_groups(self, small_instance_exact_optimum):
        result = run_strategy_comparison(
            family="independent", n_candidates=4, seed=0, exact_optimum=small_instance_exact_optimum
        )
        assert set(result.keys()) == {
            "cold_start_random_p1", "cold_start_random_p2", "linear_ramp", "interp_p2",
        }

    def test_cold_start_has_one_result_per_seed(self, small_instance_exact_optimum):
        result = run_strategy_comparison(
            family="independent", n_candidates=4, seed=0, exact_optimum=small_instance_exact_optimum
        )
        assert len(result["cold_start_random_p1"]) == len(SEEDS)
        assert len(result["cold_start_random_p2"]) == len(SEEDS)
        for r in result["cold_start_random_p1"]:
            assert r.seed is not None
            assert r.p == 1

    def test_linear_ramp_has_exactly_two_results_p1_and_p2(self, small_instance_exact_optimum):
        result = run_strategy_comparison(
            family="independent", n_candidates=4, seed=0, exact_optimum=small_instance_exact_optimum
        )
        lr = result["linear_ramp"]
        assert len(lr) == 2
        assert {r.p for r in lr} == {1, 2}
        assert all(r.seed is None for r in lr)  # deterministic, no seed

    def test_interp_has_one_result_per_p1_seed_and_is_depth_two(self, small_instance_exact_optimum):
        result = run_strategy_comparison(
            family="independent", n_candidates=4, seed=0, exact_optimum=small_instance_exact_optimum
        )
        interp = result["interp_p2"]
        assert len(interp) == len(SEEDS)
        assert all(r.p == 2 for r in interp)
        assert {r.seed for r in interp} == set(SEEDS)

    def test_objective_gap_never_exceeds_one_and_never_negative_beyond_tolerance(
        self, small_instance_exact_optimum
    ):
        """A QAOA heuristic's best feasible objective can tie but never
        legitimately beat the exact optimum by more than floating-point
        tolerance -- a meaningfully negative gap would indicate a real
        bug (e.g. comparing against the wrong ground truth)."""
        result = run_strategy_comparison(
            family="independent", n_candidates=4, seed=0, exact_optimum=small_instance_exact_optimum
        )
        for group in result.values():
            for r in group:
                if r.objective_gap is not None:
                    assert r.objective_gap >= -1e-6
                    assert r.objective_gap <= 1.0 + 1e-6

    def test_all_results_are_strategy_run_result(self, small_instance_exact_optimum):
        result = run_strategy_comparison(
            family="independent", n_candidates=4, seed=0, exact_optimum=small_instance_exact_optimum
        )
        for group in result.values():
            for r in group:
                assert isinstance(r, StrategyRunResult)
                assert r.n_evaluations > 0
                assert np.isfinite(r.final_training_objective)
                assert 0.0 <= r.feasible_fraction <= 1.0
