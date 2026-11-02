"""Correctness tests for Track A's simulated-annealing fallback
(`experiments/quantum1b/classical_heuristic.py`) -- the documented,
explicitly-heuristic path EXP-A2 falls back to on a MILP timeout."""
from __future__ import annotations

import pytest

from experiments.quantum1b.classical_exact import ClassicalSolveResult, solve_exact_enumeration
from experiments.quantum1b.classical_heuristic import solve_simulated_annealing
from experiments.quantum1b.generators import FAMILIES, generate_instance
from experiments.quantum1b.instance import evaluate_selection


class TestSimulatedAnnealingBasics:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_returns_result_dataclass_shape(self, family):
        inst = generate_instance(family, 10, seed=0)
        r = solve_simulated_annealing(inst, n_iters=2000, n_restarts=3, seed=0)
        assert isinstance(r, ClassicalSolveResult)
        assert r.method == "simulated_annealing"
        assert len(r.selection) == inst.n()
        assert r.runtime_s >= 0.0

    def test_result_is_re_scored_through_evaluate_selection(self):
        inst = generate_instance("dense", 10, seed=0)
        r = solve_simulated_annealing(inst, n_iters=3000, n_restarts=3, seed=0)
        ev = evaluate_selection(inst, r.selection)
        assert ev.objective == pytest.approx(r.objective, abs=1e-12)
        assert ev.feasible == r.feasible

    def test_deterministic_given_seed(self):
        inst = generate_instance("dense", 10, seed=0)
        r1 = solve_simulated_annealing(inst, n_iters=2000, n_restarts=3, seed=7)
        r2 = solve_simulated_annealing(inst, n_iters=2000, n_restarts=3, seed=7)
        assert r1.selection == r2.selection
        assert r1.objective == pytest.approx(r2.objective, abs=1e-12)

    def test_different_seed_may_differ_solver_info_seeds_recorded(self):
        inst = generate_instance("dense", 10, seed=0)
        r = solve_simulated_annealing(inst, n_iters=2000, n_restarts=5, seed=3)
        assert r.solver_info["seeds"] == [3, 4, 5, 6, 7]
        assert len(r.solver_info["per_restart"]) == 5

    def test_n_restarts_respected(self):
        inst = generate_instance("independent", 8, seed=0)
        r = solve_simulated_annealing(inst, n_iters=1000, n_restarts=7, seed=0)
        assert r.solver_info["n_restarts"] == 7
        assert len(r.solver_info["per_restart"]) == 7


class TestSimulatedAnnealingQualityAndHonesty:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_feasible_or_honestly_labeled_infeasible(self, family):
        inst = generate_instance(family, 10, seed=1)
        r = solve_simulated_annealing(inst, n_iters=5000, n_restarts=5, seed=0)
        if not r.feasible:
            assert "NO feasible selection found" in r.solver_info["note"]
        else:
            assert "best FEASIBLE selection" in r.solver_info["note"]

    def test_never_beats_the_true_optimum(self):
        # A heuristic's reported (feasible) objective can never exceed the
        # true global optimum -- if it did, that would indicate a scoring
        # bug, not a lucky heuristic.
        for family in FAMILIES:
            inst = generate_instance(family, 10, seed=2)
            r_exact = solve_exact_enumeration(inst)
            r_sa = solve_simulated_annealing(inst, n_iters=8000, n_restarts=5, seed=0)
            if r_sa.feasible:
                assert r_sa.objective <= r_exact.objective + 1e-9, (
                    f"{family}: SA objective {r_sa.objective} exceeds true "
                    f"optimum {r_exact.objective}"
                )

    def test_reasonable_quality_on_easy_instance(self):
        # independent family has no quadratic risk term and a loose enough
        # budget that SA should find at least a decent-quality feasible
        # answer with enough iterations -- a sanity check the annealer
        # isn't broken (e.g. always returning the random initial state).
        inst = generate_instance("independent", 12, seed=0)
        r_exact = solve_exact_enumeration(inst)
        r_sa = solve_simulated_annealing(inst, n_iters=20000, n_restarts=5, seed=0)
        assert r_sa.feasible
        gap = (r_exact.objective - r_sa.objective) / abs(r_exact.objective)
        assert gap < 0.5, f"SA gap unexpectedly large on an easy instance: {gap}"


class TestSimulatedAnnealingScaling:
    def test_runs_fast_at_exp_a2_settings_n24(self):
        inst = generate_instance("dense", 24, seed=0)
        r = solve_simulated_annealing(inst, n_iters=20000, n_restarts=5, seed=0)
        assert r.runtime_s < 10.0
        assert len(r.selection) == 24
