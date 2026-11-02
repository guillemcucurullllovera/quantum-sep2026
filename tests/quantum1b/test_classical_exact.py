"""Correctness tests for Track A's exact classical comparator
(`experiments/quantum1b/classical_exact.py`).

The centerpiece is `TestExactnessCrossCheck` -- the actual pytest form of
EXP-A1 (docs/quantum/QUANTUM1B_PREREGISTRATION.md): exhaustive enumeration
(ground truth) must agree with the Fortet-linearized MILP solve on every
synthetic family, exactly, within tolerance 1e-6. This test set uses a
reduced n range (<=16, one seed per (family, n) cell) for CI speed; the
FULL preregistered ladder (n up to 18, seeds 0-4) is executed by
`experiments/quantum1b/run_track_a.py` as the actual registered EXP-A1 run,
with results written to `experiments/quantum1b/results/track_a_results.json`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.quantum1b.classical_exact import (
    ClassicalSolveResult,
    MILPTimeoutError,
    solve_exact_enumeration,
    solve_exact_milp,
)
from experiments.quantum1b.generators import FAMILIES, generate_instance
from experiments.quantum1b.instance import (
    CandidateSpec,
    OptionSelectionInstance,
    evaluate_selection,
)


def _naive_enumeration_objective(instance: OptionSelectionInstance) -> tuple[float, bool]:
    """Literal per-bitstring `evaluate_selection` loop -- the un-vectorized
    reference this module's fast vectorized search must match exactly."""
    best = None
    for bits in itertools.product((0, 1), repeat=instance.n()):
        ev = evaluate_selection(instance, bits)
        if ev.feasible and (best is None or ev.objective > best.objective):
            best = ev
    assert best is not None  # NO_TRADE always feasible
    return best.objective, best.feasible


class TestVectorizedEnumerationMatchesNaive:
    """The vectorized numpy search inside `solve_exact_enumeration` must
    agree bit-for-bit (in objective/feasibility) with a literal
    per-bitstring `evaluate_selection` loop -- this is what lets
    `solve_exact_enumeration` stay fast (numpy) while still being "ground
    truth via evaluate_selection" in substance (see module docstring)."""

    @pytest.mark.parametrize("family", FAMILIES)
    def test_matches_naive_loop(self, family):
        inst = generate_instance(family, 8, seed=2)
        fast = solve_exact_enumeration(inst)
        naive_obj, naive_feasible = _naive_enumeration_objective(inst)
        assert fast.feasible == naive_feasible
        assert fast.objective == pytest.approx(naive_obj, abs=1e-9)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_matches_naive_loop_small_n(self, family):
        inst = generate_instance(family, 6, seed=5)
        fast = solve_exact_enumeration(inst)
        naive_obj, naive_feasible = _naive_enumeration_objective(inst)
        assert fast.feasible == naive_feasible
        assert fast.objective == pytest.approx(naive_obj, abs=1e-9)

    def test_result_selection_is_re_scored_through_evaluate_selection(self):
        # The returned selection/objective/feasible must be exactly what
        # evaluate_selection reports on that same selection -- never a
        # hand-rolled number that could silently drift from it.
        inst = generate_instance("dense", 8, seed=0)
        result = solve_exact_enumeration(inst)
        ev = evaluate_selection(inst, result.selection)
        assert ev.objective == pytest.approx(result.objective, abs=1e-12)
        assert ev.feasible == result.feasible

    def test_raises_above_max_vars(self):
        inst = generate_instance("dense", 8, seed=0)
        with pytest.raises(ValueError):
            solve_exact_enumeration(inst, max_vars=4)


class TestExactnessCrossCheck:
    """EXP-A1 as an executable test: exhaustive enumeration vs the
    Fortet-linearized MILP must agree exactly on every family, within
    tolerance 1e-6, for both objective value and feasibility. Reduced n
    range (<=16) and one seed per cell for test-suite speed; the full
    preregistered n<=18 / seeds 0-4 ladder runs in run_track_a.py."""

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
    def test_enumeration_milp_agreement(self, family, n):
        inst = generate_instance(family, n, seed=1)
        r_enum = solve_exact_enumeration(inst)
        r_milp = solve_exact_milp(inst)
        assert r_enum.feasible == r_milp.feasible, (
            f"{family} n={n}: feasibility mismatch enum={r_enum.feasible} "
            f"milp={r_milp.feasible}"
        )
        assert r_enum.objective == pytest.approx(r_milp.objective, abs=1e-6), (
            f"{family} n={n}: objective mismatch enum={r_enum.objective} "
            f"milp={r_milp.objective}"
        )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_enumeration_milp_agreement_n16(self, family):
        # One larger size, still within enumeration's practical ceiling,
        # to specifically exercise the Fortet auxiliary-variable path at a
        # size closer to EXP-B1's own ladder ceiling.
        inst = generate_instance(family, 16, seed=1)
        r_enum = solve_exact_enumeration(inst)
        r_milp = solve_exact_milp(inst)
        assert r_enum.feasible == r_milp.feasible
        assert r_enum.objective == pytest.approx(r_milp.objective, abs=1e-6)

    def test_multiple_seeds_agree(self):
        for seed in range(5):
            inst = generate_instance("correlated", 8, seed=seed)
            r_enum = solve_exact_enumeration(inst)
            r_milp = solve_exact_milp(inst)
            assert r_enum.feasible == r_milp.feasible
            assert r_enum.objective == pytest.approx(r_milp.objective, abs=1e-6)


class TestMILPConstraintSurface:
    """Targeted checks that solve_exact_milp actually encodes every field
    on OptionSelectionInstance as a real constraint (not silently ignoring
    any of them, unlike qubo_encoding.build_qubo's disclosed exclusions)."""

    def _instance(self, **overrides) -> OptionSelectionInstance:
        cands = tuple(
            CandidateSpec(
                candidate_id=f"c{i}",
                underlying=["SPY", "QQQ", "TLT"][i % 3],
                expected_utility=1.0 + 0.3 * i,
                premium_cost=round(1.0 + 0.25 * i, 2),
                max_loss=5.0,
                liquidity_score=0.5 + 0.05 * i,
                direction=["long", "short", "neutral"][i % 3],
                incompatible_with=("c0",) if i == 3 else (),
            )
            for i in range(6)
        )
        kwargs = dict(
            instance_id="t1",
            candidates=cands,
            risk_budget=3.5,
            family="test",
            seed=0,
            max_positions=3,
            liquidity_floor=0.4,
            risk_aversion=0.5,
            pairwise_risk=(("c0", "c1", 1.0), ("c1", "c2", 0.5)),
        )
        kwargs.update(overrides)
        return OptionSelectionInstance(**kwargs)

    def test_max_per_underlying_respected(self):
        inst = self._instance(max_per_underlying={"SPY": 0})
        r = solve_exact_milp(inst)
        assert r.feasible
        idx = inst.candidate_index()
        for c in inst.candidates:
            if c.underlying == "SPY":
                assert r.selection[idx[c.candidate_id]] == 0

    def test_max_concentration_respected(self):
        inst = self._instance(max_concentration=0.01, max_per_underlying=None)
        r = solve_exact_milp(inst)
        ev = evaluate_selection(inst, r.selection)
        assert ev.feasible
        assert r.feasible

    def test_max_directional_exposure_respected(self):
        inst = self._instance(max_directional_exposure=0.0, max_per_underlying=None)
        r = solve_exact_milp(inst)
        ev = evaluate_selection(inst, r.selection)
        assert ev.feasible

    def test_incompatible_pair_never_both_selected(self):
        inst = self._instance(max_per_underlying=None)
        r = solve_exact_milp(inst)
        idx = inst.candidate_index()
        assert not (r.selection[idx["c0"]] == 1 and r.selection[idx["c3"]] == 1)

    def test_liquidity_floor_drops_candidates_from_milp(self):
        inst = self._instance(liquidity_floor=10.0, max_per_underlying=None)
        r = solve_exact_milp(inst)
        assert r.feasible
        assert sum(r.selection) == 0
        assert r.objective == 0.0

    def test_matches_enumeration_with_full_constraint_surface(self):
        inst = self._instance(
            max_per_underlying={"SPY": 1, "QQQ": 1, "TLT": 1},
            max_concentration=0.9,
            max_directional_exposure=5.0,
        )
        r_enum = solve_exact_enumeration(inst)
        r_milp = solve_exact_milp(inst)
        assert r_enum.feasible == r_milp.feasible
        assert r_enum.objective == pytest.approx(r_milp.objective, abs=1e-6)

    def test_no_pairwise_risk_no_auxiliary_variables(self):
        inst = self._instance(risk_aversion=0.0, pairwise_risk=())
        r = solve_exact_milp(inst)
        assert r.solver_info["n_y_auxiliary"] == 0

    def test_all_candidates_filtered_gives_no_trade(self):
        inst = self._instance(liquidity_floor=999.0)
        r = solve_exact_milp(inst)
        assert r.feasible
        assert r.objective == 0.0
        assert sum(r.selection) == 0
        assert r.solver_info["status"] == "trivial_all_candidates_liquidity_filtered"


class TestMILPTimeout:
    def test_absurdly_small_time_limit_raises_milp_timeout_error(self):
        inst = generate_instance("dense", 20, seed=0)
        with pytest.raises(MILPTimeoutError) as excinfo:
            solve_exact_milp(inst, time_limit_s=1e-9)
        assert excinfo.value.runtime_s >= 0.0
        assert "time_limit_s" in excinfo.value.solver_info


class TestSolverInfoRecordsVersion:
    def test_scipy_version_recorded(self):
        inst = generate_instance("independent", 6, seed=0)
        r = solve_exact_milp(inst)
        assert "scipy_version" in r.solver_info
        assert r.solver_info["solver"] == "scipy.optimize.milp (HiGHS)"

    def test_result_dataclass_shape(self):
        inst = generate_instance("independent", 6, seed=0)
        r = solve_exact_milp(inst)
        assert isinstance(r, ClassicalSolveResult)
        assert len(r.selection) == inst.n()
        assert isinstance(r.feasible, bool)
        assert r.method == "milp_highs"
        assert r.runtime_s >= 0.0
