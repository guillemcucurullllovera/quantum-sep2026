"""Correctness tests for the shared QUANTUM-1B foundation: instance
schema, QUBO encoding, generators, metrics. Track A/B/C all build on top
of these -- these tests are the load-bearing guarantee that the QUBO
encoding's global optimum actually matches the true constrained problem's
optimum (the exact-slack-encoding property, verified here deterministically
rather than by manual inspection)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments.quantum1b.generators import FAMILIES, generate_instance, generate_ladder
from experiments.quantum1b.instance import (
    CandidateSpec,
    OptionSelectionInstance,
    evaluate_selection,
    no_trade_evaluation,
)
from experiments.quantum1b.metrics import approximation_ratio, evaluate_sample_set, objective_gap
from experiments.quantum1b.qubo_encoding import build_qubo, selection_from_bits
from experiments.quantum1b.registry_schema import load_registry


def _small_instance(**overrides) -> OptionSelectionInstance:
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


class TestInstance:
    def test_no_trade_always_feasible(self):
        for family in FAMILIES:
            inst = generate_instance(family, 8, seed=0)
            ev = no_trade_evaluation(inst)
            assert ev.feasible
            assert ev.objective == 0.0
            assert ev.n_positions == 0

    def test_fingerprint_deterministic_and_order_independent_of_call(self):
        inst = _small_instance()
        assert inst.fingerprint() == inst.fingerprint()

    def test_fingerprint_changes_with_content(self):
        a = _small_instance()
        b = _small_instance(risk_budget=99.0)
        assert a.fingerprint() != b.fingerprint()

    def test_evaluate_selection_flags_budget_violation(self):
        inst = _small_instance()
        ev = evaluate_selection(inst, [1, 1, 1, 1, 1, 1])
        assert not ev.feasible
        kinds = {v.kind for v in ev.violations}
        assert "risk_budget" in kinds

    def test_incompatible_pair_detected(self):
        inst = _small_instance()
        # c0 and c3 are incompatible; select both, nothing else, within budget
        ev = evaluate_selection(inst, [1, 0, 0, 1, 0, 0])
        kinds = {v.kind for v in ev.violations}
        assert "incompatible_pair" in kinds


class TestQUBOEncodingExactness:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_global_qubo_optimum_matches_brute_force(self, family):
        inst = generate_instance(family, 5, seed=1)
        enc = build_qubo(inst)
        n = enc.n_qubits()
        if n > 18:
            pytest.skip(f"n_qubits={n} too large for brute force in this test")

        best_qubo = None
        for bits in itertools.product((0, 1), repeat=n):
            x = np.array(bits, dtype=float)
            val = float(x @ enc.Q @ x) + enc.offset
            if best_qubo is None or val < best_qubo[0]:
                best_qubo = (val, bits)

        qubo_val, qubo_bits = best_qubo
        decoded = selection_from_bits(enc, inst, qubo_bits)
        decoded_eval = evaluate_selection(inst, decoded)

        # Ground truth: brute force over the TRUE constrained objective.
        best_true = None
        for bits5 in itertools.product((0, 1), repeat=inst.n()):
            ev = evaluate_selection(inst, bits5)
            if ev.feasible and (best_true is None or ev.objective > best_true.objective):
                best_true = ev

        assert decoded_eval.feasible, f"QUBO optimum decoded to an infeasible selection: {decoded}"
        assert abs(decoded_eval.objective - best_true.objective) < 1e-6
        assert abs(qubo_val - (-best_true.objective)) < 1e-6

    def test_infeasible_selections_never_beat_best_feasible_in_qubo_space(self):
        inst = _small_instance()
        enc = build_qubo(inst)
        n = enc.n_qubits()
        best_feasible_qubo = None
        best_infeasible_qubo = None
        for bits in itertools.product((0, 1), repeat=n):
            x = np.array(bits, dtype=float)
            val = float(x @ enc.Q @ x) + enc.offset
            decoded = selection_from_bits(enc, inst, bits)
            feasible = evaluate_selection(inst, decoded).feasible
            if feasible:
                if best_feasible_qubo is None or val < best_feasible_qubo:
                    best_feasible_qubo = val
            else:
                if best_infeasible_qubo is None or val < best_infeasible_qubo:
                    best_infeasible_qubo = val
        assert best_infeasible_qubo > best_feasible_qubo

    def test_dropped_candidates_recorded_and_excluded_from_qubo(self):
        inst = _small_instance()
        enc = build_qubo(inst)
        assert set(enc.dropped_candidates) | set(enc.candidate_ids) == {
            c.candidate_id for c in inst.candidates
        }
        for dropped_id in enc.dropped_candidates:
            assert dropped_id not in enc.candidate_ids

    def test_liquidity_floor_drops_low_liquidity_candidates(self):
        inst = _small_instance(liquidity_floor=10.0)  # above every candidate's score
        enc = build_qubo(inst)
        assert len(enc.candidate_ids) == 0
        assert len(enc.dropped_candidates) == 6

    def test_excluded_constraints_reported_when_present(self):
        inst = _small_instance(max_concentration=0.5, max_directional_exposure=2.0)
        enc = build_qubo(inst)
        assert "max_concentration" in enc.excluded_constraints
        assert "max_directional_exposure" in enc.excluded_constraints


class TestGenerators:
    def test_ladder_generates_expected_count(self):
        ladder = generate_ladder(("independent", "correlated"), (4, 6), (0, 1))
        assert len(ladder) == 2 * 2 * 2

    def test_ladder_instances_are_deterministic(self):
        a = generate_instance("correlated", 8, seed=3)
        b = generate_instance("correlated", 8, seed=3)
        assert a.fingerprint() == b.fingerprint()

    def test_ladder_instances_differ_by_seed(self):
        a = generate_instance("correlated", 8, seed=3)
        b = generate_instance("correlated", 8, seed=4)
        assert a.fingerprint() != b.fingerprint()

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            generate_instance("not_a_family", 4, seed=0)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_family_produces_qubit_count_under_ceiling_at_n10(self, family):
        inst = generate_instance(family, 10, seed=0)
        enc = build_qubo(inst)
        assert enc.n_qubits() <= 20  # generous ceiling; EXP-B1 itself caps at n<=10


class TestMetrics:
    def test_objective_gap_zero_at_optimum(self):
        assert objective_gap(10.0, 10.0) == pytest.approx(0.0)

    def test_objective_gap_positive_when_worse(self):
        assert objective_gap(10.0, 8.0) == pytest.approx(0.2)

    def test_objective_gap_none_at_zero_optimum(self):
        assert objective_gap(0.0, 0.0) is None

    def test_approximation_ratio_one_at_optimum(self):
        assert approximation_ratio(5.0, 5.0) == pytest.approx(1.0)

    def test_evaluate_sample_set_cvar_alpha_one_equals_expectation(self):
        inst = generate_instance("independent", 5, seed=0)
        samples = [tuple(b) for b in itertools.product((0, 1), repeat=5)]
        m_full = evaluate_sample_set(inst, samples, cvar_alpha=1.0)
        assert m_full.cvar_objective == pytest.approx(m_full.expected_objective)

    def test_evaluate_sample_set_cvar_alpha_less_than_one_at_least_expectation(self):
        inst = generate_instance("independent", 5, seed=0)
        samples = [tuple(b) for b in itertools.product((0, 1), repeat=5)]
        m = evaluate_sample_set(inst, samples, cvar_alpha=0.2)
        assert m.cvar_objective >= m.expected_objective - 1e-9

    def test_evaluate_sample_set_rejects_empty(self):
        inst = generate_instance("independent", 4, seed=0)
        with pytest.raises(ValueError):
            evaluate_sample_set(inst, [])

    def test_evaluate_sample_set_rejects_bad_alpha(self):
        inst = generate_instance("independent", 4, seed=0)
        with pytest.raises(ValueError):
            evaluate_sample_set(inst, [(0, 0, 0, 0)], cvar_alpha=1.5)


class TestRegistry:
    def test_registry_loads_all_six_experiments(self):
        reg = load_registry()
        assert set(reg.keys()) == {
            "EXP-A1", "EXP-A2", "EXP-B1", "EXP-B2", "EXP-B3", "EXP-C1",
        }

    def test_no_experiment_permits_financial_edge_claim(self):
        reg = load_registry()
        for exp_id, exp in reg.items():
            assert "G" not in exp.evidence_level.split("/") or exp_id.startswith(
                "__never__"
            ), f"{exp_id} must never target evidence level G in QUANTUM-1B"
