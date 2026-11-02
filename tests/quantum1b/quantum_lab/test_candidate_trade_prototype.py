"""Tests for `experiments.quantum1b.quantum_lab.candidate_trade_prototype`.

These tests genuinely run the pipeline (small instance, cheap relative to
Track B/D's own sized sweeps, but NOT free -- the noisy FakeKingston stage
alone takes roughly a minute, matching EXP-D1's own measured cost at this
instance size) rather than mocking it -- the whole point of this module is
"does the end-to-end pipeline actually work," which a mocked test cannot
demonstrate. `run()` is called exactly ONCE for the whole module via a
module-scoped fixture, not once per test -- repeating the noisy COBYLA
optimization per assertion would make this file take ~10x longer for no
added coverage."""
from __future__ import annotations

import json

import pytest

from experiments.quantum1b.hardware_readiness.frozen_controls import FROZEN_CONTROLS
from experiments.quantum1b.quantum_lab.candidate_trade_prototype import (
    OUTPUT_PATH,
    CandidateTradeE2EReport,
    main,
    run,
)


@pytest.fixture(scope="module")
def report() -> CandidateTradeE2EReport:
    return run()


class TestRun:
    def test_run_returns_a_report(self, report):
        assert isinstance(report, CandidateTradeE2EReport)

    def test_matches_the_frozen_controls_instance_by_default(self, report):
        """The default instance must be literally the same one
        FROZEN_CONTROLS[0] pins -- confirmed by fingerprint, not just by
        n_candidates/family/seed matching coincidentally."""
        assert report.instance_fingerprint == FROZEN_CONTROLS[0].expected_fingerprint

    def test_classical_optimum_matches_the_frozen_controls_pinned_value(self, report):
        assert report.classical_exact_optimum == FROZEN_CONTROLS[0].expected_exact_optimum

    def test_no_financial_edge_or_quantum_advantage_claimed(self, report):
        assert report.financial_edge_claimed is False
        assert report.quantum_advantage_claimed is False

    def test_evidence_levels_are_populated_and_distinguish_ideal_from_noisy(self, report):
        assert report.classical_evidence_level
        assert report.ideal_expectation_qaoa_evidence_level
        assert report.ideal_cvar_qaoa_evidence_level
        assert report.noisy_expectation_qaoa_evidence_level
        # The noisy (FakeKingston) stage must report a DIFFERENT, weaker
        # evidence level than the ideal stages -- collapsing this
        # distinction would hide exactly the ideal-vs-noisy gap this
        # pipeline exists to surface.
        assert report.noisy_expectation_qaoa_evidence_level != report.ideal_expectation_qaoa_evidence_level

    def test_noisy_stage_reports_a_deeper_circuit_than_zero(self, report):
        assert report.noisy_expectation_qaoa_depth > 0
        assert report.noisy_expectation_qaoa_two_qubit_gates > 0

    def test_gaps_are_computed_and_non_negative_at_this_easy_instance(self, report):
        """At this tiny instance, every prior Track B/D finding shows every
        method reaching the exact optimum (gap=0) -- a non-zero NEGATIVE gap
        (quantum beating the classical exact optimum) would indicate a bug,
        since the exact solver is definitionally optimal."""
        for gap in (
            report.classical_vs_ideal_expectation_gap,
            report.classical_vs_ideal_cvar_gap,
            report.classical_vs_noisy_gap,
        ):
            assert gap is not None
            assert gap >= -1e-9

    def test_rejects_an_instance_not_in_frozen_control_set(self):
        with pytest.raises(AssertionError):
            run(n_candidates=999, family="independent", seed=0)


class TestMainWritesOutput:
    def test_main_writes_valid_json_to_the_declared_output_path(self):
        main()
        assert OUTPUT_PATH.is_file()
        assert OUTPUT_PATH.parent.name == "results"
        assert OUTPUT_PATH.parent.parent.name == "quantum1b"  # regression guard: not experiments/results/
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        assert data["financial_edge_claimed"] is False
        assert "classical_exact_optimum" in data
