"""Tests for `experiments.quantum1b.run_track_b_exp_b1.classify_cvar_qaoa_result`.

Regression tests for independent-review finding P2-18 ("EXP-B1's classifier
code does not implement the full registered NO_GO criterion"). Does NOT
re-run the actual statevector sweep (expensive, already covered by Track
B's own committed result and `docs/quantum/QUANTUM1B_CVAR_QAOA.md`) --
`classify_cvar_qaoa_result` was extracted into its own function specifically
so its decision table can be tested directly against synthetic inputs,
including the scenario the review's own suggested reproduction engineered:
a result where CVaR ties/improves on objective gap while feasible
probability collapses toward 0 in most cells, which the code previously
had no path to classify NO_GO for."""
from __future__ import annotations

from experiments.quantum1b.run_track_b_exp_b1 import classify_cvar_qaoa_result


class TestClassifyCvarQaoaResult:
    def test_promising_requires_all_three_positive_conditions(self):
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=True, significant=True,
                majority_feasible=True, majority_feasible_negligible=False,
            )
            == "PROMISING_FOR_FURTHER_TESTING"
        )

    def test_improves_but_not_significant_is_research_only(self):
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=True, significant=False,
                majority_feasible=True, majority_feasible_negligible=False,
            )
            == "RESEARCH_ONLY"
        )

    def test_improves_significant_but_not_majority_feasible_is_research_only(self):
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=True, significant=True,
                majority_feasible=False, majority_feasible_negligible=False,
            )
            == "RESEARCH_ONLY"
        )

    def test_does_not_improve_is_no_go(self):
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=False, significant=False,
                majority_feasible=False, majority_feasible_negligible=False,
            )
            == "NO_GO"
        )

    def test_matches_exp_b1s_actual_real_result(self):
        """EXP-B1's own real, committed result: both variants already hit
        the classical optimum at every tested size (gap=0 both), so
        cvar_improves_or_ties=True (tie) but the Wilcoxon test is
        uncomputable (all-zero diffs) -> significant=False. Real measured
        feasible probabilities were ~0.245-0.256 -- nowhere near the 0.5
        majority bar (majority_feasible=False) and nowhere near negligible
        either (majority_feasible_negligible=False). Classification:
        RESEARCH_ONLY, matching the actual committed result."""
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=True, significant=False,
                majority_feasible=False, majority_feasible_negligible=False,
            )
            == "RESEARCH_ONLY"
        )

    def test_P2_18_the_previously_missing_branch_negligible_feasibility_overrides_improvement(self):
        """The exact gap the review identified: CVaR ties/improves on
        objective gap (cvar_improves_or_ties=True) AND would otherwise
        qualify for PROMISING (significant=True, majority_feasible could
        even be True in a pathological case) -- but feasible probability
        has collapsed to near-zero in most cells. The registered rule's
        third NO_GO clause ("negligible feasible-probability across most
        cells") must win regardless of the other two conditions. Before
        this fix, the code had NO path to reach NO_GO when
        cvar_improves_or_ties was True -- this scenario would have been
        misclassified PROMISING_FOR_FURTHER_TESTING or RESEARCH_ONLY."""
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=True, significant=True,
                majority_feasible=True, majority_feasible_negligible=True,
            )
            == "NO_GO"
        )

    def test_negligible_feasibility_overrides_even_without_improvement(self):
        assert (
            classify_cvar_qaoa_result(
                cvar_improves_or_ties=False, significant=False,
                majority_feasible=False, majority_feasible_negligible=True,
            )
            == "NO_GO"
        )
