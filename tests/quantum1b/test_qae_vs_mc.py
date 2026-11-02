"""Tests for `experiments.quantum1b.qae_vs_mc` (Track E, EXP-E2/E3).

The classical-MC-only tests are fast (no quantum circuits) and exercised
thoroughly. The IQAE-involving tests use small trial counts and the loose
noisy epsilon to stay fast -- the actual EXP-E2 (30 ideal trials) and
EXP-E3 (10 noisy trials) runs are done by the dedicated runner scripts,
not by this unit-test suite (same convention as EXP-B3/EXP-D1's own
smoke-test-vs-full-run split elsewhere in this package).
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.qae_iqae.finance_integrands import build_crash_barrier
from experiments.quantum1b.qae_vs_mc import (
    EstimatorSummary,
    QaeVsMcComparisonResult,
    SignificanceTestResult,
    _wilson_ci,
    compare_qae_vs_mc_ideal,
    compare_qae_vs_mc_noisy,
    paired_significance_test,
    run_classical_mc_trials,
)


class TestWilsonCI:
    def test_ci_brackets_point_estimate(self):
        lo, hi = _wilson_ci(50, 100, alpha=0.05)
        assert lo < 0.5 < hi

    def test_ci_narrows_with_more_samples(self):
        lo1, hi1 = _wilson_ci(500, 1000, alpha=0.05)
        lo2, hi2 = _wilson_ci(50, 100, alpha=0.05)
        assert (hi1 - lo1) < (hi2 - lo2)

    def test_ci_bounds_are_valid_probabilities(self):
        for k, n in [(0, 100), (100, 100), (1, 1000)]:
            lo, hi = _wilson_ci(k, n, alpha=0.05)
            assert 0.0 <= lo <= hi <= 1.0


class TestClassicalMCTrials:
    def test_produces_n_trials(self):
        barrier = build_crash_barrier(n=3)
        trials = run_classical_mc_trials(barrier, n_samples=2048, n_trials=10, alpha=0.05, seed=0)
        assert len(trials) == 10

    def test_a_hat_converges_toward_true_value_with_more_samples(self):
        barrier = build_crash_barrier(n=3)
        small = run_classical_mc_trials(barrier, n_samples=64, n_trials=50, alpha=0.05, seed=0)
        large = run_classical_mc_trials(barrier, n_samples=8192, n_trials=50, alpha=0.05, seed=0)
        small_rmse = np.sqrt(np.mean([(t.a_hat - barrier.exact_grid_a) ** 2 for t in small]))
        large_rmse = np.sqrt(np.mean([(t.a_hat - barrier.exact_grid_a) ** 2 for t in large]))
        assert large_rmse < small_rmse

    def test_deterministic_given_same_seed(self):
        barrier = build_crash_barrier(n=3)
        t1 = run_classical_mc_trials(barrier, n_samples=1024, n_trials=5, alpha=0.05, seed=42)
        t2 = run_classical_mc_trials(barrier, n_samples=1024, n_trials=5, alpha=0.05, seed=42)
        assert [t.a_hat for t in t1] == [t.a_hat for t in t2]

    def test_ci_coverage_near_nominal_at_large_n_many_trials(self):
        """A basic sanity check on the Wilson-CI machinery itself: at a
        large sample size and many repeated trials, empirical coverage of
        a 95% CI should be reasonably close to 95% (not a tight
        statistical test, just a sanity bound against a badly broken CI
        formula)."""
        barrier = build_crash_barrier(n=3)
        trials = run_classical_mc_trials(barrier, n_samples=4096, n_trials=200, alpha=0.05, seed=7)
        coverage = np.mean([t.ci_contains_truth for t in trials])
        assert 0.85 <= coverage <= 1.0


class TestCompareQaeVsMcIdeal:
    """Small, bounded smoke test (few trials) -- the full EXP-E2 run (30
    trials) is done by run_track_e_exp_e2.py, not here."""

    def test_produces_comparison_result_with_both_summaries(self):
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_ideal(barrier, n_trials=3, seed=0)

        assert isinstance(result, QaeVsMcComparisonResult)
        assert isinstance(result.iqae, EstimatorSummary)
        assert isinstance(result.classical_mc, EstimatorSummary)
        assert result.iqae.n_trials == 3
        assert result.classical_mc.n_trials == 3
        assert result.backend_label == "ideal"

    def test_iqae_and_mc_target_the_same_true_value(self):
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_ideal(barrier, n_trials=3, seed=0)
        assert result.true_value_numeric == barrier.exact_grid_a

    def test_bias_and_rmse_are_finite(self):
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_ideal(barrier, n_trials=3, seed=0)
        for summary in (result.iqae, result.classical_mc):
            assert np.isfinite(summary.bias)
            assert np.isfinite(summary.rmse)
            assert summary.rmse >= 0.0

    def test_budget_matched_classical_comparator_present_and_matched(self):
        """The ideal path performs real Grover amplification, so the
        unmatched `classical_mc` (at `shots` samples) is NOT a fair
        comparison to IQAE's own (much larger) query count -- this test
        guards that the budget-matched comparator exists and genuinely
        uses the same query count as IQAE's own measured oracle_queries,
        not the unmatched `shots` value."""
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_ideal(barrier, n_trials=5, seed=0)

        assert result.classical_mc_budget_matched is not None
        matched_n = round(result.iqae.mean_oracle_queries)
        assert result.classical_mc_budget_matched.mean_oracle_queries == pytest.approx(matched_n, abs=1)
        # and it must differ from the unmatched comparator's budget,
        # confirming this scenario actually exercises the mismatch this
        # comparator exists to correct (if IQAE ever needed zero
        # amplification at EPSILON_IDEAL, this assumption would need
        # revisiting -- but that would itself be a notable finding, not a
        # silent pass).
        assert result.classical_mc_budget_matched.mean_oracle_queries != result.classical_mc.mean_oracle_queries

    def test_noisy_path_has_no_budget_matched_comparator(self):
        """In the noisy/loose-epsilon path, oracle_queries==0, so a
        budget-matched comparator would be identical to the unmatched one
        -- None signals 'not applicable', not 'forgotten'."""
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_noisy(barrier, n_trials=2, seed=0)
        assert result.classical_mc_budget_matched is None


class TestPairedSignificanceTest:
    def test_detects_a_clear_difference(self):
        """Two estimators with obviously different precision (one tight
        around the truth, one much noisier) must be flagged significant,
        with the tighter one correctly identified as more precise."""
        barrier = build_crash_barrier(n=3)
        true_value = barrier.exact_grid_a

        from experiments.quantum1b.qae_vs_mc import EstimatorSummary, TrialResult

        tight_trials = [
            TrialResult(a_hat=true_value + d, ci_lo=0, ci_hi=1, oracle_queries=100, runtime_s=0.0, ci_contains_truth=True)
            for d in np.linspace(-0.001, 0.001, 20)
        ]
        noisy_trials = [
            TrialResult(a_hat=true_value + d, ci_lo=0, ci_hi=1, oracle_queries=100, runtime_s=0.0, ci_contains_truth=True)
            for d in np.linspace(-0.2, 0.2, 20)
        ]
        tight_summary = EstimatorSummary(
            method="tight", n_trials=20, mean_a_hat=true_value, bias=0.0, variance=0.0, rmse=0.001,
            mean_ci_halfwidth=0.0, empirical_ci_coverage=1.0, mean_oracle_queries=100, mean_runtime_s=0.0,
            trials=tight_trials,
        )
        noisy_summary = EstimatorSummary(
            method="noisy", n_trials=20, mean_a_hat=true_value, bias=0.0, variance=0.01, rmse=0.1,
            mean_ci_halfwidth=0.0, empirical_ci_coverage=1.0, mean_oracle_queries=100, mean_runtime_s=0.0,
            trials=noisy_trials,
        )

        result = paired_significance_test(tight_summary, noisy_summary, true_value)
        assert isinstance(result, SignificanceTestResult)
        assert result.p_value < 0.05
        assert result.a_significantly_more_precise is True
        assert result.median_abs_error_a < result.median_abs_error_b

    def test_no_difference_when_identical_distributions(self):
        """Same underlying distribution (same seed) compared against
        itself must NOT be flagged significant -- a basic sanity check
        that the test isn't biased toward always rejecting the null."""
        barrier = build_crash_barrier(n=3)
        trials = run_classical_mc_trials(barrier, n_samples=2048, n_trials=20, alpha=0.05, seed=0)
        from experiments.quantum1b.qae_vs_mc import EstimatorSummary

        summary = EstimatorSummary(
            method="self", n_trials=20, mean_a_hat=0.0, bias=0.0, variance=0.0, rmse=0.0,
            mean_ci_halfwidth=0.0, empirical_ci_coverage=1.0, mean_oracle_queries=2048, mean_runtime_s=0.0,
            trials=trials,
        )
        result = paired_significance_test(summary, summary, barrier.exact_grid_a)
        assert result.p_value == pytest.approx(1.0, abs=1e-9)  # identical samples -> U test p=1
        assert result.a_significantly_more_precise is False

    def test_integrates_with_real_ideal_comparison(self):
        """End-to-end: run the real (small, bounded) ideal comparison and
        feed its own iqae/classical_mc_budget_matched summaries through
        the significance test -- confirms the two modules' types are
        actually compatible, not just unit-tested in isolation."""
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_ideal(barrier, n_trials=10, seed=0)
        sig = paired_significance_test(result.iqae, result.classical_mc_budget_matched, result.true_value_numeric)
        assert isinstance(sig, SignificanceTestResult)
        assert 0.0 <= sig.p_value <= 1.0


class TestCompareQaeVsMcNoisy:
    """Even smaller/bounded -- FakeKingston noisy runs are the expensive
    path (per Track D's own lesson); this test uses n_trials=2 and the
    module's own loose default epsilon to stay fast and safe."""

    def test_produces_comparison_result(self):
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_noisy(barrier, n_trials=2, seed=0)

        assert result.backend_label == "fake_kingston"
        assert result.iqae.n_trials == 2
        assert result.classical_mc.n_trials == 2

    def test_noisy_epsilon_is_loose_by_default(self):
        """Regression guard: this module's whole noisy-safety argument
        depends on EPSILON_NOISY staying loose enough that IQAE needs zero
        Grover amplification (oracle_queries=0) -- if a future edit
        tightens it, this should fail loudly rather than silently
        reintroducing the Track-D-style depth risk."""
        barrier = build_crash_barrier(n=3)
        result = compare_qae_vs_mc_noisy(barrier, n_trials=2, seed=0)
        assert result.iqae.mean_oracle_queries == 0
