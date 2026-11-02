"""Tests for `experiments.quantum1b.classical_competitor_gate` (Track E,
EXP-E1) -- the standing classical-competitor gate generalizing
`algorithms.qae_iqae.erosion_audit`'s existing measurements, per that
module's own `deliverables/FUTURE_DIRECTIONS.md` recommendation.
"""
from __future__ import annotations

import pytest

from experiments.quantum1b.classical_competitor_gate import (
    GateVerdict,
    LOADING_EXPONENT_FAIL_HIGH,
    LOADING_EXPONENT_FAIL_LOW,
    VARIANCE_REDUCTION_FAIL_THRESHOLD,
    ZETA_QMC_FAIL_THRESHOLD,
    evaluate_classical_competitor_gate,
)


class TestGateOnDefaultBook:
    """The default (bull-call-spread / crash-barrier) book is exactly what
    `erosion_audit.py`'s own prior analysis already found erodes AE's
    edge on all three fronts -- this is a reproduction-of-the-known-
    finding sanity check, not a new empirical claim."""

    def test_default_book_fails_the_gate(self):
        result = evaluate_classical_competitor_gate(integrand_label="default_book_reproduction", seed=0)
        assert result.overall_verdict == GateVerdict.FAIL
        assert len(result.checks) == 3

    def test_qmc_check_matches_known_finding(self):
        """erosion_audit.py's own module docstring states zeta_mc~-0.5,
        zeta_qmc~-1 on the smooth spread payoff -- i.e. QMC already
        converges near the AE-matching rate, which this gate's threshold
        (-0.9) should catch as FAIL."""
        result = evaluate_classical_competitor_gate(integrand_label="qmc_check", seed=0)
        qmc_check = next(c for c in result.checks if c.name == "qmc_convergence_edge")
        assert qmc_check.measured_value <= ZETA_QMC_FAIL_THRESHOLD
        assert qmc_check.verdict == GateVerdict.FAIL

    def test_importance_sampling_check_matches_known_finding(self):
        result = evaluate_classical_competitor_gate(integrand_label="is_check", seed=0)
        is_check = next(c for c in result.checks if c.name == "importance_sampling_erosion")
        assert is_check.measured_value >= VARIANCE_REDUCTION_FAIL_THRESHOLD
        assert is_check.verdict == GateVerdict.FAIL

    def test_loading_cost_check_matches_known_finding(self):
        """Both the AE-query count and the classical Grover-Rudolph loading
        cost scale as O(1/eps) (Herbert 2021) -- i.e. both log-log slopes
        vs 1/epsilon should sit near 1.0, which this gate's threshold
        should catch as FAIL (same asymptotic order, so no net saving)."""
        result = evaluate_classical_competitor_gate(integrand_label="loading_check", seed=0)
        loading_check = next(c for c in result.checks if c.name == "grover_rudolph_loading_cost")
        assert LOADING_EXPONENT_FAIL_LOW <= loading_check.measured_value <= LOADING_EXPONENT_FAIL_HIGH
        assert loading_check.verdict == GateVerdict.FAIL

    def test_raw_audit_is_preserved_verbatim(self):
        result = evaluate_classical_competitor_gate(integrand_label="raw_audit_check", seed=0)
        assert "A_qmc_vs_mc" in result.raw_audit
        assert "B_importance_sampling" in result.raw_audit
        assert "C_herbert_loading" in result.raw_audit
        # sanity: the raw numbers are literally what erosion_audit.py produces
        assert "zeta_qmc" in result.raw_audit["A_qmc_vs_mc"]


class TestGateIsDeterministic:
    def test_same_seed_same_verdict(self):
        r1 = evaluate_classical_competitor_gate(integrand_label="det_check", seed=42)
        r2 = evaluate_classical_competitor_gate(integrand_label="det_check", seed=42)
        assert r1.overall_verdict == r2.overall_verdict
        for c1, c2 in zip(r1.checks, r2.checks):
            assert c1.measured_value == pytest.approx(c2.measured_value)


class TestGateCanPassInPrinciple:
    """Construct a synthetic scenario where all three checks WOULD pass --
    proves the gate isn't hardcoded to always FAIL, by using unrealistic
    but mechanically valid inputs that push each measured value to the
    PASS side of its threshold."""

    def test_gate_logic_is_not_hardcoded_to_fail(self):
        from experiments.quantum1b.classical_competitor_gate import (
            _check_importance_sampling,
            _check_loading_cost,
            _check_qmc_edge,
        )

        passing_qmc = _check_qmc_edge({"zeta_qmc": -0.3})  # far from AE rate
        assert passing_qmc.verdict == GateVerdict.PASS

        passing_is = _check_importance_sampling({"variance_reduction_factor": 1.2})  # small IS edge
        assert passing_is.verdict == GateVerdict.PASS

        # classical cost scaling as O(1/eps^0.1) (barely growing) vs AE's
        # O(1/eps) -- wildly different scaling exponents, should PASS.
        passing_loading = _check_loading_cost(
            {
                "rows": [
                    {"epsilon": 1e-2, "ae_oracle_queries_order": 100.0, "grover_rudolph_classical_cdf_ops_order": 10.0},
                    {"epsilon": 1e-3, "ae_oracle_queries_order": 1000.0, "grover_rudolph_classical_cdf_ops_order": 10.5},
                    {"epsilon": 1e-4, "ae_oracle_queries_order": 10000.0, "grover_rudolph_classical_cdf_ops_order": 11.0},
                ]
            },
            epsilon=1e-2,
        )
        assert passing_loading.verdict == GateVerdict.PASS

    def test_summary_is_human_readable(self):
        result = evaluate_classical_competitor_gate(integrand_label="summary_check", seed=0)
        summary = result.summary()
        assert "summary_check" in summary
        assert result.overall_verdict.value in summary
        assert len(summary.splitlines()) == 4  # header + 3 checks
