import numpy as np
import pytest

from applications.tail_risk.comparison import compare_methods
from applications.tail_risk.domain import TailRiskSpec
from applications.tail_risk.errors import DegenerateTailError, EmptySampleError


def _fat_tailed_pnl(seed: int = 1, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Student-t with low df: real fat tails, so Cornish-Fisher/historical
    # should diverge materially from the Gaussian closed form.
    return rng.standard_t(df=3, size=n) * 0.01


def test_five_way_cvar_geq_var_classical_only():
    pnl = _fat_tailed_pnl()
    spec = TailRiskSpec(confidence_levels=[0.95], run_quantum=False, seed=3)
    report = compare_methods(pnl, spec)

    assert set(report.results) == {
        "historical", "parametric_gaussian", "cornish_fisher", "monte_carlo_gbm", "meta_agent_bootstrap",
    }
    for name, m in report.results.items():
        assert m.cvar >= m.var - 1e-9, f"{name}: cvar={m.cvar} < var={m.var}"
        assert m.var > 0, f"{name}: expected a positive loss magnitude, got var={m.var}"


def test_six_way_cvar_geq_var_with_quantum():
    # Kept deliberately cheap (small grid/shots/sample) -- this exercises the
    # real QAE bisection path end to end without the cost of a
    # production-resolution grid; algorithms/qae_iqae's own test suite
    # covers accuracy at higher n_grid_qubits.
    pnl = _fat_tailed_pnl(seed=2, n=800)
    spec = TailRiskSpec(
        confidence_levels=[0.95], run_quantum=True, n_grid_qubits=2, epsilon=0.1, shots=256, seed=5,
    )
    report = compare_methods(pnl, spec)

    assert "qae_amplitude_estimation" in report.results
    assert len(report.results) == 6
    # The real invariant: CVaR >= VaR at the same confidence level, checked
    # across every one of the six independently-implemented estimators.
    for name, m in report.results.items():
        assert m.cvar >= m.var - 1e-9, f"{name}: cvar={m.cvar} < var={m.var}"

    qae = report.results["qae_amplitude_estimation"]
    assert qae.extra["n_qae_calls"] >= 1
    assert 0.0 <= qae.extra["prob_at_threshold_ideal"] <= 1.0


def test_cornish_fisher_diverges_from_gaussian_under_fat_tails():
    # Student-t(df=3): real, substantial excess kurtosis (heavy tails)
    # without pushing the Cornish-Fisher polynomial correction into its
    # known breakdown regime (|skew| and excess kurtosis both extreme,
    # e.g. from an aggressive crash-mixture construction, can make the CF
    # expansion non-monotonic -- a real, documented limitation of that
    # estimator, not something this test is trying to exercise).
    rng = np.random.default_rng(1)
    base = rng.standard_t(df=3, size=6000) * 0.01

    spec = TailRiskSpec(confidence_levels=[0.99], run_quantum=False, seed=9)
    report = compare_methods(base, spec)

    gauss_var = report.results["parametric_gaussian"].var
    cf_var = report.results["cornish_fisher"].var
    hist_var = report.results["historical"].var

    # Fat left tail: Cornish-Fisher (whose cubic correction reacts strongly
    # to excess kurtosis) should sit well above the Gaussian estimate;
    # historical (a single empirical percentile, noisier but unbiased)
    # should still exceed it, just by a smaller, less amplified margin.
    assert cf_var > gauss_var * 1.5
    assert hist_var > gauss_var
    assert "cornish_fisher_vs_gaussian_rel_gap" in report.diagnostics
    assert any("Cornish-Fisher" in note for note in report.notes)


def test_prob_of_loss_wired_and_in_range():
    pnl = _fat_tailed_pnl(seed=4)
    report = compare_methods(pnl, TailRiskSpec(run_quantum=False))
    assert 0.0 < report.prob_of_loss < 1.0


def test_empty_sample_raises_typed_error():
    with pytest.raises(EmptySampleError):
        compare_methods(np.array([]), TailRiskSpec(run_quantum=False))


def test_all_positive_pnl_raises_degenerate_tail_error():
    rng = np.random.default_rng(6)
    pnl = np.abs(rng.normal(1.0, 0.1, 500))  # strictly positive: no loss tail
    with pytest.raises(DegenerateTailError):
        compare_methods(pnl, TailRiskSpec(run_quantum=False))


def test_meta_agent_bootstrap_close_to_historical_under_degenerate_regime():
    pnl = _fat_tailed_pnl(seed=8, n=6000)
    report = compare_methods(pnl, TailRiskSpec(run_quantum=False, n_bootstrap=8000, seed=8))
    hist_var = report.results["historical"].var
    meta_var = report.results["meta_agent_bootstrap"].var
    # Degenerate single-regime bootstrap of the same sample should land
    # within bootstrap noise of the direct empirical percentile.
    assert abs(meta_var - hist_var) / hist_var < 0.25
