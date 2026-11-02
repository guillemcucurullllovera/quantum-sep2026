import numpy as np

from applications.tail_risk.var_cvar import assess_tail_risk


def test_cvar_at_least_var_and_matches_core_metrics():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0, 1, 20000)
    report = assess_tail_risk(pnl, alpha=0.95, run_quantum=False)
    assert report.cvar >= report.var
    assert 0.0 < report.prob_of_loss < 1.0


def test_quantum_probability_close_to_empirical():
    # n_grid_qubits=4 (16 bins): fast enough for CI while still resolving the tail
    # of a standard-normal-ish P&L distribution reasonably.
    rng = np.random.default_rng(2)
    pnl = rng.normal(0, 1, 20000)
    report = assess_tail_risk(
        pnl, alpha=0.95, run_quantum=True, n_grid_qubits=4, epsilon=0.02, shots=2048
    )
    assert abs(report.quantum_prob_below_threshold - report.empirical_prob_below_threshold) < 0.08
    assert report.discretisation_gap < 0.05


def test_default_threshold_is_negative_var():
    rng = np.random.default_rng(3)
    pnl = rng.normal(0, 1, 5000)
    report = assess_tail_risk(pnl, alpha=0.95, run_quantum=False)
    assert abs(report.threshold - (-report.var)) < 1e-12
