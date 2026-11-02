from benchmarks.quantum_vs_classical.qaoa import qaoa_vs_classical_small_portfolio, qaoa_vs_simulated_annealing
from benchmarks.quantum_vs_classical.qrc import qrc_vs_esn_forecast
from benchmarks.quantum_vs_classical.pqk import pqk_vs_rbf_classification


def test_qaoa_vs_brute_force_runs_and_reports_real_numbers():
    r = qaoa_vs_classical_small_portfolio(n_assets=6, seed=0)
    assert r.method_a == "qaoa_numpy" and r.method_b == "brute_force_exact"
    assert isinstance(r.value_a, float) and isinstance(r.value_b, float)
    assert r.wall_time_a_s is not None and r.wall_time_a_s >= 0
    # brute force is exact optimum: QAOA's objective can never beat it (minimization)
    assert r.value_a >= r.value_b - 1e-6


def test_qaoa_vs_simulated_annealing_runs():
    r = qaoa_vs_simulated_annealing(n_assets=10, seed=1)
    assert r.value_a == r.value_a  # not NaN
    assert r.value_b == r.value_b


def test_qrc_vs_esn_forecast_runs_and_reports_real_mse():
    r = qrc_vs_esn_forecast(n_samples=150, n_features=4, seed=0)
    assert r.metric_name == "forecast_mse"
    assert r.value_a >= 0 and r.value_b >= 0


def test_pqk_vs_rbf_classification_runs_and_reports_real_accuracy():
    r = pqk_vs_rbf_classification(n_samples=30, seed=0)
    assert 0.0 <= r.value_a <= 1.0
    assert 0.0 <= r.value_b <= 1.0
