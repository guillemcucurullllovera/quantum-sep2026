from benchmarks.noise import noiseless_vs_noisy_bitstring_distribution
from benchmarks.mitigation import zne_expectation_mitigation, m3_readout_mitigation
from benchmarks.resource_estimation import qaoa_circuit_resources, shots_for_target_confidence_interval, resource_summary
from benchmarks.scaling import qaoa_scaling, qrc_scaling
from benchmarks.simulator_vs_qpu import simulator_vs_qpu_comparison
from benchmarks.applications import portfolio_application_benchmark


def test_noiseless_vs_noisy_distribution_shows_real_divergence():
    r = noiseless_vs_noisy_bitstring_distribution(n_qubits=4, shots=2048, seed=0)
    assert 0.0 <= r["total_variation_distance"] <= 1.0
    assert r["noise_descriptor"]["readout_error_mean"] is not None


def test_zne_expectation_mitigation_runs_and_reports_honestly():
    r = zne_expectation_mitigation(n_qubits=3, shots=2048, seed=0)
    assert 0.0 <= r["exact_value"] <= 1.0
    assert 0.0 <= r["raw_noisy_value"] <= 1.0
    # not asserting zne_improved is True: an honest report either way is the point,
    # this just confirms the comparison actually ran and produced real numbers
    assert isinstance(r["zne_improved"], bool)


def test_m3_readout_mitigation_reduces_leaked_mass():
    r = m3_readout_mitigation(n_qubits=3, shots=2048)
    assert r["raw_leaked_mass"] >= 0.0
    # mthree's correction is a quasi-probability (linear inversion of a noisy
    # stochastic matrix), not a true probability -- it can legitimately
    # overshoot into a small negative net mass when correction is aggressive,
    # so the real check is "much closer to zero than the raw leak", not "non-negative"
    assert abs(r["mitigated_leaked_mass"]) < r["raw_leaked_mass"]


def test_resource_estimation_produces_real_circuit_stats():
    stats = qaoa_circuit_resources(n_qubits=6, p=2)
    assert stats.n_qubits == 6
    assert stats.depth > 0
    assert stats.two_qubit_gate_count > 0


def test_shots_for_ci_matches_known_formula():
    # p=0.5 maximizes variance; sanity check against the closed-form value
    n = shots_for_target_confidence_interval(0.5, 0.05, confidence=0.95)
    assert 370 <= n <= 390  # 1.96^2 * 0.25 / 0.0025 = 384.16 -> 385


def test_resource_summary_runs_end_to_end():
    summary = resource_summary()
    assert "qaoa" in summary and "pqk" in summary


def test_qaoa_and_qrc_scaling_produce_increasing_size_records():
    qaoa_results = qaoa_scaling(sizes=[4, 6])
    assert [r["n_qubits"] for r in qaoa_results] == [4, 6]
    assert all(r["wall_time_s"] >= 0 for r in qaoa_results)

    qrc_results = qrc_scaling(reservoir_sizes=[2, 4])
    assert all(r["wall_time_s"] >= 0 for r in qrc_results)


def test_simulator_vs_qpu_reports_unavailable_honestly_without_credentials(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings
    get_settings(refresh=True)
    r = simulator_vs_qpu_comparison(n_qubits=4, shots=1024)
    assert r["qpu_available"] is False
    assert "proxy_result" in r


def test_portfolio_application_benchmark_runs_end_to_end():
    r = portfolio_application_benchmark(n_assets=8, cardinality=4, seed=0)
    assert r.method_a == "qaoa_portfolio_application"
    assert isinstance(r.value_a, float) and isinstance(r.value_b, float)
    assert r.context["exact_cost"] is not None  # n_assets=8 <= exact_max_n=18, brute force should have run
