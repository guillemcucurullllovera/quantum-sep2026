from qiskit import QuantumCircuit

from hardware.resource_estimation import CircuitJob, estimate_campaign


def _circuit(depth_reps: int) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    for _ in range(depth_reps):
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
    return qc


def test_campaign_estimate_scales_with_shots_and_depth():
    small = estimate_campaign([CircuitJob("small", _circuit(2), shots=1000)])
    large = estimate_campaign([CircuitJob("large", _circuit(20), shots=1000)])
    assert large.total_circuit_seconds > small.total_circuit_seconds


def test_campaign_estimate_aggregates_multiple_jobs():
    jobs = [CircuitJob(f"job{i}", _circuit(3), shots=500) for i in range(5)]
    est = estimate_campaign(jobs)
    assert est.n_jobs == 5
    assert est.total_shots == 2500
    assert len(est.per_job) == 5
    assert est.total_estimated_seconds > est.total_circuit_seconds  # overhead is additive
