import numpy as np

from hardware.zne import ZNEMitigator, fold_circuit_gates, linear_extrapolate, richardson_extrapolate


def test_richardson_recovers_exact_value_for_linear_noise_model():
    true_value = 0.8
    noise_factors = [1, 2, 3]
    # linear decay: E(lambda) = true - 0.05*lambda
    noisy = [true_value - 0.05 * lam for lam in noise_factors]
    mitigated = richardson_extrapolate(np.array(noisy), noise_factors)
    assert abs(mitigated - true_value) < 1e-9


def test_richardson_beats_raw_noisiest_measurement():
    true_value = 0.8
    noise_factors = [1, 2, 3]
    noisy = [true_value - 0.05 * lam - 0.01 * lam**2 for lam in noise_factors]
    mitigated = richardson_extrapolate(np.array(noisy), noise_factors)
    noisiest = noisy[-1]
    assert abs(mitigated - true_value) < abs(noisiest - true_value)


def test_linear_extrapolate_matches_two_point_slope():
    result = linear_extrapolate(np.array([1.0, 0.8]), [1, 2])
    assert abs(result - 1.2) < 1e-9  # slope -0.2, extrapolated to lambda=0


def test_zne_mitigator_mitigate_calls_run_fn_once_per_noise_factor():
    calls = []

    def run_fn(scale):
        calls.append(scale)
        return 1.0 - 0.1 * scale

    zne = ZNEMitigator(noise_factors=[1, 3, 5])
    zne.mitigate(run_fn)
    assert calls == [1, 3, 5]


def test_fold_circuit_gates_triples_two_qubit_gate_count_at_scale_3():
    from qiskit import QuantumCircuit
    from hardware.circuit_analysis import analyze

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    folded = fold_circuit_gates(qc, scale_factor=3)
    original_stats = analyze(qc)
    folded_stats = analyze(folded)
    assert folded_stats.two_qubit_gate_count == 3 * original_stats.two_qubit_gate_count
