import numpy as np

from algorithms_extended.error_mitigation.readout_mitigation import (
    CalibrationMatrix,
    LocalReadoutMitigator,
    clip_to_valid_distribution,
)


def _confusion_matrix(p_1given0: float, p_0given1: float) -> np.ndarray:
    return np.array([[1 - p_1given0, p_0given1], [p_1given0, 1 - p_0given1]])


def test_mitigation_recovers_ideal_distribution_for_two_qubits():
    # two qubits with different, known asymmetric bit-flip error rates
    A0 = _confusion_matrix(p_1given0=0.06, p_0given1=0.10)
    A1 = _confusion_matrix(p_1given0=0.03, p_0given1=0.08)
    calib = CalibrationMatrix(per_qubit=[A0, A1])

    # ideal (noiseless) distribution: a perfect Bell-like split over |00> and |11>
    ideal = np.array([0.5, 0.0, 0.0, 0.5])

    # simulate what the noisy measured distribution would be
    full_A = calib.full_matrix()
    noisy = full_A @ ideal
    noisy_counts = {format(i, "02b"): int(round(p * 100000)) for i, p in enumerate(noisy)}

    mitigator = LocalReadoutMitigator(calib)
    mitigated = mitigator.mitigate_counts(noisy_counts)

    noisy_probs = {format(i, "02b"): noisy[i] for i in range(4)}
    ideal_probs = {format(i, "02b"): ideal[i] for i in range(4)}

    err_before = sum(abs(noisy_probs[k] - ideal_probs[k]) for k in ideal_probs)
    err_after = sum(abs(mitigated[k] - ideal_probs[k]) for k in ideal_probs)

    assert err_after < err_before
    assert err_after < 0.01  # near-exact recovery since calibration matches the injected noise exactly


def test_mitigated_expectation_value():
    A0 = _confusion_matrix(p_1given0=0.05, p_0given1=0.05)
    calib = CalibrationMatrix(per_qubit=[A0])
    mitigator = LocalReadoutMitigator(calib)

    ideal = np.array([1.0, 0.0])  # always measures |0>
    full_A = calib.full_matrix()
    noisy = full_A @ ideal
    noisy_counts = {format(i, "01b"): int(round(p * 100000)) for i, p in enumerate(noisy)}

    observable = {"0": 1.0, "1": -1.0}  # Z observable
    raw_expectation = sum((noisy_counts[b] / sum(noisy_counts.values())) * v for b, v in observable.items())
    mitigated_expectation = mitigator.mitigated_expectation(noisy_counts, observable)

    assert abs(mitigated_expectation - 1.0) < abs(raw_expectation - 1.0)
    assert abs(mitigated_expectation - 1.0) < 0.01


def test_clip_to_valid_distribution_renormalizes():
    quasi = {"00": 0.55, "01": -0.05, "10": 0.4, "11": 0.1}
    clipped = clip_to_valid_distribution(quasi)
    assert all(v >= 0 for v in clipped.values())
    assert abs(sum(clipped.values()) - 1.0) < 1e-9
