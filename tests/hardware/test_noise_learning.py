import numpy as np
import pytest

from hardware.noise_learning import fit_readout_noise_from_counts, fit_depolarizing_rate_from_shadow


def test_fit_readout_noise_perfect_measurement_gives_zero_error():
    # qubit 0: prepared |0> always reads 0, prepared |1> always reads 1 -> zero error
    counts = {0: {"0": 100}, 1: {"1": 100}}
    errors = fit_readout_noise_from_counts({0: counts}, n_qubits=1)
    assert errors[0] == pytest.approx(0.0)


def test_fit_readout_noise_detects_known_error_rate():
    # prepared |0>: 90 correct (read 0), 10 wrong (read 1); prepared |1>: 90 correct, 10 wrong
    counts = {0: {"0": 90, "1": 10}, 1: {"1": 90, "0": 10}}
    errors = fit_readout_noise_from_counts({0: counts}, n_qubits=1)
    assert errors[0] == pytest.approx(0.1, abs=1e-6)


def test_depolarizing_rate_zero_when_purity_matches_theoretical():
    p = fit_depolarizing_rate_from_shadow(measured_purity=1.0, n_qubits=2, theoretical_purity=1.0)
    assert p == pytest.approx(0.0, abs=1e-6)


def test_depolarizing_rate_positive_when_purity_degraded():
    p = fit_depolarizing_rate_from_shadow(measured_purity=0.7, n_qubits=2, theoretical_purity=1.0)
    assert 0.0 < p <= 1.0


def test_depolarizing_rate_rejects_impossible_purity():
    with pytest.raises(ValueError):
        fit_depolarizing_rate_from_shadow(measured_purity=1.5, n_qubits=2, theoretical_purity=1.0)
