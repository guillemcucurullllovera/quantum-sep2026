from hardware.mitigation import PECMitigator
from hardware.mitigation.pec import pec_one_norm


def test_overhead_grows_with_gate_count():
    small = pec_one_norm(0.02, n_two_qubit_gates=5)
    large = pec_one_norm(0.02, n_two_qubit_gates=50)
    assert large > small
    assert small >= 1.0


def test_zero_error_rate_has_no_overhead():
    assert pec_one_norm(0.0, n_two_qubit_gates=100) == 1.0


def test_feasibility_flags_infeasible_when_shots_insufficient():
    mit = PECMitigator(gate_error_rate=0.05, n_two_qubit_gates=100)
    report = mit.check_feasibility(target_std=0.001, max_shots=1000)
    assert report["feasible"] is False
    assert report["required_shots"] > 1000
