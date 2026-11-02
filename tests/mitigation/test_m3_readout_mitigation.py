from hardware.backends import BackendCatalog
from hardware.readout_mitigation import M3Mitigator


def test_m3_mitigation_actually_changes_noisy_counts():
    r = BackendCatalog.get("fake_heron")
    qubits = [0, 1, 2]
    mit = M3Mitigator(r.backend, qubits=qubits)
    mit.calibrate()

    # a deliberately "corrupted" counts dict (as if measured on noisy hardware)
    raw_counts = {"000": 800, "001": 100, "010": 60, "100": 40}
    result = mit.apply(raw_counts)

    assert result.raw_counts == raw_counts
    assert result.mapping == qubits
    # mitigation must have transformed the distribution, not returned it verbatim
    raw_frac_000 = raw_counts["000"] / sum(raw_counts.values())
    mitigated_000 = result.mitigated_quasi_probabilities.get("000", 0.0)
    assert mitigated_000 != raw_frac_000
