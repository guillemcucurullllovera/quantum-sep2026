from qiskit import QuantumCircuit

from hardware.backends import BackendCatalog
from hardware.ibm_runtime import run_sampler


def test_sampler_bell_state_is_correlated():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    r = BackendCatalog.get("noiseless")
    result = run_sampler(qc, r.descriptor, r.backend, shots=2000)
    # a noiseless Bell state should only ever produce 00 or 11
    assert set(result.counts.keys()) <= {"00", "11"}
    assert sum(result.counts.values()) == 2000


def test_sampler_reports_correct_backend_and_shots():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    r = BackendCatalog.get("noiseless")
    result = run_sampler(qc, r.descriptor, r.backend, shots=100)
    assert result.backend_name == "aer_simulator_noiseless"
    assert result.shots == 100
