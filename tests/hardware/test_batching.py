from qiskit import QuantumCircuit

from hardware.backends import BackendCatalog
from hardware.batching import run_batch


def _bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_batch_runs_all_valid_circuits():
    r = BackendCatalog.get("noiseless")
    circuits = [_bell(), _bell(), _bell()]
    batch = run_batch(circuits, r.descriptor, r.backend, shots=200)
    assert batch.n_succeeded == 3
    assert batch.n_failed == 0
    assert len(batch.successful_results()) == 3


def test_batch_isolates_one_bad_circuit_without_losing_the_others():
    r = BackendCatalog.get("noiseless")
    bad = "not a circuit"  # will raise when run_sampler tries to use it
    circuits = [_bell(), bad, _bell()]
    batch = run_batch(circuits, r.descriptor, r.backend, shots=200)
    assert batch.n_succeeded == 2
    assert batch.n_failed == 1
    assert batch.items[1].ok is False
    assert batch.items[1].error is not None
    assert batch.items[0].ok and batch.items[2].ok
