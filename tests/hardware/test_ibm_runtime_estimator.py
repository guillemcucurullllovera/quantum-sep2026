"""Regression test for a real bug caught during build: run_estimator's
FAKE-backend path originally ran an untranspiled circuit against a noise
model keyed by native basis-gate names (sx/x/cz/...), so an h/cx circuit
never touched a noisy instruction and silently came back noiseless.
"""
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from hardware.backends import BackendCatalog
from hardware.ibm_runtime import run_estimator


def test_noisy_estimator_expectation_value_actually_reflects_noise():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp("ZZ")

    r = BackendCatalog.get("fake_heron")
    noisy = run_estimator(qc, [obs], r.descriptor, r.backend)

    r2 = BackendCatalog.get("noiseless")
    clean = run_estimator(qc, [obs], r2.descriptor, r2.backend)

    assert clean.expectation_values[0] == 1.0  # Bell state, ideal <ZZ> = 1
    assert noisy.expectation_values[0] < 1.0  # must show real degradation
    assert noisy.expectation_values[0] > 0.9  # but not absurdly large for one CX
