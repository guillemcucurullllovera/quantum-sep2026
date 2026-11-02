from hardware.backends import BackendCatalog
from hardware.noise import from_fake_backend, synthetic


def test_fake_backend_noise_model_has_nonzero_error_rates():
    r = BackendCatalog.get("fake_heron")
    noise_model, descriptor = from_fake_backend(r.backend)
    assert descriptor.readout_error_mean is not None
    assert descriptor.readout_error_mean > 0
    assert descriptor.gate_error_mean is not None
    assert descriptor.gate_error_mean > 0
    assert descriptor.source == "fake_backend"


def test_synthetic_noise_model_matches_requested_parameters():
    noise_model, descriptor = synthetic(readout_error_prob=0.05)
    assert descriptor.readout_error_mean == 0.05
    assert descriptor.source == "synthetic"


def test_noiseless_and_noisy_simulation_diverge():
    """Regression test for a real bug caught during build: run_sampler's
    FAKE-backend path originally ignored the backend entirely and always
    ran noiseless. This confirms hardware.ibm_runtime.run_sampler actually
    applies device-calibrated noise when given a FAKE backend.
    """
    from qiskit import QuantumCircuit

    from hardware.ibm_runtime import run_sampler

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    r = BackendCatalog.get("fake_heron")
    result = run_sampler(qc, r.descriptor, r.backend, shots=4000)

    # a real device noise model should produce *some* bit-flip leakage
    # outside the ideal {00, 11} support — if it doesn't, the noise model
    # wasn't actually applied.
    leaked = sum(v for k, v in result.counts.items() if k not in ("00", "11"))
    assert leaked > 0
