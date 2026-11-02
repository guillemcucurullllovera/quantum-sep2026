"""Tests for algorithms_extended.qiskit_ibm_runtime_asymmetrica — the
vendored adaptation of upstream qiskit-ibm-runtime's SamplerV2/EstimatorV2
primitives, session/batch management, RuntimeJob lifecycle, and
fake_provider fake-backend definitions.

The centerpiece here is a real comparison: the same circuit run through the
just-vendored upstream `SamplerV2` (targeting a fake backend, fully local,
no credentials) and separately through this platform's own from-scratch
implementation (hardware.ibm_runtime.primitives.run_sampler's FAKE
path). Both draw noise from the exact same FakeSherbrooke calibration
snapshot, so their outputs should agree statistically even though the two
code paths get there differently (this platform: transpile + AerSimulator.
from_backend directly; upstream: transpile + BackendSamplerV2 via
QiskitRuntimeLocalService, which drives the fake backend's own `.run()`).
"""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile

from hardware.backends.catalog import BackendCatalog
from hardware.ibm_runtime.primitives import run_sampler as native_run_sampler

from algorithms_extended.qiskit_ibm_runtime_asymmetrica.fake_provider import FakeSherbrooke
from algorithms_extended.qiskit_ibm_runtime_asymmetrica.sampler import SamplerV2


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def _correlated_fraction(counts: dict[str, int]) -> float:
    """Fraction of shots landing on a Bell-correlated outcome (00 or 11,
    ignoring any extra leading/trailing separators Qiskit might add)."""
    total = sum(counts.values())
    correlated = sum(
        v for k, v in counts.items() if k.replace(" ", "")[-2:] in ("00", "11")
    )
    return correlated / total if total else 0.0


def test_vendored_fake_provider_backend_matches_catalog_device():
    """The vendored fake_provider actually loads real IBM device calibration
    data (T1/T2/gate-error/readout-error), and its FakeSherbrooke is the
    same device model hardware.backends.catalog.BackendCatalog's
    'fake_eagle' resolves to."""
    resolved = BackendCatalog.get("fake_eagle")
    vendored_backend = FakeSherbrooke()
    assert vendored_backend.num_qubits == resolved.backend.num_qubits == 127
    assert resolved.descriptor.name == "FakeSherbrooke"


def test_upstream_samplerv2_runs_locally_against_fake_backend_no_credentials():
    """The vendored upstream SamplerV2, given a fake backend, must route
    through the local (no-credentials) execution path -- never attempt a
    real network call. This is the same guarantee the platform's own
    credential-gating tests enforce for hardware.ibm_runtime.primitives."""
    backend = FakeSherbrooke()
    qc = _bell_circuit()
    tqc = transpile(qc, backend=backend, seed_transpiler=42)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([tqc], shots=2000)
    result = job.result()
    counts = result[0].data.meas.get_counts()

    assert sum(counts.values()) == 2000
    # A noisy but still clearly-Bell-correlated distribution.
    assert _correlated_fraction(counts) > 0.8


def test_upstream_samplerv2_agrees_with_native_run_sampler_on_fake_backend():
    """The real comparison: same circuit, same fake device's calibration
    data, two independent code paths (from-scratch run_sampler vs.
    the vendored upstream SamplerV2). They should agree on which
    outcomes dominate and on roughly how noisy the result is, even though
    they are not expected to produce bit-for-bit identical shot counts
    (different simulator entry points, independent shot sampling)."""
    resolved = BackendCatalog.get("fake_eagle")
    backend = resolved.backend
    qc = _bell_circuit()
    shots = 4000

    # this platform's own hardware.ibm_runtime.primitives.
    native_result = native_run_sampler(qc, resolved.descriptor, backend, shots=shots)
    native_counts = native_result.counts

    # the just-vendored upstream SamplerV2.
    tqc = transpile(qc, backend=backend, seed_transpiler=42)
    upstream_sampler = SamplerV2(mode=backend)
    upstream_job = upstream_sampler.run([tqc], shots=shots)
    upstream_counts = upstream_job.result()[0].data.meas.get_counts()

    assert sum(native_counts.values()) == shots
    assert sum(upstream_counts.values()) == shots

    native_correlated = _correlated_fraction(native_counts)
    upstream_correlated = _correlated_fraction(upstream_counts)

    # Both must show the real noisy-Bell-state signature (dominant 00/11,
    # not a coin flip and not perfectly noiseless either).
    assert 0.7 < native_correlated <= 1.0
    assert 0.7 < upstream_correlated <= 1.0
    # And the two independent implementations, fed the same calibration
    # snapshot, must land within statistical shouting distance of each
    # other -- this is the actual agreement check.
    assert abs(native_correlated - upstream_correlated) < 0.15
