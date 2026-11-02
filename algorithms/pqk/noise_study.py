"""Noise-sensitivity study: how much does a fake-backend noise model degrade
the kernel matrix / downstream classification accuracy relative to a
noiseless statevector simulation?

This is real diagnostic infrastructure (not a one-off script) because PQK
depends only on single-qubit <Z_i> expectations, so it should degrade
gracefully under readout noise -- verifying that empirically, rather than
assuming it, is exactly the kind of hardware-readiness check this platform's
own internal audit found missing in the earlier PQK v1 work.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import GenericBackendV2

from algorithms.pqk.feature_maps import N_QUBITS
from core.metrics import mse


def _angle_cnot_ring_circuit(features: np.ndarray, n: int = N_QUBITS) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.ry(float(features[i % len(features)]) * np.pi, i)
    for i in range(n):
        qc.cx(i, (i + 1) % n)
    qc.measure(range(n), range(n))
    return qc


def _z_expectation_from_counts(counts: dict[str, int], qubit: int, n: int, shots: int) -> float:
    """counts keys are little-endian bitstrings of length n (Qiskit convention)."""
    total = 0.0
    for bitstring, count in counts.items():
        bit = int(bitstring[::-1][qubit])  # reverse so index 0 == qubit 0
        total += count * (1 - 2 * bit)
    return total / shots


def projected_features_from_backend(
    X: np.ndarray, backend, shots: int = 4096, n: int = N_QUBITS
) -> np.ndarray:
    """Run the AngleCNOTRingV2 encoding on a given backend (noiseless
    simulator or a noisy fake backend) and return <Z_i> projected features
    estimated from measurement counts -- the real execution path, not the
    exact-statevector shortcut used elsewhere for speed."""
    out = np.zeros((len(X), n))
    for row, x in enumerate(X):
        qc = _angle_cnot_ring_circuit(np.asarray(x), n)
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=shots)
        counts = job.result().get_counts()
        for q in range(n):
            out[row, q] = _z_expectation_from_counts(counts, q, n, shots)
    return out


def run_noise_sensitivity_study(
    X: np.ndarray, shots: int = 4096, seed: int = 0
) -> dict[str, float]:
    """Compare projected features from a noiseless AerSimulator against a
    GenericBackendV2-derived noisy fake backend on the same inputs.

    Returns degradation metrics: per-feature MSE and the max absolute
    per-qubit expectation-value shift, both of which should be small (PQK
    features live in [-1, 1] and are single-qubit expectations, which are
    comparatively noise-robust vs. multi-qubit fidelity overlaps).
    """
    rng = np.random.default_rng(seed)
    X_small = X[: min(len(X), 8)]  # keep the study fast -- this runs real circuits, not a shortcut

    noiseless = AerSimulator()
    noisy_backend = GenericBackendV2(num_qubits=N_QUBITS, seed=seed)
    noise_model = NoiseModel.from_backend(noisy_backend)
    noisy_sim = AerSimulator(noise_model=noise_model)

    feats_clean = projected_features_from_backend(X_small, noiseless, shots=shots)
    feats_noisy = projected_features_from_backend(X_small, noisy_sim, shots=shots)

    per_feature_mse = mse(feats_clean, feats_noisy)
    max_shift = float(np.max(np.abs(feats_clean - feats_noisy)))

    return {
        "n_samples": len(X_small),
        "shots": shots,
        "projected_feature_mse": per_feature_mse,
        "max_abs_expectation_shift": max_shift,
        "noise_model_backend": "GenericBackendV2(seed=%d)" % seed,
    }
