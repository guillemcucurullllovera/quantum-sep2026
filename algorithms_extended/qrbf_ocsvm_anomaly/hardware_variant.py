"""Noisy-backend variant of the qrbf 1Q-RDM extraction.

NEW (not from the paper, not in the original prototype): the paper's own
Section III.A studies simulator-limit (exact) vs. hardware/noisy behaviour
separately, but the replication in this platform's `kernels.py` only
implements the exact-statevector path. This module adds the noisy path,
using this platform's own `hardware.noise`/`hardware.fake_backends`
infrastructure rather than a bespoke noise model, so the qrbf-OCSVM
technique's noise sensitivity can be measured honestly (and compared) rather
than assumed.

Physics: under noise the state is mixed, so the pure-state RDM shortcut in
``kernels.state_to_1q_rdms`` (``mat.conj().T @ mat``) no longer applies. We
instead simulate to a full density matrix (Aer's ``density_matrix`` method,
with the fake backend's noise model attached) and use
``qiskit.quantum_info.partial_trace`` for a mathematically correct reduced
density matrix under noise.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit_aer import AerSimulator

from algorithms_extended.qrbf_ocsvm_anomaly.kernels import N_QUBITS
from hardware.fake_backends import resolve as resolve_fake_backend
from hardware.noise.models import from_fake_backend


def noisy_1q_rdms(circuit: QuantumCircuit, noisy_backend: AerSimulator) -> np.ndarray:
    """Reduced 1-qubit density matrices under a noise model, shape (n_qubits, 2, 2)."""
    qc = circuit.copy()
    qc.save_density_matrix()
    result = noisy_backend.run(qc).result()
    dm = DensityMatrix(result.data(0)["density_matrix"])

    rdms = np.empty((N_QUBITS, 2, 2), dtype=np.complex128)
    for k in range(N_QUBITS):
        trace_out = [q for q in range(N_QUBITS) if q != k]
        rho_k = partial_trace(dm, trace_out)
        rdms[k] = rho_k.data
    return rdms


def compute_noisy_rdms_for_dataset(
    X: np.ndarray, feature_map, *, backend_family: str = "fake_heron",
) -> tuple[np.ndarray, str]:
    """Noisy counterpart of kernels.compute_rdms_for_dataset. Returns
    (rdms, backend_name_used) so callers can record real provenance."""
    resolved = resolve_fake_backend(backend_family)
    noise_model, noise_descriptor = from_fake_backend(resolved.backend)
    noisy_backend = AerSimulator(method="density_matrix", noise_model=noise_model)

    n_samples = X.shape[0]
    rdms = np.empty((n_samples, N_QUBITS, 2, 2), dtype=np.complex128)
    for i, x in enumerate(X):
        rdms[i] = noisy_1q_rdms(feature_map(x), noisy_backend)
    return rdms, resolved.descriptor.name


def kernel_concentration(rdms: np.ndarray) -> dict:
    """Diagnostic matching REPLICATION_REPORT.md section 3: how concentrated
    (near-1.0) is the off-diagonal similarity structure. Real quantitative
    check of whether noise makes the kernel-concentration failure mode
    better, worse, or unchanged -- not assumed either way."""
    from algorithms_extended.qrbf_ocsvm_anomaly.kernels import qrbf_kernel_matrix

    K = qrbf_kernel_matrix(rdms, rdms, gamma=0.1)
    n = K.shape[0]
    off_diag = K[~np.eye(n, dtype=bool)]
    return {
        "min_off_diag": float(off_diag.min()),
        "max_off_diag": float(off_diag.max()),
        "mean_off_diag": float(off_diag.mean()),
    }
