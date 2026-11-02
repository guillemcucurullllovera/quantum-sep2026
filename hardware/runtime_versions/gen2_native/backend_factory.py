"""
Injectable Backend Factory — Qiskit Runtime V2 (Gen2, historical).

DESIGN PRINCIPLE: backend topology is an injectable dependency.
The same quantum circuit code runs on three backends — one config change:

    QUANTUM_BACKEND=statevector   -> StatevectorSimulator (no noise, fastest)
    QUANTUM_BACKEND=fake_heron    -> FakeHeronV2 (noise model, pre-QPU testing)
    QUANTUM_BACKEND=ibm_boston    -> live Heron r3 (requires IBM credentials)

Never hardcode a backend name in component code. Always call get_backend().

WHEN IBM ACCESS IS GRANTED:
    1. Set QUANTUM_BACKEND=ibm_boston in server environment
    2. Set IBM_QUANTUM_TOKEN in secrets (never in code) — see .env.example
    3. Pull live coupling map: backend = get_backend(); cm = backend.coupling_map
    4. Re-run the subgraph-selector logic (hardware.qubit_selection) against
       the live coupling map
    5. Re-transpile all ISA circuits — FakeHeron indices != ibm_boston indices

QPU COST ESTIMATES (ibm_boston, 2026, as documented at time of original build):
    Single circuit (~100 gates): ~$0.001
    QRC-10 inference run:        ~$0.10  (100 circuits x 1000 shots)
    PQK v2 Gram matrix (m=200):  ~$20    (200x200/2 pairs x 1000 shots)
    VQC v5 training run:         ~$50    (500 parameter-shift steps)
    Full benchmarking paper:     ~$1,000-$2,000 total

Requires qiskit >= 2.0, qiskit-aer >= 0.14, qiskit-ibm-runtime >= 0.20
(all present in this repo's venv).

CREDENTIAL HANDLING: this port replaces the original's raw `os.getenv(
"IBM_QUANTUM_TOKEN")` with `core.config.get_settings().ibm` per the
platform-wide rule — no code path in this repository may read that
env var directly. See core/config.py.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from core.config import get_settings
from core.errors import CredentialError

_VALID_MODES = ("statevector", "fake_heron", "ibm_boston")


def get_backend(mode: Optional[str] = None) -> Any:
    """
    Return a Qiskit backend for the requested mode.

    Args:
        mode : one of "statevector", "fake_heron", "ibm_boston".
               Defaults to QUANTUM_BACKEND env var, then "statevector".
    Returns:
        A Qiskit backend object (AerSimulator or IBMBackend).
    Raises:
        ImportError      : if qiskit / qiskit-aer / qiskit-ibm-runtime not installed
        ValueError       : if mode is unrecognised
        CredentialError  : if ibm_boston mode is requested but IBM token is missing
                            (raised by core.config.IBMQuantumCredentials.require())
    """
    if mode is None:
        mode = os.getenv("QUANTUM_BACKEND", "statevector").lower()

    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown backend mode '{mode}'. "
            f"Valid options: {_VALID_MODES}"
        )

    if mode == "statevector":
        return _get_statevector()
    elif mode == "fake_heron":
        return _get_fake_heron()
    elif mode == "ibm_boston":
        return _get_ibm_boston()


def _get_statevector():
    """AerSimulator in statevector mode — exact, no noise, fastest."""
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError(
            "statevector backend requires qiskit-aer. "
            "Install with: pip install qiskit-aer"
        )
    return AerSimulator(method="statevector")


def _get_fake_heron():
    """
    FakeHeronV2 — Heron r3 noise model for pre-QPU validation.

    Uses IBM's published noise characterization for ibm_boston (Heron r3):
    - T1 ~ 108us (varies +/-31us over 2h — use NoiseLearner at Session start)
    - T2 ~ 120us
    - Gate error (ECR): ~ 0.3% average
    - Readout error: ~ 1.5%

    NOTE: FakeHeron coupling map ~ ibm_boston but not identical.
    Qubit subgraph indices MUST be recalculated for live ibm_boston.
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeHeronV2 as _FakeHeronV2
        return _FakeHeronV2()
    except ImportError:
        pass

    # Fall back to qiskit.providers.fake_provider (older qiskit versions)
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        # 156-qubit generic backend approximating Heron r3
        return GenericBackendV2(num_qubits=156, basis_gates=["ecr", "id", "rz", "sx", "x"])
    except ImportError:
        raise ImportError(
            "fake_heron backend requires qiskit-ibm-runtime >= 0.20 or qiskit >= 2.0. "
            "Install with: pip install qiskit qiskit-ibm-runtime"
        )


def _get_ibm_boston():
    """
    Live Heron r3 backend (ibm_boston, 156q, tunable couplers).

    REQUIRES:
        - IBM Quantum account with ibm_boston access
        - IBM_QUANTUM_TOKEN set in environment (read via core.config only,
          never hardcoded, never read via raw os.environ in this file)
        - qiskit-ibm-runtime installed

    After getting the backend, always run NoiseLearner before circuit
    submission — see noise_learner.run_noise_learner().
    """
    creds = get_settings().ibm.require()  # raises CredentialError if absent

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "ibm_boston backend requires qiskit-ibm-runtime. "
            "Install with: pip install qiskit-ibm-runtime"
        )

    service = QiskitRuntimeService(
        channel=creds.channel,
        token=creds.token,
        instance=creds.instance,
    )
    backend = service.backend("ibm_boston")
    return backend


def backend_info(backend: Any) -> dict:
    """
    Return a summary dict for the given backend.

    Safe to call without IBM credentials (uses fake backend attributes).
    """
    name = getattr(backend, "name", str(backend))
    try:
        n_qubits = backend.num_qubits
    except AttributeError:
        n_qubits = None
    try:
        coupling = backend.coupling_map
        n_edges = len(list(coupling.get_edges())) if hasattr(coupling, "get_edges") else None
    except (AttributeError, TypeError):
        coupling = None
        n_edges = None

    return {
        "name": name,
        "n_qubits": n_qubits,
        "n_coupling_edges": n_edges,
        "mode": os.getenv("QUANTUM_BACKEND", "statevector"),
    }


__all__ = ["get_backend", "backend_info"]
