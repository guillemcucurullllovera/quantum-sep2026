"""Simulator-vs-QPU comparison entry point.

This environment has no IBM Quantum credentials, so the real-hardware side
of this comparison is structurally unavailable here -- and this module must
say so plainly rather than fabricate a result. What IS genuinely testable in
this environment is used as a documented proxy: noiseless-simulator vs.
noisy-fake-backend-simulator divergence (benchmarks.noise), which is
the same distributional-divergence question QPU-vs-simulator would ask, just
without a real device on the other side.
"""
from __future__ import annotations

from core.config import get_settings
from core.errors import CredentialError
from benchmarks.noise import noiseless_vs_noisy_bitstring_distribution


def simulator_vs_qpu_comparison(n_qubits: int = 4, shots: int = 4096, seed: int = 0) -> dict:
    settings = get_settings()
    if not settings.ibm.is_present:
        proxy = noiseless_vs_noisy_bitstring_distribution(n_qubits=n_qubits, shots=shots, seed=seed)
        return {
            "qpu_available": False,
            "reason": (
                "IBM_QUANTUM_TOKEN not set in this environment -- real-hardware "
                "comparison is structurally unavailable here, not silently skipped. "
                "Reporting the noiseless-vs-noisy-fake-backend divergence as a "
                "documented proxy for the same underlying question."
            ),
            "proxy_result": proxy,
        }

    # Real path: only reachable once credentials exist. Deliberately does not
    # attempt any network call in this environment -- .require() below will
    # raise CredentialError before anything else runs if the token is absent,
    # which is already handled above, but this stays here as the real
    # integration point for whoever adds live-hardware support later.
    settings.ibm.require()
    raise NotImplementedError(
        "IBM_QUANTUM_TOKEN is present, but no real-QPU submission path is "
        "wired here yet -- this is the intended integration point "
        "(hardware.qpu_execution), not implemented in this benchmark module."
    )
