"""
SamplerV2 Wrapper for Bitstring Sampling (Gen2, historical — QRC, QAOA).

Wraps Qiskit Runtime SamplerV2 for components that need bitstring samples:
  QRC-10:  1000 shots, DD + TREX readout error mitigation
  QAOA v2: shots for energy estimation, Pauli twirl + ZNE

EM config follows the per-component matrix in estimator_wrapper.EMConfig:
  QRC:  DD + TREX only (intentional decoherence preserved — no ZNE!)
  QAOA: ZNE [1,2,3] + DD + twirl (128) + TREX

Requires qiskit-ibm-runtime >= 0.21.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .estimator_wrapper import QuantumComponent, EMConfig


class SamplerWrapper:
    """
    SamplerV2 wrapper for bitstring sampling.

    Raises RuntimeError from .run() if qiskit not installed; the
    numpy-fallback sampling methods below are for testing the
    post-processing logic without hardware/QPU access, not a silent
    substitute for a real run.
    """

    def __init__(
        self,
        backend: Any,
        em_config: Optional[EMConfig] = None,
    ):
        self.backend = backend
        self.em_config = em_config or EMConfig.for_component(QuantumComponent.QRC)
        self._qiskit_sampler = None
        self._setup_sampler()

    def _setup_sampler(self) -> None:
        """See EstimatorWrapper._setup_estimator's docstring for why backend
        is checked before attempting real construction."""
        if self.backend is None:
            self._qiskit_sampler = None
            return
        try:
            from qiskit_ibm_runtime import SamplerV2, SamplerOptions
            options = SamplerOptions()
            em = self.em_config

            if em.use_dd:
                options.dynamical_decoupling.enable = True
                options.dynamical_decoupling.sequence_type = em.dd_sequence

            if em.use_twirl:
                options.twirling.enable_gates = True
                options.twirling.num_randomizations = em.n_twirl_circuits

            options.default_shots = em.shots
            self._qiskit_sampler = SamplerV2(mode=self.backend, options=options)
        except ImportError:
            self._qiskit_sampler = None

    @classmethod
    def for_component(
        cls,
        component: QuantumComponent,
        backend: Any,
    ) -> "SamplerWrapper":
        em_config = EMConfig.for_component(component)
        return cls(backend, em_config)

    def run(self, pubs: list) -> Any:
        """Submit sampling PUBs. Raises if qiskit_ibm_runtime unavailable."""
        if self._qiskit_sampler is None:
            raise RuntimeError(
                "qiskit_ibm_runtime not installed.\n"
                "Install: pip install qiskit-ibm-runtime>=0.21\n"
                "For local testing, use the numpy simulation path."
            )
        job = self._qiskit_sampler.run(pubs)
        return job.result()

    def sample_bitstrings(
        self,
        statevector: np.ndarray,
        n_shots: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Numpy fallback: sample bitstrings from a statevector probability
        distribution. Used for local testing without QPU access.

        Args:
            statevector: Complex statevector, shape (2**n,).
            n_shots: Number of samples.
            rng: Random generator.

        Returns:
            counts: Array of shape (2**n,) with bitstring counts.
        """
        if rng is None:
            rng = np.random.default_rng()
        probs = np.abs(statevector) ** 2
        probs /= probs.sum()
        samples = rng.choice(len(statevector), size=n_shots, p=probs)
        counts = np.bincount(samples, minlength=len(statevector))
        return counts

    def bitstrings_to_z_expectations(
        self,
        counts: np.ndarray,
        n: int,
        n_shots: int,
    ) -> np.ndarray:
        """
        Convert bitstring counts to per-qubit Z-expectation values.

        <Z_i> = (counts where bit_i=0 - counts where bit_i=1) / n_shots

        Args:
            counts: Bitstring count array, shape (2**n,).
            n: Number of qubits.
            n_shots: Total number of shots.

        Returns:
            z_exp: Z-expectations, shape (n,), in [-1, 1].
        """
        z_exp = np.zeros(n)
        basis = np.arange(len(counts))
        for i in range(n):
            bits = (basis >> i) & 1
            z_exp[i] = float(np.dot(counts, 1 - 2 * bits)) / n_shots
        return z_exp
