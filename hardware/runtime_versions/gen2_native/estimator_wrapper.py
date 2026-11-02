"""
EstimatorV2 Wrapper with Error Mitigation Config (Gen2, historical).

Wraps Qiskit Runtime EstimatorV2 with the per-component EM stack:
  VQC v5:  ZNE [1,2,3] + DD (XY-4) + Pauli twirl (128 circuits) + TREX
  PQK v2:  ZNE [1,2,3] + PEC (lambda~0.024) + DD + twirl + TREX
  QAOA v2: ZNE [1,2,3] + DD + twirl + TREX
  QRC-10:  DD + TREX only (intentional decoherence preserved)

Requires Qiskit Runtime >= 0.21 (EstimatorV2 API).
All Qiskit imports are guarded; raises a clear RuntimeError (not a silent
no-op) if qiskit_ibm_runtime is absent when .run() is actually called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

import numpy as np


class QuantumComponent(str, Enum):
    QRC = "qrc"
    QAOA = "qaoa_v2"
    PQK = "pqk_v2"
    VQC = "vqc_v5"


@dataclass
class EMConfig:
    """Error mitigation configuration for EstimatorV2."""
    component: QuantumComponent = QuantumComponent.VQC
    zne_noise_factors: list = field(default_factory=lambda: [1, 2, 3])
    use_pec: bool = False
    pec_noise_model: Any = None  # NoiseLearner output; None = skip PEC
    use_dd: bool = True
    dd_sequence: str = "XY4"
    use_twirl: bool = True
    n_twirl_circuits: int = 128
    use_trex: bool = True
    shots: int = 4096

    @classmethod
    def for_component(cls, component: QuantumComponent) -> "EMConfig":
        """Factory: returns the EM config for a given quantum component."""
        if component == QuantumComponent.QRC:
            return cls(
                component=component,
                zne_noise_factors=[],   # No ZNE for QRC
                use_pec=False,
                use_dd=True,
                use_twirl=False,
                use_trex=True,
            )
        elif component == QuantumComponent.QAOA:
            return cls(
                component=component,
                zne_noise_factors=[1, 2, 3],
                use_pec=False,
                use_dd=True,
                use_twirl=True,
                n_twirl_circuits=128,
                use_trex=True,
            )
        elif component == QuantumComponent.PQK:
            return cls(
                component=component,
                zne_noise_factors=[1, 2, 3],
                use_pec=True,           # PEC only for PQK v2 (kernel is noise-sensitive)
                use_dd=True,
                use_twirl=True,
                n_twirl_circuits=128,
                use_trex=True,
            )
        elif component == QuantumComponent.VQC:
            return cls(
                component=component,
                zne_noise_factors=[1, 2, 3],
                use_pec=False,
                use_dd=True,
                use_twirl=True,
                n_twirl_circuits=128,
                use_trex=True,
            )
        raise ValueError(f"Unknown component: {component}")


class EstimatorWrapper:
    """
    EstimatorV2 wrapper for expectation value computation.

    Usage:
      wrapper = EstimatorWrapper.for_component(QuantumComponent.VQC, backend)
      result = wrapper.run([(circuit, observables, params)])

    Raises RuntimeError from .run() if qiskit_ibm_runtime is not installed —
    does not silently fall back to a numpy simulation.
    """

    def __init__(
        self,
        backend: Any,
        em_config: Optional[EMConfig] = None,
        noise_learner_result: Any = None,
    ):
        self.backend = backend
        self.em_config = em_config or EMConfig()
        self.noise_learner_result = noise_learner_result
        self._qiskit_estimator = None
        self._setup_estimator()

    def _setup_estimator(self) -> None:
        """Try to set up the Qiskit Runtime EstimatorV2.

        No backend means no primitive can be constructed regardless of
        whether qiskit_ibm_runtime is installed — short-circuit explicitly
        rather than letting EstimatorV2(mode=None, ...) raise its own
        ValueError, which the bare `except ImportError` below would not
        catch (found by this port's own tests: qiskit_ibm_runtime IS
        installed in this repo's venv, unlike wherever the original tests
        last passed, so the ImportError path alone no longer covers the
        backend=None case used throughout the test suite).
        """
        if self.backend is None:
            self._qiskit_estimator = None
            return
        try:
            from qiskit_ibm_runtime import EstimatorV2, EstimatorOptions
            options = EstimatorOptions()
            em = self.em_config

            if em.zne_noise_factors:
                options.resilience_level = 2  # ZNE
                options.resilience.zne.noise_factors = em.zne_noise_factors
                options.resilience.zne.extrapolator = "richardson"

            if em.use_pec and self.noise_learner_result is not None:
                options.resilience_level = 3  # PEC
                options.resilience.pec.noise_model = self.noise_learner_result

            if em.use_dd:
                options.dynamical_decoupling.enable = True
                options.dynamical_decoupling.sequence_type = em.dd_sequence

            if em.use_twirl:
                options.twirling.enable_gates = True
                options.twirling.num_randomizations = em.n_twirl_circuits

            options.default_shots = em.shots
            self._qiskit_estimator = EstimatorV2(mode=self.backend, options=options)
        except ImportError:
            self._qiskit_estimator = None

    @classmethod
    def for_component(
        cls,
        component: QuantumComponent,
        backend: Any,
        noise_learner_result: Any = None,
    ) -> "EstimatorWrapper":
        """Factory: create wrapper with correct EM config for component."""
        em_config = EMConfig.for_component(component)
        if em_config.use_pec and noise_learner_result is None and backend is not None:
            # Guard: PEC on real hardware requires a fresh NoiseLearner calibration.
            from hardware.runtime_versions.gen2_native.noise_learner import assert_fresh_or_raise
            assert_fresh_or_raise(component.value)
        return cls(backend, em_config, noise_learner_result)

    def run(self, pubs: list) -> Any:
        """
        Submit PUBs (Primitive Unified Blocs) to EstimatorV2.

        Args:
            pubs: List of (circuit, observables, params) tuples.

        Returns:
            PrimitiveResult if Qiskit Runtime available, else raises RuntimeError.
        """
        if self._qiskit_estimator is None:
            raise RuntimeError(
                "qiskit_ibm_runtime not installed. Install with:\n"
                "  pip install qiskit-ibm-runtime>=0.21\n"
                "Or use hardware.ibm_runtime's simulator-backed path for local testing."
            )
        job = self._qiskit_estimator.run(pubs)
        return job.result()

    def em_summary(self) -> dict:
        """Return EM configuration as a dict for reporting/provenance."""
        em = self.em_config
        return {
            "component": em.component.value,
            "zne": em.zne_noise_factors if em.zne_noise_factors else None,
            "pec": em.use_pec,
            "dd_sequence": em.dd_sequence if em.use_dd else None,
            "twirl_circuits": em.n_twirl_circuits if em.use_twirl else None,
            "trex": em.use_trex,
            "shots": em.shots,
        }
