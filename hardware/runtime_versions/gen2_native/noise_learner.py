"""
NoiseLearner Protocol — QPU Session Noise Calibration (Gen2, historical).

IBM Heron r3 (ibm_boston) T1 varies 108 +/- 31 us over a 2-hour window.
NoiseLearner must run at the START of every QPU Session and the result
cached for the Session duration. PEC's unbiasing coefficients (lambda~0.024
for ECR gates on Heron r3) depend on the current noise model.

Protocol:
  1. At session start: run NoiseLearner -> cache noise_model
  2. At each run: check if cache is fresh (< 2h since calibration)
  3. If stale: re-run NoiseLearner before PEC-mitigation runs
  4. PQK v2 (only component using PEC): requires fresh noise model

This module provides the freshness check and the NoiseLearner runner. It
takes a `backend` object (from backend_factory.get_backend) rather than
reading any credential itself — no env var access here at all.

Requires qiskit_ibm_runtime >= 0.21. Gracefully stubs (returns None) if not
installed, so PEC is skipped but ZNE remains available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

# Freshness threshold: 2 hours (T1 variation is significant beyond this)
SESSION_TTL_SECONDS: int = 7200


@dataclass
class NoiseLearnerCache:
    """Holds a cached NoiseLearner result and its acquisition timestamp."""
    backend_name: str
    noise_model: Any           # qiskit_ibm_runtime NoiseLearnerResult
    timestamp_s: float         # Unix timestamp when calibration was acquired
    ecr_noise_rate: float = 0.024   # lambda for ECR gates (Heron r3 typical value)

    def is_fresh(self, ttl_s: int = SESSION_TTL_SECONDS) -> bool:
        """True if the cache was acquired within the TTL window."""
        return (time.time() - self.timestamp_s) < ttl_s

    def age_minutes(self) -> float:
        return (time.time() - self.timestamp_s) / 60


_CACHE: Optional[NoiseLearnerCache] = None


def run_noise_learner(
    backend: Any,
    n_circuits: int = 200,
    shots_per_circuit: int = 1000,
) -> Optional[Any]:
    """
    Run NoiseLearner on the given backend and return the noise model.

    Updates the global cache. If qiskit_ibm_runtime is not installed,
    returns None (PEC will be skipped, ZNE remains available).

    Args:
        backend: A Qiskit Runtime backend (ibm_boston or FakeHeronV2).
        n_circuits: Number of Pauli noise learning circuits.
        shots_per_circuit: Shots per learning circuit.

    Returns:
        noise_model: NoiseLearnerResult or None.
    """
    global _CACHE
    try:
        from qiskit_ibm_runtime import NoiseLearner, NoiseLearnerOptions
        options = NoiseLearnerOptions()
        options.num_randomizations = n_circuits
        options.shots_per_randomization = shots_per_circuit
        learner = NoiseLearner(mode=backend, options=options)
        job = learner.run()
        noise_model = job.result()
        _CACHE = NoiseLearnerCache(
            backend_name=str(getattr(backend, 'name', 'unknown')),
            noise_model=noise_model,
            timestamp_s=time.time(),
        )
        return noise_model
    except ImportError:
        return None
    except Exception as e:
        # Surface as warning — don't crash; PEC will be skipped
        import warnings
        warnings.warn(f"NoiseLearner failed: {e}. PEC will be skipped.")
        return None


def get_noise_model(
    backend: Any,
    force_refresh: bool = False,
    ttl_s: int = SESSION_TTL_SECONDS,
) -> Optional[Any]:
    """
    Get a fresh noise model, refreshing if stale.

    Args:
        backend: QPU backend (used if re-calibration needed).
        force_refresh: If True, always re-run NoiseLearner.
        ttl_s: Cache TTL in seconds (default 2h).

    Returns:
        noise_model: Fresh NoiseLearnerResult or None.
    """
    global _CACHE
    if force_refresh or _CACHE is None or not _CACHE.is_fresh(ttl_s):
        return run_noise_learner(backend)
    return _CACHE.noise_model


def cache_status() -> dict:
    """Return noise learner cache status for monitoring/dashboards."""
    if _CACHE is None:
        return {"status": "uninitialized", "age_minutes": None, "fresh": False}
    return {
        "status": "cached",
        "backend": _CACHE.backend_name,
        "age_minutes": round(_CACHE.age_minutes(), 1),
        "fresh": _CACHE.is_fresh(),
        "ecr_noise_rate": _CACHE.ecr_noise_rate,
        "ttl_seconds": SESSION_TTL_SECONDS,
    }


def assert_fresh_or_raise(component: str = "PQK_v2") -> None:
    """
    Raise RuntimeError if no fresh noise model is available.

    Called before any PEC-enabled run (PQK v2 only).
    """
    if _CACHE is None:
        raise RuntimeError(
            f"{component} requires PEC, which needs a NoiseLearner result. "
            "Call run_noise_learner(backend) at the start of the QPU session."
        )
    if not _CACHE.is_fresh():
        age = _CACHE.age_minutes()
        raise RuntimeError(
            f"NoiseLearner cache is stale ({age:.1f} min old > "
            f"{SESSION_TTL_SECONDS / 60:.0f} min TTL). "
            "Re-run run_noise_learner(backend) before PEC-enabled circuits."
        )
