"""
Shadow Deployment -- Export Classical Classifiers to Redis

Final step of Phase 6: after QPU training, export the trained
classical shadow (ridge weights + Nystrom features) to Redis for
sub-10ms inference in the production trading system.

The shadow deployment is:
  - READ ONLY from the quantum training environment
  - WRITE ONLY to a shadow Redis namespace (quantum:shadow:*)
  - The live meta-agent picks up the shadow signal as an alpha signal

This module NEVER writes to the live Postgres/Redis production systems
directly. It writes to the shadow namespace and the meta-agent reads it.

Deployment payload per component:
  QRC:  LGBM readout weights + Z-expectation normalization params
  PQK:  Nystrom landmark features + ridge kernel weights
  VQC:  Trained VQC params (frozen) + ridge readout weights + bias
  QAOA: Approximation ratio + warm-start angles (for next optimization)

Redis connectivity is read exclusively
from the REDIS_URL environment variable (never a hardcoded value) --
this is not an IBM Quantum credential, but the same "env-var only, no
fallback" discipline applies. publish_shadow/load_shadow degrade
gracefully (return False/None, log a warning) when REDIS_URL is unset
or redis-py isn't installed -- verified by test, not by inspection.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


SHADOW_NAMESPACE = "quantum:shadow"
INFERENCE_LATENCY_TARGET_MS = 10.0


def _serialize_array(arr: np.ndarray) -> dict:
    """Serialize numpy array for JSON storage."""
    return {
        "__ndarray__": True,
        "data": arr.tolist(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _deserialize_array(obj: dict) -> np.ndarray:
    """Deserialize numpy array from JSON."""
    if not obj.get("__ndarray__"):
        raise ValueError("Not a serialized ndarray")
    return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])


def package_qrc_shadow(
    lgbm_model_bytes: bytes,
    normalization_params: Dict[str, Any],
    metadata: Optional[Dict] = None,
) -> dict:
    """Package QRC shadow deployment payload."""
    return {
        "component": "qrc",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_type": "lgbm",
        "lgbm_model_b64": lgbm_model_bytes.hex() if lgbm_model_bytes else None,
        "normalization": normalization_params,
        "metadata": metadata or {},
    }


def package_vqc_shadow(
    params: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    n_qubits: int,
    d_layers: int,
    metadata: Optional[Dict] = None,
) -> dict:
    """
    Package VQC v5 shadow deployment payload.

    The shadow contains frozen VQC circuit params + ridge readout weights.
    Inference: feature(x) = VQC_circuit(x, params) -> ridge: proba = softmax(Phi.W + b)
    """
    return {
        "component": "vqc_v5",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_type": "statevector_ridge",
        "n_qubits": n_qubits,
        "d_layers": d_layers,
        "params": _serialize_array(params),
        "W": _serialize_array(W),
        "b": _serialize_array(b),
        "metadata": metadata or {},
    }


def package_pqk_shadow(
    landmarks: np.ndarray,
    K_LL_inv_sqrt: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    encoding_params: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
) -> dict:
    """Package PQK v2 shadow deployment payload (Nystrom + ridge)."""
    payload = {
        "component": "pqk_v2",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_type": "nystrom_ridge",
        "landmarks": _serialize_array(landmarks),
        "K_LL_inv_sqrt": _serialize_array(K_LL_inv_sqrt),
        "W": _serialize_array(W),
        "b": _serialize_array(b),
        "metadata": metadata or {},
    }
    if encoding_params is not None:
        payload["encoding_params"] = _serialize_array(encoding_params)
    return payload


def publish_shadow(
    payload: dict,
    redis_url: Optional[str] = None,
    ttl_seconds: int = 86400,
) -> bool:
    """
    Publish a shadow deployment payload to Redis.

    Key format: quantum:shadow:{component}:latest. TTL: 24h (refreshed on
    each training run). redis_url falls back to the REDIS_URL
    environment variable -- never a hardcoded connection string.
    """
    url = redis_url or os.environ.get("REDIS_URL", "")
    if not url:
        logger.warning("REDIS_URL not set -- shadow payload not published")
        return False

    component = payload.get("component", "unknown")
    key = f"{SHADOW_NAMESPACE}:{component}:latest"

    try:
        import redis
        r = redis.Redis.from_url(url, decode_responses=True)
        r.setex(key, ttl_seconds, json.dumps(payload))
        logger.info(f"Published shadow payload to Redis: {key}")
        return True
    except ImportError:
        logger.warning("redis-py not installed -- shadow not published")
        return False
    except Exception as exc:
        logger.warning(f"Redis shadow publish failed: {exc}")
        return False


def load_shadow(
    component: str,
    redis_url: Optional[str] = None,
) -> Optional[dict]:
    """Load a shadow deployment payload from Redis."""
    url = redis_url or os.environ.get("REDIS_URL", "")
    if not url:
        return None

    key = f"{SHADOW_NAMESPACE}:{component}:latest"
    try:
        import redis
        r = redis.Redis.from_url(url, decode_responses=True)
        raw = r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def vqc_shadow_infer(
    x: np.ndarray,
    payload: dict,
) -> np.ndarray:
    """
    Classical VQC shadow inference: compute proba from frozen params + ridge readout.

    This is the <10ms production inference path.
    """
    from algorithms.vqc_qml.v5.ansatz import circuit_features

    params = _deserialize_array(payload["params"])
    W = _deserialize_array(payload["W"])
    b = _deserialize_array(payload["b"])
    n = payload["n_qubits"]
    d = payload["d_layers"]

    phi = circuit_features(x, params, n, d)
    scores = phi @ W + b
    scores = scores - scores.max()
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum()
