"""
Shadow Extractor — QPU Feature Extraction for Shadow Deployment

Phase 6: Train quantum models on QPU, deploy classically to Redis
for <10ms inference latency. This module extracts quantum features
on the QPU (or simulator) and caches them for classical use.

Shadow deployment flow:
  1. shadow_extractor: Run QRC/PQK on QPU -> extract feature vectors
  2. shadow_kernel: Cache PQK kernel values (Nystrom landmarks)
  3. shadow_deploy: Export classical classifiers to Redis (<10ms inference)

The classical shadow means: once quantum training is complete, the
trained readout weights + Nystrom landmark features are all that's needed
for inference -- no QPU required at prediction time.

This is the path to production: QPU trains, CPU deploys.

"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_qrc_features(
    X: np.ndarray,
    n1: int = 5,
    n2: int = 5,
    reservoir_alpha: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract QRC reservoir features for a batch of samples.

    Runs the IsingXX reservoir for each sample in X and returns
    the Z-expectation feature vectors.

    Args:
        X: Input matrix, shape (N, F). First n1 features used.
        n1: Input register qubits.
        n2: Reservoir qubits.
        reservoir_alpha: Spectral radius for eigenvalue normalization.
        seed: RNG seed for reservoir initialization.

    Returns:
        Phi: Feature matrix, shape (N, n1+n2).
    """
    from algorithms.qrc.gen2_v1_original.qrc_v1 import QRCClassifier, QRCConfig
    from algorithms.qrc.gen2_v1_original.qrc_encoder import encode_inputs

    cfg = QRCConfig(n1=n1, n2=n2, alpha=reservoir_alpha, random_seed=seed)
    clf = QRCClassifier(cfg)
    clf._build_reservoir()
    n_total = n1 + n2
    Phi = np.zeros((len(X), n_total))

    for i, x in enumerate(X):
        angles = encode_inputs(x, n1=n1)                              # (n1,) in [-pi, pi]
        X_window = np.tile(angles.reshape(1, n1), (cfg.n_windows, 1))  # (n_windows, n1)
        Phi[i] = clf._extract_features(X_window)[0]

    return Phi


def extract_pqk_features(
    X: np.ndarray,
    n: int = 5,
    encoding_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract PQK (Projected Quantum Kernel) features via Nystrom approximation.

    Returns the Nystrom feature map phi(X) for classical kernel inference.

    Args:
        X: Input matrix, shape (N, F).
        n: Number of qubits.
        encoding_params: Optional learned encoding parameters.

    Returns:
        Phi: Nystrom feature matrix, shape (N, m) where m = landmark count.
    """
    from algorithms.pqk.pqk_v2 import PQKv2

    pqk = PQKv2(n_qubits=n)
    if not hasattr(pqk, 'nystrom_') or pqk.nystrom_ is None:
        raise RuntimeError(
            "PQK v2 must be fitted (KTA ascent + Nystrom) before feature extraction. "
            "Call pqk.fit(X_train, y_train) first."
        )
    return pqk.nystrom_.transform(X)


def shadow_dataset(
    X: np.ndarray,
    y: np.ndarray,
    component: str = "qrc",
    seed: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract quantum features for a full dataset and package for shadow deployment.

    Returns a dict containing:
      - Phi: Feature matrix (N, d)
      - y: Labels (N,)
      - component: Which quantum component was used
      - timestamp: When features were extracted
      - n_samples: N
      - feature_dim: d
      - data_hash: SHA-256 of X for audit trail

    Args:
        X: Input data, shape (N, F).
        y: Labels, shape (N,).
        component: "qrc" or "pqk".
        seed: RNG seed.
        **kwargs: Passed to component-specific extractor.

    Returns:
        Shadow dataset dict.
    """
    if component == "qrc":
        Phi = extract_qrc_features(X, seed=seed, **kwargs)
    elif component == "pqk":
        Phi = extract_pqk_features(X, **kwargs)
    else:
        raise ValueError(f"Unknown component: {component}. Use 'qrc' or 'pqk'.")

    data_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    return {
        "Phi": Phi,
        "y": y,
        "component": component,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
        "feature_dim": Phi.shape[1],
        "data_hash": data_hash,
        "seed": seed,
    }


def save_shadow_dataset(
    dataset: Dict[str, Any],
    path: str,
) -> None:
    """
    Save a shadow dataset to disk for later deployment.
    Uses numpy .npz format for features + JSON metadata.
    """
    np.savez_compressed(
        path,
        Phi=dataset["Phi"],
        y=dataset["y"],
    )
    meta_path = path.replace(".npz", "_meta.json")
    meta = {k: v for k, v in dataset.items() if k not in ("Phi", "y")}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved shadow dataset to {path} and {meta_path}")


def load_shadow_dataset(path: str) -> Dict[str, Any]:
    """Load a shadow dataset from disk."""
    data = np.load(path)
    meta_path = path.replace(".npz", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "Phi": data["Phi"],
        "y": data["y"],
        **meta,
    }
