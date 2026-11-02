"""Unified kernel-computation API on top of feature_maps.py + kernels_gen1.py.

New in this consolidation (not present in either generation): PSD correction
via core.metrics (a real quantum-hardware-relevant concern -- a Gram matrix
estimated from noisy shots or a mismatched train/test split is not
guaranteed PSD, and feeding a non-PSD kernel into an SVM solver produces
undefined behavior rather than a clean error), kernel-target-alignment
scoring for *any* feature map (not just AngleCNOTRingV2 -- useful for
comparing Gen1 maps against each other before deciding one needs the full
Gen2 KTA-training treatment), and a single dispatch function so callers
don't need to know which generation's module a given feature map lives in.
"""
from __future__ import annotations

import numpy as np

from algorithms.pqk import kernels_gen1
from algorithms.pqk.feature_maps import FEATURE_MAPS, project_state
from algorithms.pqk.kta_alignment import compute_gram_matrix, compute_kta, kernel_concentration_measure
from core.errors import ValidationError
from core.metrics import is_positive_semidefinite, nearest_psd


def fidelity_gram(
    X: np.ndarray, feature_map_name: str, X2: np.ndarray | None = None, psd_correct: bool = True
) -> np.ndarray:
    if feature_map_name not in FEATURE_MAPS:
        raise ValidationError(f"Unknown feature map '{feature_map_name}'. Known: {sorted(FEATURE_MAPS)}")
    fmap = FEATURE_MAPS[feature_map_name]()
    K = kernels_gen1.compute_fidelity_gram(X, fmap, X2)
    if psd_correct and X2 is None and not is_positive_semidefinite(K):
        K = nearest_psd(K)
    return K


def projected_gram(
    X: np.ndarray,
    feature_map_name: str,
    X2: np.ndarray | None = None,
    gamma: float = 1.0,
    psd_correct: bool = True,
) -> np.ndarray:
    if feature_map_name not in FEATURE_MAPS:
        raise ValidationError(f"Unknown feature map '{feature_map_name}'. Known: {sorted(FEATURE_MAPS)}")
    fmap = FEATURE_MAPS[feature_map_name]()
    K, _, _ = kernels_gen1.compute_projected_gram(X, fmap, X2, gamma=gamma)
    if psd_correct and X2 is None and not is_positive_semidefinite(K):
        K = nearest_psd(K)
    return K


def score_feature_map(
    X: np.ndarray, y: np.ndarray, feature_map_name: str, method: str = "projected", gamma: float = 1.0
) -> dict[str, float]:
    """Compute KTA and concentration diagnostics for a feature map against
    labeled data, without any training -- useful to screen candidate maps
    (Gen1's three plus Gen2's AngleCNOTRingV2) before committing to the full
    KTA-gradient-ascent + Nystrom pipeline for the most promising one."""
    if method == "fidelity":
        K = fidelity_gram(X, feature_map_name, psd_correct=False)
    elif method == "projected":
        K = projected_gram(X, feature_map_name, gamma=gamma, psd_correct=False)
    else:
        raise ValidationError(f"method must be 'fidelity' or 'projected', got {method!r}")

    return {
        "kta": compute_kta(K, y.astype(float)),
        "concentration_std": kernel_concentration_measure(K),
        "is_psd": float(is_positive_semidefinite(K)),
    }


def compare_feature_maps(X: np.ndarray, y: np.ndarray, method: str = "projected") -> dict[str, dict[str, float]]:
    """Screen every registered feature map (Gen1's three + Gen2's
    AngleCNOTRingV2) and return their KTA/concentration diagnostics side by
    side, so a caller can see the concentration failure mode directly rather
    than taking it on faith from the docstrings."""
    return {name: score_feature_map(X, y, name, method=method) for name in FEATURE_MAPS}
