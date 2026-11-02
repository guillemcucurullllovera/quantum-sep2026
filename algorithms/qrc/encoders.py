"""Financial feature encoders for QRC input preparation.

The angle-encoding scheme (tanh-bounded, preserves sign) and signal-
formatting convention follow the platform's Gen2 QRC design, isolated
here from unrelated deployment-integration code.
"""
from __future__ import annotations

import numpy as np

from core.errors import ValidationError


def encode_inputs(x: np.ndarray, n1: int, scale: float = np.pi, clip: bool = True) -> np.ndarray:
    """Map a feature vector x to rotation angles theta = scale * tanh(x).

    tanh bounds the angle (prevents reservoir saturation), preserves sign
    (directional information), and is well-behaved if gradients are ever
    needed.
    """
    if len(x) < n1:
        raise ValidationError(f"Feature vector has length {len(x)}, need n1={n1}")
    x_in = np.asarray(x[:n1], dtype=float)
    angles = scale * np.tanh(x_in)
    if clip:
        angles = np.clip(angles, -np.pi, np.pi)
    return angles


def encode_sequence(X: np.ndarray, n1: int, scale: float = np.pi) -> np.ndarray:
    """Encode a (T, F) feature matrix row-by-row into (T, n1) angles."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValidationError(f"X must be 2-D (T, F), got shape {X.shape}")
    if X.shape[1] < n1:
        raise ValidationError(f"X has {X.shape[1]} features, need n1={n1}")
    return np.stack([encode_inputs(X[t], n1, scale) for t in range(len(X))])


def minmax_encode_sequence(X: np.ndarray, n1: int) -> np.ndarray:
    """Alternative encoder: per-column min-max scaling to [-pi, pi], matching
    the arXiv-replication component's convention (rather than tanh). Useful
    when comparing the two generations on identical inputs, since encoding
    choice materially affects reservoir dynamics.
    """
    X = np.asarray(X, dtype=float)[:, :n1]
    col_min, col_max = X.min(axis=0), X.max(axis=0)
    span = np.where(col_max - col_min < 1e-12, 1.0, col_max - col_min)
    return (X - col_min) / span * (2 * np.pi) - np.pi


def triple_band_signal(z_expectations: np.ndarray) -> dict:
    """Project a reservoir Z-expectation vector into a coarse (buy, sell,
    hold)-shaped probability vector via softmax over three equal bands.
    A simple, documented projection — not a claim of directional edge.
    """
    n = len(z_expectations)
    third = max(1, n // 3)
    buy = float(np.mean(z_expectations[:third]))
    sell = float(np.mean(z_expectations[third : 2 * third]))
    hold = float(np.mean(z_expectations[2 * third :]))
    raw = np.array([buy, sell, hold])
    shifted = raw - raw.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    return {
        "p_buy": float(probs[0]),
        "p_sell": float(probs[1]),
        "p_hold": float(probs[2]),
        "confidence": float(probs.max()),
    }


__all__ = ["encode_inputs", "encode_sequence", "minmax_encode_sequence", "triple_band_signal"]
