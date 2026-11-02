"""Specific exception types for the time-series/volatility application
layer. All subclass core.errors.ValidationError so existing callers that
already catch that base class keep working, while callers who want to
handle one failure mode specifically (an unknown method name vs. a data
sizing problem vs. a bad noise config) can do so.
"""
from __future__ import annotations

from core.errors import ValidationError


class UnknownForecastMethodError(ValidationError):
    """Raised when a MethodChoice has no registered feature-extraction adapter."""


class InsufficientDataError(ValidationError):
    """Raised when a dataset/spec combination doesn't leave enough rows for
    the requested validation strategy or preprocessing step (e.g. fracdiff's
    warm-up window, or a walk-forward split with too few folds)."""


class NoiseConfigError(ValidationError):
    """Raised when a NoiseSpec is invalid or incompatible with the data
    (e.g. a probability outside [0, 1], or too little data for the
    noisy-vs-noiseless diagnostic split)."""


__all__ = ["UnknownForecastMethodError", "InsufficientDataError", "NoiseConfigError"]
