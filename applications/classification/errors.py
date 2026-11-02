"""Exceptions specific to the unified classification/regime/anomaly product
layer. Subclass `core.errors.ValidationError` so callers that already catch
the platform-wide error hierarchy keep working without changes."""
from __future__ import annotations

from core.errors import ValidationError


class UnsupportedTaskError(ValidationError):
    """Raised when an `Algorithm` is dispatched against a `TaskType` it does
    not support (e.g. asking `REGIME_HMM` to do binary classification, or
    `CLASSICAL_RIDGE_REGRESSION` to do classification)."""


class AlgorithmUnavailableError(ValidationError):
    """Raised when an algorithm's runtime dependency is missing (e.g.
    scikit-learn absent for `CLASSICAL_RANDOM_FOREST`). Distinct from
    `UnsupportedTaskError`: the algorithm+task pairing is valid, the
    environment just can't run it right now."""


class LeakageError(ValidationError):
    """Raised when a validation strategy would train on information from the
    future relative to its own test window -- fails loudly rather than
    silently producing an optimistic result."""
