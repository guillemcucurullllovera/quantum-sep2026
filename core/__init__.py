"""Asymmetrica Quantum Platform — core infrastructure.

Every algorithm, application, and hardware module in this repository is
expected to depend on `core`, not the other way around. Nothing in this
package should import from `algorithms`, `applications`, or `hardware`.
"""
from core.errors import (
    QuantumSep2026Error,
    CredentialError,
    BackendUnavailableError,
    ValidationError,
    ExperimentNotFoundError,
)

__all__ = [
    "QuantumSep2026Error",
    "CredentialError",
    "BackendUnavailableError",
    "ValidationError",
    "ExperimentNotFoundError",
]
