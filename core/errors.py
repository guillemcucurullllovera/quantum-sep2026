"""Typed exception hierarchy shared across the platform."""
from __future__ import annotations


class QuantumSep2026Error(Exception):
    """Base class for all platform-specific errors."""


class CredentialError(QuantumSep2026Error):
    """Raised when a required credential is missing or invalid.

    This must NEVER be silently swallowed into a no-op fallback — hardware
    execution paths must fail loudly rather than pretend to run.
    """


class BackendUnavailableError(QuantumSep2026Error):
    """Raised when a requested backend cannot be reached or is not eligible."""


class ValidationError(QuantumSep2026Error):
    """Raised when an algorithm/application input fails validation."""


class ExperimentNotFoundError(QuantumSep2026Error):
    """Raised when a run/experiment id is not present in the registry."""


class EvidenceIntegrityError(QuantumSep2026Error):
    """Raised when a result's claimed evidence cannot be substantiated from
    its own recorded fields — e.g. a QPU-kind backend with no auditable
    hardware job/session identifiers, or a financial-edge claim with no
    executed backtest artifact. Must NEVER be silently downgraded or
    swallowed: an unsubstantiated evidence claim is exactly the failure
    mode (unexecuted headline numbers presented as validated) that
    QUANTUM-0 found in `qrc_investor_demo.ipynb` and the withdrawn IBM
    decks.
    """
