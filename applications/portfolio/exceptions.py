"""Typed exceptions for the portfolio application layer.

Extends `core.errors`'s platform-wide hierarchy rather than replacing it --
`CredentialError` and `BackendUnavailableError` from `core.errors` are reused
directly for hardware execution paths so callers can catch either the
platform-wide or the portfolio-specific exceptions with one `except` clause
where that's useful (all of these are `QuantumSep2026Error` subclasses).
"""
from __future__ import annotations

from core.errors import QuantumSep2026Error, ValidationError


class UnsupportedConstraintError(ValidationError):
    """A `ConstraintSpec.kind` is not supported by the selected solver
    backend (e.g. an extended constraint kind passed to MODELED_REPAIR,
    which only understands the constraint vocabulary native to
    `algorithms_extended.qiskit_optimization_finance.FinancePortfolioModel`).
    """


class InfeasibleProblemError(ValidationError):
    """No solver backend -- including any repair step -- could produce a
    bitstring satisfying this PortfolioSpec's hard constraints. Distinct
    from a bare heuristic sub-optimality: this means every constraint-
    satisfaction attempt genuinely failed, not just that the objective is
    suboptimal.
    """


class SolverFailureError(QuantumSep2026Error):
    """A solver backend raised internally, or returned a result outside its
    documented contract (wrong bitstring length, NaN/inf cost, missing
    required fields) -- never silently swallowed into a fallback result.
    """
