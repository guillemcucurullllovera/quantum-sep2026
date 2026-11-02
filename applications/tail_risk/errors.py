"""Typed exceptions for degenerate tail-risk inputs, plus the shared
validation helper every entry point in this package runs its P&L sample
through first.

Both subclass core.errors.ValidationError so callers that already catch the
platform-wide validation exception keep working; the subclasses exist so a
caller that wants to distinguish "empty data" from "no loss tail present"
can.
"""
from __future__ import annotations

import numpy as np

from core.errors import ValidationError


class EmptySampleError(ValidationError):
    """Raised when a P&L sample has zero observations."""


class DegenerateTailError(ValidationError):
    """Raised when a P&L sample has no left (loss) tail to estimate VaR/CVaR
    from -- e.g. every observation is a gain. This is almost always a sign
    convention bug upstream (PnL should be signed gain-positive/loss-negative)
    rather than a genuinely riskless sample, so it is treated as an error
    rather than silently returning VaR=0.
    """


def validate_pnl_sample(pnl_samples: np.ndarray) -> np.ndarray:
    """Coerce `pnl_samples` to a finite float array and raise a specific,
    typed error for each degenerate case this package's estimators cannot
    honestly handle. Every public entry point in applications.tail_risk
    should route its input through this function before doing anything else.
    """
    pnl = np.asarray(pnl_samples, dtype=float).reshape(-1)
    if pnl.size == 0:
        raise EmptySampleError(
            "pnl_samples is empty; cannot estimate tail risk from zero observations."
        )
    if not np.isfinite(pnl).all():
        raise ValidationError("pnl_samples contains NaN or inf values.")
    if float(np.min(pnl)) >= 0.0:
        raise DegenerateTailError(
            "pnl_samples has no losses (min(pnl) >= 0); there is no left tail to "
            "estimate VaR/CVaR from. This usually indicates a sign-convention bug "
            "upstream (PnL should be gain-positive / loss-negative) rather than a "
            "genuinely riskless sample."
        )
    return pnl
