"""Resource estimation for amplitude-estimation payoff circuits: given a
circuit (or just a target accuracy), estimate qubit count, post-transpile
depth/2Q-gate count (via hardware.circuit_analysis, not reimplemented), and
the number of oracle queries a canonical AE algorithm needs to reach a
target epsilon at a target confidence. Nothing in this platform estimates
AE query complexity yet -- IQAE's own module reports realized query counts
after running, not a pre-run estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from hardware.circuit_analysis.metrics import CircuitStats, analyze


@dataclass(frozen=True)
class AEResourceEstimate:
    target_epsilon: float
    confidence: float
    canonical_ae_queries: int
    iqae_expected_queries: int
    circuit_stats: CircuitStats | None
    note: str


def canonical_ae_query_count(epsilon: float, confidence: float = 0.95) -> int:
    """Canonical (QPE-based) amplitude estimation: O(1/epsilon) queries to the
    Grover operator for a fixed confidence, per Brassard-Hoyer-Mosca-Tapp 2002.
    Query count ~ ceil(pi / epsilon) at the textbook constant, inflated by the
    standard log(1/(1-confidence)) factor for majority-vote confidence boosting.
    """
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0, 1)")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    base = math.pi / epsilon
    boost = max(1.0, math.log(1.0 / (1.0 - confidence)))
    return int(math.ceil(base * boost))


def iqae_expected_query_count(epsilon: float, confidence: float = 0.95, alpha: float | None = None) -> int:
    """IQAE (Grinko et al. 2021, arXiv:1912.05559): O(1/epsilon * log(1/alpha))
    expected queries, a large practical improvement over canonical AE's
    constant-factor overhead -- this is *why* the platform's algorithms.qae_iqae
    module exists rather than a QPE-based estimator."""
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0, 1)")
    a = alpha if alpha is not None else (1.0 - confidence)
    if not (0 < a < 1):
        raise ValueError("alpha/confidence out of range")
    return int(math.ceil((1.0 / epsilon) * math.log(1.0 / a) * 2.0))


def estimate(circuit, target_epsilon: float, confidence: float = 0.95) -> AEResourceEstimate:
    """circuit: optional built A-circuit (e.g. from barrier_payoff.build_A_barrier_payoff)
    to report real post-build depth/gate-count alongside the query estimate;
    pass None to get just the query-complexity numbers."""
    stats = analyze(circuit) if circuit is not None else None
    canon = canonical_ae_query_count(target_epsilon, confidence)
    iqae = iqae_expected_query_count(target_epsilon, confidence)
    note = (
        f"IQAE needs ~{canon / iqae:.1f}x fewer expected oracle calls than canonical "
        f"QPE-based AE at epsilon={target_epsilon}, confidence={confidence} — consistent "
        "with why this platform's algorithms.qae_iqae uses IQAE, not QPE-AE."
    )
    return AEResourceEstimate(
        target_epsilon=target_epsilon, confidence=confidence,
        canonical_ae_queries=canon, iqae_expected_queries=iqae,
        circuit_stats=stats, note=note,
    )
