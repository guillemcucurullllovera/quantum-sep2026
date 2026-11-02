"""Tail-risk assessment: classical VaR/CVaR/probability-of-loss from a P&L sample,
plus a quantum (amplitude-estimation) cross-check of P(PnL <= threshold) against the
same empirical distribution -- the comparison is the deliverable, not either number
alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.qae_iqae.finance_integrands import build_return_distribution_barrier
from algorithms.qae_iqae.register import estimate_barrier_probability
from core.metrics import expected_shortfall, probability_of_loss, value_at_risk
from core.registry import APPLICATION_REGISTRY


@dataclass(frozen=True)
class TailRiskReport:
    var: float
    cvar: float
    prob_of_loss: float
    quantum_prob_below_threshold: float
    empirical_prob_below_threshold: float
    threshold: float
    alpha: float
    discretisation_gap: float
    n_samples: int
    meta_agent_cvar: float | None = None


def _meta_agent_cvar_from_pnl(pnl_samples: np.ndarray, alpha: float, seed: int) -> float:
    """Adapter: classical.meta_agent.MonteCarloCVaR (the C1.5 gate) is a
    regime-conditional PORTFOLIO-WEIGHT
    estimator (fit on a (T,K) returns matrix + (T,S) regime-probability
    matrix, queried with a weight vector + a regime-probability vector) --
    a structurally different interface than a raw PnL-sample CVaR. This
    treats `pnl_samples` as one single-strategy return history under one
    degenerate always-on regime, which collapses the regime-conditional
    bootstrap to a plain historical bootstrap of `pnl_samples` itself --
    a legitimate (if maximally simple) special case, not a misuse, and
    gives a third, independently-implemented CVaR estimate to cross-check
    against `core.metrics.expected_shortfall`'s direct percentile method.
    """
    from classical.meta_agent.monte_carlo_cvar import MonteCarloCVaR

    returns = np.asarray(pnl_samples, dtype=np.float64).reshape(-1, 1)
    regime_probs = np.ones((returns.shape[0], 1))
    model = MonteCarloCVaR(alpha=alpha, target_loss=abs(float(np.percentile(pnl_samples, 100 * (1 - alpha)))) or 0.01,
                            random_state=seed)
    model.fit(returns, regime_probs)
    est = model.estimate_cvar(weights=np.array([1.0]), regime_probs_t=np.array([1.0]))
    return float(est["cvar"])


def assess_tail_risk(
    pnl_samples: np.ndarray,
    *,
    alpha: float = 0.95,
    loss_threshold: float | None = None,
    n_grid_qubits: int = 4,
    run_quantum: bool = True,
    epsilon: float = 0.02,
    shots: int = 2048,
    run_meta_agent_comparator: bool = True,
    seed: int = 42,
) -> TailRiskReport:
    """Classical VaR/CVaR/probability-of-loss from `pnl_samples`, and (if
    `run_quantum`) a QAE-based cross-check of P(PnL <= loss_threshold) against the
    same empirical distribution, histogrammed into 2**n_grid_qubits bins.

    `loss_threshold` defaults to -VaR (i.e. checks P(loss exceeds the VaR bar), the
    natural quantum cross-check of the classical VaR estimate itself).

    PERFORMANCE NOTE: the factored barrier circuit's comparator needs an
    n_grid_qubits-controlled X per in-barrier index, which several backend
    transpilers (including FakeKingston, used here) synthesize expensively past
    about n_grid_qubits=5 -- keep n_grid_qubits<=4 for interactive/CI use; a real
    optimized MCX synthesis pass is future hardware/ workstream scope, not fixed here.
    """
    pnl = np.asarray(pnl_samples, dtype=float)
    var = value_at_risk(pnl, alpha)
    cvar = expected_shortfall(pnl, alpha)
    pol = probability_of_loss(pnl)
    threshold = loss_threshold if loss_threshold is not None else -var

    empirical_prob = float(np.mean(pnl <= threshold))
    quantum_prob = float("nan")
    gap = float("nan")
    if run_quantum:
        barrier = build_return_distribution_barrier(pnl, loss_threshold=threshold, n=n_grid_qubits)
        result = estimate_barrier_probability(
            barrier, epsilon=epsilon, shots=shots, log_to_registry=False
        )
        quantum_prob = result.metrics["a_hat_ideal"]
        gap = result.metrics["discretisation_gap"]

    meta_cvar = None
    if run_meta_agent_comparator:
        meta_cvar = _meta_agent_cvar_from_pnl(pnl, alpha, seed)

    return TailRiskReport(
        var=var, cvar=cvar, prob_of_loss=pol,
        quantum_prob_below_threshold=quantum_prob, empirical_prob_below_threshold=empirical_prob,
        threshold=threshold, alpha=alpha, discretisation_gap=gap, n_samples=len(pnl),
        meta_agent_cvar=meta_cvar,
    )


def _register() -> None:
    if "tail_risk" not in APPLICATION_REGISTRY.list():
        APPLICATION_REGISTRY.register("tail_risk")(assess_tail_risk)


_register()
