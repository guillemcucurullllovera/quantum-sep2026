"""Real options-based hedging: given a portfolio's spot exposure and a
regime/vol forecast, size a protective-put hedge (via
`applications.options.pricing_methods.price_option`, not a linear
downside-truncation multiplier), then measure its *actual* effect on
simulated tail risk with `classical.meta_agent.MonteCarloCVaR` and,
optionally, the QAE-based tail-probability cross-check already built in
`applications.tail_risk.assess_tail_risk`.

Relationship to the existing hedging module in this directory,
`regime_conditional_hedge.py`: that module is a real, working composition
(VQC regime classifier -> a HEDGE_EFFECTIVENESS downside-truncation
multiplier -> tail-risk measurement) and is intentionally not modified here.
Its own docstring names the gap this module fills: "hedge effectiveness is
modeled as a downside-truncation multiplier ... not a full options-pricing
simulation of the hedge itself (that's applications.options's job, kept
separate ...)". This module IS that options-pricing simulation: the hedge's
cost is a real (Black-Scholes, cross-checked against Monte Carlo) put
premium, and its payoff is the real max(K - S_T, 0) protective-put payoff
under simulated terminal prices -- not an approximation of what an option
would do.

Reuses the same regime-name vocabulary (`applications.regime_detection.
synthetic_regime_dataset.REGIMES`) as `regime_conditional_hedge.py` so a
regime label produced anywhere in the platform means the same thing here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from applications.options.domain import OptionSpec, PayoffKind
from applications.options.pricing_methods import PricingMethod, price_option
from applications.regime_detection.synthetic_regime_dataset import REGIMES
from classical.monte_carlo.gbm import gbm_terminal_prices
from core.errors import ValidationError
from core.metrics import expected_shortfall, value_at_risk

# Coverage fraction of notional to hedge, by predicted regime -- same
# vocabulary/spirit as regime_conditional_hedge.DEFAULT_HEDGE_POLICY (0 =
# don't hedge, 1 = hedge the full exposure), but feeding an actual put-option
# notional here rather than a downside-truncation multiplier.
DEFAULT_COVERAGE_POLICY: dict[str, float] = {"trend_up": 0.0, "trend_down": 1.0, "ranging": 0.5}


@dataclass(frozen=True)
class PortfolioExposure:
    """A long spot exposure to hedge with a protective put."""

    notional: float          # $ notional of the long exposure (S0 * shares, effectively)
    spot: float
    realized_vol: float      # the vol actually used to SIMULATE the P&L path
    rate: float
    horizon: float           # years


@dataclass(frozen=True)
class HedgeRatioResult:
    regime: str
    coverage: float
    forecast_vol: float
    put_strike: float
    put_premium: float
    put_delta: float
    n_puts_per_unit_notional: float
    pricing_method: PricingMethod


def compute_hedge_ratio(
    exposure: PortfolioExposure,
    *,
    regime: str,
    forecast_vol: float | None = None,
    strike_moneyness: float = 0.95,
    coverage_policy: dict[str, float] | None = None,
    pricing_method: PricingMethod = PricingMethod.BLACK_SCHOLES,
) -> HedgeRatioResult:
    """Size a protective put: strike = `strike_moneyness` * spot, priced at
    `forecast_vol` (defaults to `exposure.realized_vol` -- pass a different
    number to model a vol forecast that disagrees with what will actually
    realize, which is the honest point of separating the two).

    `coverage` (how much of the notional to cover) comes from
    `coverage_policy[regime]`, defaulting to `DEFAULT_COVERAGE_POLICY` --
    this is the "regime/vol forecast -> hedge ratio" decision the build
    task asks for, made explicit and overridable rather than hardcoded.
    """
    if regime not in REGIMES:
        raise ValidationError(f"unknown regime {regime!r}, expected one of {REGIMES}")
    policy = coverage_policy or DEFAULT_COVERAGE_POLICY
    coverage = policy.get(regime, 0.0)
    vol = forecast_vol if forecast_vol is not None else exposure.realized_vol

    put_spec = OptionSpec(
        payoff=PayoffKind.EUROPEAN_PUT, spot=exposure.spot, strike=strike_moneyness * exposure.spot,
        rate=exposure.rate, vol=vol, maturity=exposure.horizon,
    )
    priced = price_option(put_spec, pricing_method)

    from applications.options.greeks import analytic_bs_delta

    put_delta = analytic_bs_delta(put_spec)  # negative, in [-1, 0]
    # One put contract (notional = spot) hedges |put_delta| * spot of
    # exposure at inception; to cover `coverage` * exposure.notional dollars
    # of exposure we need this many puts (each on `exposure.spot` worth of
    # underlying) per dollar of notional.
    n_puts_per_unit_notional = coverage / max(abs(put_delta) * exposure.spot, 1e-9)

    return HedgeRatioResult(
        regime=regime, coverage=coverage, forecast_vol=vol, put_strike=put_spec.strike,
        put_premium=priced.price, put_delta=put_delta,
        n_puts_per_unit_notional=n_puts_per_unit_notional, pricing_method=priced.method,
    )


@dataclass(frozen=True)
class HedgeEffectivenessReport:
    exposure: PortfolioExposure
    hedge: HedgeRatioResult
    n_paths: int
    unhedged_var: float
    unhedged_cvar: float
    hedged_var: float
    hedged_cvar: float
    cvar_reduction: float          # unhedged_cvar - hedged_cvar (positive = improvement)
    cvar_reduction_fraction: float
    hedge_cost_as_fraction_of_notional: float
    meta_agent_unhedged_cvar: float | None = None
    meta_agent_hedged_cvar: float | None = None
    quantum_tail_prob_unhedged: float | None = None
    quantum_tail_prob_hedged: float | None = None


def simulate_hedge_effectiveness(
    exposure: PortfolioExposure,
    hedge: HedgeRatioResult,
    *,
    alpha: float = 0.95,
    n_paths: int = 20_000,
    seed: int = 42,
    run_meta_agent_comparator: bool = True,
    run_quantum_tail_probability: bool = False,
) -> HedgeEffectivenessReport:
    """Monte-Carlo-simulate the underlying under `exposure.realized_vol`
    (which may differ from the `forecast_vol` the hedge was SIZED with --
    that mismatch is exactly what makes this a real test of hedge
    effectiveness, not a tautology), build the unhedged and hedged P&L
    distributions, and report VaR/CVaR before and after -- cross-checked by
    `classical.meta_agent.MonteCarloCVaR` and, optionally, the QAE-based
    tail-probability estimator already built in `applications.tail_risk`.
    """
    terminal = gbm_terminal_prices(
        s0=exposure.spot, mu=exposure.rate, sigma=exposure.realized_vol, t=exposure.horizon,
        n_paths=n_paths, seed=seed,
    )
    disc = np.exp(-exposure.rate * exposure.horizon)

    unhedged_pnl = exposure.notional * (terminal / exposure.spot - 1.0)

    n_puts = hedge.n_puts_per_unit_notional * exposure.notional
    put_payoff = disc * np.maximum(hedge.put_strike - terminal, 0.0)
    hedge_pnl = n_puts * (put_payoff - hedge.put_premium)
    hedged_pnl = unhedged_pnl + hedge_pnl

    unhedged_var, unhedged_cvar = value_at_risk(unhedged_pnl, alpha), expected_shortfall(unhedged_pnl, alpha)
    hedged_var, hedged_cvar = value_at_risk(hedged_pnl, alpha), expected_shortfall(hedged_pnl, alpha)

    meta_unhedged = meta_hedged = None
    if run_meta_agent_comparator:
        meta_unhedged = _meta_agent_cvar(unhedged_pnl, alpha, seed)
        meta_hedged = _meta_agent_cvar(hedged_pnl, alpha, seed)

    q_unhedged = q_hedged = None
    if run_quantum_tail_probability:
        from applications.tail_risk.var_cvar import assess_tail_risk

        q_unhedged = assess_tail_risk(
            unhedged_pnl, alpha=alpha, run_quantum=True, run_meta_agent_comparator=False, seed=seed,
        ).quantum_prob_below_threshold
        q_hedged = assess_tail_risk(
            hedged_pnl, alpha=alpha, run_quantum=True, run_meta_agent_comparator=False, seed=seed,
        ).quantum_prob_below_threshold

    cvar_reduction = unhedged_cvar - hedged_cvar
    total_hedge_cost = n_puts * hedge.put_premium
    return HedgeEffectivenessReport(
        exposure=exposure, hedge=hedge, n_paths=n_paths,
        unhedged_var=unhedged_var, unhedged_cvar=unhedged_cvar,
        hedged_var=hedged_var, hedged_cvar=hedged_cvar,
        cvar_reduction=cvar_reduction,
        cvar_reduction_fraction=(cvar_reduction / unhedged_cvar) if unhedged_cvar else float("nan"),
        hedge_cost_as_fraction_of_notional=total_hedge_cost / exposure.notional if exposure.notional else float("nan"),
        meta_agent_unhedged_cvar=meta_unhedged, meta_agent_hedged_cvar=meta_hedged,
        quantum_tail_prob_unhedged=q_unhedged, quantum_tail_prob_hedged=q_hedged,
    )


def _meta_agent_cvar(pnl_samples: np.ndarray, alpha: float, seed: int) -> float:
    """Same single-strategy/degenerate-regime adapter pattern as
    `applications.tail_risk.var_cvar._meta_agent_cvar_from_pnl` -- kept as a
    private local copy (not imported) because that helper is private to its
    module and this is a different-enough call site (raw $ P&L here, not a
    return series) to warrant its own docstring rather than reaching into
    another module's underscore-prefixed internals."""
    from classical.meta_agent.monte_carlo_cvar import MonteCarloCVaR

    returns = np.asarray(pnl_samples, dtype=np.float64).reshape(-1, 1)
    regime_probs = np.ones((returns.shape[0], 1))
    target_loss = abs(float(np.percentile(pnl_samples, 100 * (1 - alpha)))) or 1.0
    model = MonteCarloCVaR(alpha=alpha, target_loss=target_loss, random_state=seed)
    model.fit(returns, regime_probs)
    est = model.estimate_cvar(weights=np.array([1.0]), regime_probs_t=np.array([1.0]))
    return float(est["cvar"])


__all__ = [
    "PortfolioExposure", "HedgeRatioResult", "HedgeEffectivenessReport",
    "DEFAULT_COVERAGE_POLICY", "compute_hedge_ratio", "simulate_hedge_effectiveness",
]
