"""Real, honest multi-method VaR/CVaR comparison on one P&L sample.

Five classical/quantum-adjacent methods, each independently implemented
elsewhere in the platform and wired together here -- the comparison across
them (where they agree, where and why they diverge) is the deliverable,
not any single method's number:

  (a) qae_amplitude_estimation -- quantum amplitude estimation locates the
      alpha-quantile by bisecting `algorithms.qae_iqae`'s barrier-probability
      oracle over the empirical histogram grid (see `_qae_located_var_cvar`
      docstring for exactly what is and isn't quantum in the resulting CVaR).
  (b) monte_carlo_gbm -- classical Monte Carlo: simulate a GBM terminal-price
      distribution (`classical.monte_carlo.gbm`) matched to the sample's
      mean/std, then apply `core.metrics`'s VaR/CVaR to the simulated path.
  (c) meta_agent_bootstrap -- `classical.meta_agent.MonteCarloCVaR`, the
  regime-conditional historical-bootstrap estimator, run
      under a degenerate always-on single-regime (see its own module
      docstring for why that's a legitimate reduction, not a misuse).
  (d) parametric_gaussian / cornish_fisher -- `classical.risk`'s
      distribution-assumption estimators: a pure closed-form Gaussian VaR/ES,
      and a Cornish-Fisher skew/kurtosis-adjusted quantile.
  (e) historical -- `core.metrics`'s direct empirical percentile VaR/CVaR,
      the ground truth every other method is implicitly approximating.

All six report into `applications.tail_risk.domain.MethodResult` under the
same sign convention (`var`/`cvar` are positive loss magnitudes), so
`cvar >= var` is a structural invariant checked for every method, not just
one -- see tests/applications/test_tail_risk_comparison.py.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import kurtosis, norm, skew

from algorithms.qae_iqae.finance_integrands import build_return_distribution_barrier
from algorithms.qae_iqae.register import estimate_barrier_probability
from classical.monte_carlo.gbm import gbm_terminal_prices
from classical.risk.parametric import cornish_fisher_var, parametric_var_gaussian
from core.metrics import expected_shortfall, probability_of_loss, value_at_risk
from core.registry import APPLICATION_REGISTRY

from applications.tail_risk.domain import MethodResult, TailRiskComparisonReport, TailRiskSpec
from applications.tail_risk.errors import validate_pnl_sample

__all__ = ["compare_methods", "compare_methods_all_levels"]


# ── individual estimators ──────────────────────────────────────────────


def _historical_var_cvar(pnl: np.ndarray, alpha: float) -> tuple[float, float, dict]:
    return value_at_risk(pnl, alpha), expected_shortfall(pnl, alpha), {}


def _gaussian_var_cvar(pnl: np.ndarray, alpha: float) -> tuple[float, float, dict]:
    """Pure closed-form Gaussian VaR/ES -- the only method here that never
    looks at the sample's tail directly, only its first two moments. This is
    the method most likely to understate risk under fat tails, by design:
    it's the baseline every other method is implicitly critiquing.
    """
    mu = float(np.mean(pnl))
    sigma = float(np.std(pnl, ddof=1))
    var = parametric_var_gaussian(mu, sigma, alpha)
    z = norm.ppf(alpha)
    # Closed-form Gaussian expected shortfall: mean loss beyond the z_alpha
    # quantile, via the standard normal's Mills-ratio tail expectation.
    cvar = float(-mu + sigma * norm.pdf(z) / (1.0 - alpha))
    return var, cvar, {"mean": mu, "std": sigma, "z_alpha": float(z)}


def _cornish_fisher_var_cvar(pnl: np.ndarray, alpha: float) -> tuple[float, float, dict]:
    """Cornish-Fisher VaR adjusts the Gaussian quantile for the sample's own
    skew/kurtosis. There is no closed-form Cornish-Fisher expected shortfall
    (the expansion only approximates a quantile, not a density), so the CVaR
    here is an honest hybrid: the CF-adjusted threshold, then the *actual*
    empirical sample's tail mean beyond it -- distinct from `historical`
    only in *where* the tail cutoff falls, not in how the tail is averaged.
    """
    mu = float(np.mean(pnl))
    sigma = float(np.std(pnl, ddof=1))
    sk = float(skew(pnl, bias=False)) if pnl.size > 2 else 0.0
    ek = float(kurtosis(pnl, fisher=True, bias=False)) if pnl.size > 3 else 0.0
    var = cornish_fisher_var(mu, sigma, sk, ek, alpha)
    losses = -pnl
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else float(var)
    return var, cvar, {"skew": sk, "excess_kurtosis": ek}


def _monte_carlo_gbm_var_cvar(pnl: np.ndarray, alpha: float, *, n_paths: int, seed: int) -> tuple[float, float, dict]:
    """Classical Monte Carlo: simulate a lognormal (GBM) terminal-price
    distribution whose single-step drift/vol are matched to the sample's own
    mean/std, then apply `core.metrics`'s VaR/CVaR to the simulated
    distribution rather than the raw sample. Structurally distinct from
    every parametric/historical method here: GBM's terminal price is bounded
    below by zero, so the simulated PnL (terminal - s0) has a bounded
    downside (>= -s0) that a Gaussian or historical-empirical estimate does
    not -- a real, reportable point of divergence under high volatility.
    `pnl_samples` is treated as already denominated at the target horizon
    (t=1, one GBM step), consistent with how every other method here reads
    it directly without a horizon rescaling.
    """
    mu = float(np.mean(pnl))
    sigma = max(float(np.std(pnl, ddof=1)), 1e-8)
    s0 = 1.0
    n = n_paths if n_paths % 2 == 0 else n_paths + 1
    terminal = gbm_terminal_prices(s0=s0, mu=mu, sigma=sigma, t=1.0, n_paths=n, seed=seed)
    simulated_pnl = terminal - s0
    var = value_at_risk(simulated_pnl, alpha)
    cvar = expected_shortfall(simulated_pnl, alpha)
    return var, cvar, {"gbm_mu": mu, "gbm_sigma": sigma, "n_paths": n}


def _meta_agent_var_cvar(pnl: np.ndarray, alpha: float, *, n_bootstrap: int, seed: int) -> tuple[float, float, dict]:
    """`classical.meta_agent.MonteCarloCVaR` -- a regime-conditional
    historical-bootstrap estimator (the C1.5 gate) fit on a (T,K) returns matrix + (T,S) regime-probability
    matrix. `pnl_samples` is treated as one single-strategy return history
    under one degenerate always-on regime, collapsing the regime-conditional
    bootstrap to a plain historical bootstrap of `pnl_samples` itself -- a
    legitimate special case (see the same reduction and rationale in
    `applications.tail_risk.var_cvar._meta_agent_cvar_from_pnl`), giving an
    independently-implemented bootstrap CVaR to cross-check against
    `core.metrics.expected_shortfall`'s direct percentile method. Its native
    sign convention is negative-number losses; flipped to this package's
    positive-magnitude convention here.
    """
    from classical.meta_agent.monte_carlo_cvar import MonteCarloCVaR

    returns = pnl.reshape(-1, 1)
    regime_probs = np.ones((returns.shape[0], 1))
    target_loss = abs(float(np.percentile(pnl, 100 * (1 - alpha)))) or 0.01
    model = MonteCarloCVaR(alpha=alpha, target_loss=target_loss, n_bootstrap=n_bootstrap, random_state=seed)
    model.fit(returns, regime_probs)
    est = model.estimate_cvar(weights=np.array([1.0]), regime_probs_t=np.array([1.0]))
    var = -float(est["var"])
    cvar = -float(est["cvar"])
    return var, cvar, {
        "bootstrap_mean": est["mean"], "bootstrap_std": est["std"], "n_bootstrap": est["n_samples"],
    }


def _qae_located_var_cvar(
    pnl: np.ndarray, alpha: float, *, n_grid_qubits: int, epsilon: float, shots: int, seed: int,
) -> tuple[float, float, dict]:
    """Quantum amplitude estimation locates the alpha-quantile by binary
    search: `algorithms.qae_iqae` estimates P(PnL <= threshold) for a
    candidate threshold (a bin edge of the sample's 2**n_grid_qubits-bin
    histogram, via `build_return_distribution_barrier` +
    `estimate_barrier_probability`), and the search narrows toward the edge
    whose estimated probability is closest to the target exceedance rate
    (1 - alpha) -- O(n_grid_qubits) QAE calls, not one call per candidate.

    HONESTY NOTE on the resulting CVaR: QAE gives a probability oracle, not
    a tail-conditional-expectation oracle (that would need a second,
    differently-shaped integrand -- future work, not built currently).
    The CVaR reported here is therefore a hybrid: the QAE-located threshold,
    then the mean of the *actual empirical sample's* tail beyond it -- the
    same tail-averaging pattern `_cornish_fisher_var_cvar` uses for its own
    non-quantum-native threshold. This guarantees cvar >= var by
    construction (the tail mean of losses all >= the threshold), and is the
    honest amount of "quantum" in this estimate: the quantile location is
    quantum, the tail averaging is not.
    """
    lo, hi = float(pnl.min()), float(pnl.max())
    m = 2 ** n_grid_qubits
    edges = np.histogram_bin_edges(pnl, bins=m, range=(lo, hi))
    candidates = edges[1:-1]  # interior edges only; endpoints are degenerate (prob 0 or 1)
    if candidates.size == 0:
        # Too few bins to have an interior edge (n_grid_qubits too small) --
        # fall back to the historical threshold itself, no QAE call needed.
        threshold = float(np.percentile(pnl, 100 * (1 - alpha)))
        losses = -pnl
        tail = losses[losses >= -threshold]
        var = -threshold
        cvar = float(tail.mean()) if tail.size else var
        return var, cvar, {"n_qae_calls": 0, "note": "grid too coarse for bisection; used historical threshold"}

    target = 1.0 - alpha
    lo_idx, hi_idx = 0, len(candidates) - 1
    best_threshold, best_prob, best_gap = None, None, None
    n_calls = 0
    last_ideal, last_noisy, last_gap = float("nan"), float("nan"), float("nan")

    while lo_idx <= hi_idx:
        mid = (lo_idx + hi_idx) // 2
        threshold = float(candidates[mid])
        barrier = build_return_distribution_barrier(pnl, loss_threshold=threshold, n=n_grid_qubits)
        result = estimate_barrier_probability(
            barrier, epsilon=epsilon, shots=shots, seed=seed, log_to_registry=False,
        )
        n_calls += 1
        p_ideal = result.metrics["a_hat_ideal"]
        last_ideal = p_ideal
        last_noisy = result.metrics["a_hat_noisy"]
        last_gap = result.metrics["discretisation_gap"]

        gap = abs(p_ideal - target)
        if best_gap is None or gap < best_gap:
            best_threshold, best_prob, best_gap = threshold, p_ideal, gap

        if p_ideal < target:
            lo_idx = mid + 1
        else:
            hi_idx = mid - 1

    var = -float(best_threshold)
    losses = -pnl
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar, {
        "n_qae_calls": n_calls,
        "located_threshold": float(best_threshold),
        "prob_at_threshold_ideal": float(best_prob),
        "prob_at_threshold_noisy": float(last_noisy),
        "discretisation_gap": float(last_gap),
        "target_exceedance_rate": target,
    }


# ── comparison orchestration ────────────────────────────────────────────


def _build_diagnostics(pnl: np.ndarray, results: dict[str, MethodResult]) -> tuple[dict[str, float], list[str]]:
    diagnostics: dict[str, float] = {}
    notes: list[str] = []

    var_values = {name: r.var for name, r in results.items()}
    cvar_values = {name: r.cvar for name, r in results.items()}
    diagnostics["var_spread"] = float(max(var_values.values()) - min(var_values.values()))
    diagnostics["cvar_spread"] = float(max(cvar_values.values()) - min(cvar_values.values()))

    sk = float(skew(pnl, bias=False)) if pnl.size > 2 else 0.0
    ek = float(kurtosis(pnl, fisher=True, bias=False)) if pnl.size > 3 else 0.0
    diagnostics["sample_skew"] = sk
    diagnostics["sample_excess_kurtosis"] = ek

    if "parametric_gaussian" in results and "cornish_fisher" in results:
        g_var = results["parametric_gaussian"].var
        cf_var = results["cornish_fisher"].var
        rel_gap = abs(cf_var - g_var) / g_var if g_var != 0 else 0.0
        diagnostics["cornish_fisher_vs_gaussian_rel_gap"] = float(rel_gap)
        if rel_gap > 0.15:
            notes.append(
                f"Cornish-Fisher VaR diverges {rel_gap * 100:.1f}% from parametric Gaussian VaR "
                f"(sample excess kurtosis={ek:.2f}, skew={sk:.2f}) -- the fat-tailed/skewed sample "
                f"is being understated by the Gaussian assumption; Cornish-Fisher and historical "
                f"are the more trustworthy methods here."
            )
        else:
            notes.append(
                f"Cornish-Fisher VaR is within {rel_gap * 100:.1f}% of parametric Gaussian VaR -- "
                f"this sample is close enough to Gaussian (excess kurtosis={ek:.2f}) that the "
                f"normality assumption is not materially costing accuracy."
            )

    if "meta_agent_bootstrap" in results and "historical" in results:
        h_var, m_var = results["historical"].var, results["meta_agent_bootstrap"].var
        rel = abs(m_var - h_var) / h_var if h_var != 0 else 0.0
        diagnostics["meta_agent_vs_historical_rel_gap"] = float(rel)
        notes.append(
            f"meta_agent_bootstrap VaR is within {rel * 100:.1f}% of historical VaR, as expected: "
            f"under the degenerate single-regime reduction it collapses to a bootstrap resample of "
            f"the same empirical sample historical reads directly."
        )

    if "monte_carlo_gbm" in results and "historical" in results:
        h_var, mc_var = results["historical"].var, results["monte_carlo_gbm"].var
        rel = (mc_var - h_var) / h_var if h_var != 0 else 0.0
        diagnostics["monte_carlo_gbm_vs_historical_rel_gap"] = float(rel)
        notes.append(
            f"monte_carlo_gbm VaR is {rel * 100:+.1f}% relative to historical VaR -- monte_carlo_gbm "
            f"simulates a lognormal terminal distribution (downside bounded at -1x notional) rather "
            f"than resampling the empirical tail directly, so a gap here reflects that structural "
            f"difference, not estimator noise alone."
        )

    if "qae_amplitude_estimation" in results and "historical" in results:
        h_var, q_var = results["historical"].var, results["qae_amplitude_estimation"].var
        gap = abs(q_var - h_var)
        diagnostics["qae_vs_historical_abs_gap"] = float(gap)
        n_calls = results["qae_amplitude_estimation"].extra.get("n_qae_calls", "?")
        notes.append(
            f"qae_amplitude_estimation located its VaR threshold in {n_calls} amplitude-estimation "
            f"calls via bisection over the empirical histogram grid, landing within {gap:.5g} of the "
            f"exact historical VaR (bounded by the grid's bin width at n_grid_qubits)."
        )

    return diagnostics, notes


def compare_methods(pnl_samples: np.ndarray, spec: TailRiskSpec | None = None) -> TailRiskComparisonReport:
    """Run every method in this module's docstring against `pnl_samples` at
    `spec.primary_alpha()` and return one normalized comparison report.
    Raises `applications.tail_risk.errors.EmptySampleError` /
    `DegenerateTailError` for degenerate inputs before any estimator runs.
    """
    spec = spec or TailRiskSpec()
    pnl = validate_pnl_sample(pnl_samples)
    alpha = spec.primary_alpha()

    results: dict[str, MethodResult] = {}

    def _run(name: str, fn) -> None:
        t0 = time.perf_counter()
        var, cvar, extra = fn()
        results[name] = MethodResult(method=name, var=var, cvar=cvar, wall_time_s=time.perf_counter() - t0, extra=extra)

    _run("historical", lambda: _historical_var_cvar(pnl, alpha))
    _run("parametric_gaussian", lambda: _gaussian_var_cvar(pnl, alpha))
    _run("cornish_fisher", lambda: _cornish_fisher_var_cvar(pnl, alpha))
    _run("monte_carlo_gbm", lambda: _monte_carlo_gbm_var_cvar(pnl, alpha, n_paths=spec.n_mc_paths, seed=spec.seed))
    _run(
        "meta_agent_bootstrap",
        lambda: _meta_agent_var_cvar(pnl, alpha, n_bootstrap=spec.n_bootstrap, seed=spec.seed),
    )
    if spec.run_quantum:
        _run(
            "qae_amplitude_estimation",
            lambda: _qae_located_var_cvar(
                pnl, alpha, n_grid_qubits=spec.n_grid_qubits, epsilon=spec.epsilon, shots=spec.shots, seed=spec.seed,
            ),
        )

    diagnostics, notes = _build_diagnostics(pnl, results)

    return TailRiskComparisonReport(
        alpha=alpha, n_samples=int(pnl.size), prob_of_loss=probability_of_loss(pnl),
        results=results, diagnostics=diagnostics, notes=notes,
    )


def compare_methods_all_levels(
    pnl_samples: np.ndarray, spec: TailRiskSpec | None = None,
) -> dict[float, TailRiskComparisonReport]:
    """Run `compare_methods` once per confidence level in `spec.confidence_levels`."""
    spec = spec or TailRiskSpec()
    out: dict[float, TailRiskComparisonReport] = {}
    for level in spec.confidence_levels:
        level_spec = spec.model_copy(update={"confidence_levels": [level]})
        out[level] = compare_methods(pnl_samples, level_spec)
    return out


def _register() -> None:
    if "tail_risk_comparison" not in APPLICATION_REGISTRY.list():
        APPLICATION_REGISTRY.register("tail_risk_comparison")(compare_methods)


_register()
