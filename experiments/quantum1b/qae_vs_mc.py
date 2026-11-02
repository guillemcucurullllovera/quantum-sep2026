"""Rigorous, repeated-trial statistical comparison: IQAE vs classical Monte
Carlo on the IDENTICAL crash-barrier estimand -- QUANTUM-1B Track E,
EXP-E2 (ideal) / EXP-E3 (FakeKingston-noisy).

**Why this is new, not a duplicate of `erosion_audit.py`.** The existing
audit measures asymptotic CONVERGENCE RATE (RMSE vs N, fit to a power-law
slope) across increasing budgets, and is entirely classical (MC vs QMC,
no quantum circuit ever runs). What it does NOT measure -- and what this
module adds -- is the quantum estimator's own empirical statistical
behavior across repeated trials AT A FIXED, REALISTIC BUDGET: bias,
variance, and (the one `erosion_audit.py` genuinely lacks) **empirical
confidence-interval coverage** -- what fraction of N independently-run
IQAE confidence intervals actually contain the true value, which is the
practical question "can I trust this CI" rather than "how fast does error
shrink in principle."

**Reuses, does not reimplement:** `algorithms.qae_iqae.iqae.run_iqae` for
every quantum estimate, `algorithms.qae_iqae.finance_integrands.
build_crash_barrier` for the estimand, `algorithms.qae_iqae.backends` for
the ideal/FakeKingston simulators.

**Budget-matching, corrected after review (this is the single most
important methodological point in this module, so it is stated carefully
rather than asserted once and trusted):** `shots` is NOT a reliable
budget-matching unit across both epsilon regimes this module uses, and an
earlier version of this docstring incorrectly claimed it was always
budget-matched. At `EPSILON_NOISY=0.05` (the noisy/loose-epsilon
comparison, `compare_qae_vs_mc_noisy`), IQAE empirically needs ZERO Grover
amplification (`oracle_queries=0` -- directly measured, see the
regression-guarding test `test_noisy_epsilon_is_loose_by_default`), so
`shots` genuinely IS the comparable per-trial query count there. At
`EPSILON_IDEAL=0.01` (`compare_qae_vs_mc_ideal`), IQAE DOES perform real
Grover amplification -- measured directly: `mean_oracle_queries≈16247`
against a classical `shots=2048`, an ~8x unmatched budget favoring IQAE,
which would make a lower-RMSE finding there meaningless as a speedup claim
(any method given 8x more queries gets better precision, trivially).
**Fix:** `compare_qae_vs_mc_ideal` now ALSO runs a `classical_mc_budget_matched`
estimator, using classical MC sample counts equal to IQAE's own measured
mean `oracle_queries` per trial (the standard "oracle-query-complexity"
unit used throughout the AE literature's quadratic-speedup claims, e.g.
Brassard et al. 2002, Grinko et al. 2021 -- one Grover-oracle application
is the unit both a classical sample and a quantum query are compared
against). The ORIGINAL (shots-only) `classical_mc` summary is still
reported too, labeled explicitly as unmatched in the ideal case, so
neither comparison is silently hidden -- see
`QaeVsMcComparisonResult.classical_mc_budget_matched` and its `None` value
in the noisy path, where it would be redundant with `classical_mc` (both
already use the same `shots` budget there).

**Resource/depth discipline, learned directly from Track D's own
incident record (`docs/quantum/QUANTUM1B_HARDWARE_READINESS.md`):** a
tight epsilon drives IQAE into deep Grover-amplified circuits, which
Track D already showed become extremely expensive once routed onto
FakeKingston's real coupling map. This module deliberately uses a LOOSE
epsilon for the noisy comparison (per `algorithms/qae_iqae/deliverables/
FUTURE_DIRECTIONS.md`'s own explicit recommendation: "prefer a loose eps
(shallow Grover)") -- directly measured before committing to this design:
at epsilon in {0.03, 0.05, 0.1} against FakeKingston, `oracle_queries=0`
and each run takes ~5-8s; at epsilon=0.01 (ideal only, never run against
FakeKingston here) a single run takes ~4s with real Grover amplification
(14336 oracle queries).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from algorithms.qae_iqae.barrier_circuits import build_A_barrier
from algorithms.qae_iqae.finance_integrands import BarrierIntegrand
from algorithms.qae_iqae.iqae import run_iqae

EPSILON_IDEAL = 0.01  # real Grover amplification, safe under ideal (fast) simulation only
EPSILON_NOISY = 0.05  # loose -- oracle_queries=0 at this epsilon, per the calibration note above
ALPHA = 0.05
SHOTS = 2048


@dataclass
class TrialResult:
    a_hat: float
    ci_lo: float
    ci_hi: float
    oracle_queries: int
    runtime_s: float
    ci_contains_truth: bool


@dataclass
class EstimatorSummary:
    method: str  # "iqae" | "classical_mc"
    n_trials: int
    mean_a_hat: float
    bias: float  # mean_a_hat - true_value
    variance: float  # across trials
    rmse: float
    mean_ci_halfwidth: float
    empirical_ci_coverage: float  # fraction of trials whose CI contains true_value
    mean_oracle_queries: float
    mean_runtime_s: float
    trials: list[TrialResult] = field(default_factory=list)


@dataclass
class QaeVsMcComparisonResult:
    integrand_label: str
    true_value: str  # which ground truth this was measured against ("exact_grid_a")
    true_value_numeric: float
    backend_label: str
    epsilon: float
    alpha: float
    shots: int
    n_trials: int
    iqae: EstimatorSummary
    classical_mc: EstimatorSummary
    """Classical MC at `shots` samples/trial -- budget-matched to IQAE ONLY
    when IQAE's own oracle_queries==0 (the noisy/loose-epsilon path).
    In the ideal path this is an UNMATCHED comparison (IQAE used far more
    queries) -- see `classical_mc_budget_matched` for the fair comparison."""
    classical_mc_budget_matched: EstimatorSummary | None = None
    """Classical MC at n_samples == iqae.mean_oracle_queries per trial --
    the methodologically fair comparison at matched total query count. None
    in the noisy path, where it would be identical to `classical_mc` (both
    already use `shots`, since oracle_queries==0 there by construction)."""


def _summarize(method: str, trials: list[TrialResult], true_value: float) -> EstimatorSummary:
    a_hats = np.array([t.a_hat for t in trials])
    bias = float(np.mean(a_hats) - true_value)
    variance = float(np.var(a_hats, ddof=1)) if len(a_hats) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((a_hats - true_value) ** 2)))
    coverage = float(np.mean([t.ci_contains_truth for t in trials]))
    mean_halfwidth = float(np.mean([(t.ci_hi - t.ci_lo) / 2.0 for t in trials]))
    mean_oracle = float(np.mean([t.oracle_queries for t in trials]))
    mean_runtime = float(np.mean([t.runtime_s for t in trials]))
    return EstimatorSummary(
        method=method,
        n_trials=len(trials),
        mean_a_hat=float(np.mean(a_hats)),
        bias=bias,
        variance=variance,
        rmse=rmse,
        mean_ci_halfwidth=mean_halfwidth,
        empirical_ci_coverage=coverage,
        mean_oracle_queries=mean_oracle,
        mean_runtime_s=mean_runtime,
        trials=trials,
    )


def _wilson_ci(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- the standard
    classical analogue to IQAE's Clopper-Pearson CI, used here so the
    classical comparator gets an equally rigorous (not a naive normal-
    approximation) interval."""
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def run_classical_mc_trials(
    barrier: BarrierIntegrand, *, n_samples: int, n_trials: int, alpha: float, seed: int
) -> list[TrialResult]:
    """Classical comparator: draw `n_samples` i.i.d. categorical samples
    from the SAME discretized grid distribution (`barrier.probs`) the
    quantum circuit encodes, per trial, and report the empirical
    proportion below the barrier plus a Wilson CI -- the exact same
    estimand (`exact_grid_a`, the grid-quantized probability), not the
    continuous analytic value, so this is a genuinely fair comparison
    against IQAE's own target (both IQAE and classical MC here estimate
    the grid-discretized quantity, not the true BS CDF -- the
    discretisation gap is a separate, already-documented effect in
    `finance_integrands.py`, orthogonal to what this module measures)."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(barrier.probs, dtype=float)
    mask = np.asarray(barrier.barrier_mask, dtype=bool)
    true_value = barrier.exact_grid_a

    trials = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        draws = rng.choice(len(probs), size=n_samples, p=probs)
        k = int(np.sum(mask[draws]))
        a_hat = k / n_samples
        ci_lo, ci_hi = _wilson_ci(k, n_samples, alpha)
        runtime = time.perf_counter() - t0
        trials.append(
            TrialResult(
                a_hat=a_hat, ci_lo=ci_lo, ci_hi=ci_hi, oracle_queries=n_samples,
                runtime_s=runtime, ci_contains_truth=(ci_lo <= true_value <= ci_hi),
            )
        )
    return trials


def run_iqae_trials(
    barrier: BarrierIntegrand, *, backend, transpiler_factory, n_trials: int,
    epsilon: float, alpha: float, shots: int, seed: int,
) -> list[TrialResult]:
    """Runs `run_iqae` (reused verbatim from `algorithms.qae_iqae.iqae`)
    `n_trials` times with independent seeds, against `backend` (an ideal
    or FakeKingston-noisy Aer sampler, injected so this function doesn't
    care which)."""
    A = build_A_barrier(barrier.probs, barrier.barrier_mask)
    anc = int(round(np.log2(len(barrier.probs))))
    true_value = barrier.exact_grid_a

    trials = []
    for i in range(n_trials):
        trial_seed = seed + i
        t0 = time.perf_counter()
        sampler = backend(shots=shots, seed=trial_seed)
        transpiler = transpiler_factory()
        res = run_iqae(A, anc, sampler=sampler, transpiler=transpiler, epsilon=epsilon, alpha=alpha)
        runtime = time.perf_counter() - t0
        ci_lo, ci_hi = res["ci"]
        trials.append(
            TrialResult(
                a_hat=res["a_hat"], ci_lo=ci_lo, ci_hi=ci_hi,
                oracle_queries=res["oracle_queries"], runtime_s=runtime,
                ci_contains_truth=(ci_lo <= true_value <= ci_hi),
            )
        )
    return trials


def compare_qae_vs_mc_ideal(
    barrier: BarrierIntegrand, *, n_trials: int = 30, epsilon: float = EPSILON_IDEAL,
    alpha: float = ALPHA, shots: int = SHOTS, seed: int = 0,
) -> QaeVsMcComparisonResult:
    """EXP-E2: ideal-simulator comparison. Fast enough (per-trial ~4s at
    epsilon=0.01 with real Grover amplification, measured directly before
    committing to this design) to run a statistically meaningful number of
    trials (default 30)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2

    def backend_factory(shots: int, seed: int):
        return SamplerV2(default_shots=shots, seed=seed)

    def transpiler_factory():
        return generate_preset_pass_manager(1, AerSimulator())

    iqae_trials = run_iqae_trials(
        barrier, backend=backend_factory, transpiler_factory=transpiler_factory,
        n_trials=n_trials, epsilon=epsilon, alpha=alpha, shots=shots, seed=seed,
    )
    mc_trials = run_classical_mc_trials(barrier, n_samples=shots, n_trials=n_trials, alpha=alpha, seed=seed + 1000)
    iqae_summary = _summarize("iqae", iqae_trials, barrier.exact_grid_a)

    # Budget-matched comparator: classical MC given the SAME total query
    # count IQAE actually used (mean oracle_queries per trial), not the
    # unmatched `shots` value -- see this module's docstring for why this
    # is the methodologically fair comparison in the ideal (tight-epsilon,
    # real-Grover-amplification) path. A separate seed offset keeps this
    # draw independent of the unmatched `mc_trials` above.
    matched_n_samples = max(1, int(round(iqae_summary.mean_oracle_queries)))
    matched_trials = run_classical_mc_trials(
        barrier, n_samples=matched_n_samples, n_trials=n_trials, alpha=alpha, seed=seed + 2000
    )

    return QaeVsMcComparisonResult(
        integrand_label=barrier.label, true_value="exact_grid_a", true_value_numeric=barrier.exact_grid_a,
        backend_label="ideal", epsilon=epsilon, alpha=alpha, shots=shots, n_trials=n_trials,
        iqae=iqae_summary,
        classical_mc=_summarize("classical_mc", mc_trials, barrier.exact_grid_a),
        classical_mc_budget_matched=_summarize(
            f"classical_mc_budget_matched(n={matched_n_samples})", matched_trials, barrier.exact_grid_a
        ),
    )


def compare_qae_vs_mc_noisy(
    barrier: BarrierIntegrand, *, n_trials: int = 10, epsilon: float = EPSILON_NOISY,
    alpha: float = ALPHA, shots: int = SHOTS, seed: int = 0,
) -> QaeVsMcComparisonResult:
    """EXP-E3: FakeKingston-noisy comparison, loose epsilon (see module
    docstring), small trial count (default 10, matching Track B/D's own
    bounded-noisy-trial convention) given the higher per-trial cost of a
    noisy simulation."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer.primitives import SamplerV2
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    fake = FakeKingston()
    noise = NoiseModel.from_backend(fake)

    def backend_factory(shots: int, seed: int):
        return SamplerV2(default_shots=shots, seed=seed, options={"backend_options": {"noise_model": noise}})

    def transpiler_factory():
        return generate_preset_pass_manager(0, fake)

    iqae_trials = run_iqae_trials(
        barrier, backend=backend_factory, transpiler_factory=transpiler_factory,
        n_trials=n_trials, epsilon=epsilon, alpha=alpha, shots=shots, seed=seed,
    )
    mc_trials = run_classical_mc_trials(barrier, n_samples=shots, n_trials=n_trials, alpha=alpha, seed=seed + 1000)

    return QaeVsMcComparisonResult(
        integrand_label=barrier.label, true_value="exact_grid_a", true_value_numeric=barrier.exact_grid_a,
        backend_label="fake_kingston", epsilon=epsilon, alpha=alpha, shots=shots, n_trials=n_trials,
        iqae=_summarize("iqae", iqae_trials, barrier.exact_grid_a),
        classical_mc=_summarize("classical_mc", mc_trials, barrier.exact_grid_a),
    )


@dataclass(frozen=True)
class SignificanceTestResult:
    """Two-sided Mann-Whitney U test comparing two estimators' PER-TRIAL
    absolute errors (`|a_hat - true_value|`) -- non-parametric (no
    normality assumption, appropriate for the small trial counts, n=10-30,
    used throughout this module) and comparing absolute error rather than
    signed error since the question is "which estimator is more precise,"
    not "which is biased in which direction" (bias is reported separately
    via `EstimatorSummary.bias`). The two estimators' trials are
    STATISTICALLY INDEPENDENT samples here (different RNG seeds/streams
    per estimator, not matched pairs by trial index), so Mann-Whitney
    (unpaired) is the correct test, not a paired test like Wilcoxon
    signed-rank."""

    method_a: str
    method_b: str
    n_a: int
    n_b: int
    median_abs_error_a: float
    median_abs_error_b: float
    u_statistic: float
    p_value: float
    a_significantly_more_precise: bool  # True iff p < 0.05 AND median_abs_error_a < median_abs_error_b
    alpha: float = 0.05


def paired_significance_test(
    summary_a: EstimatorSummary, summary_b: EstimatorSummary, true_value: float, *, alpha: float = 0.05
) -> SignificanceTestResult:
    """Mann-Whitney U test on `|a_hat - true_value|` between two
    `EstimatorSummary`s' own trial lists -- see `SignificanceTestResult`
    docstring for why this test, not a paired one. Reusable for any two
    estimators this module produces (IQAE vs classical_mc, or
    classical_mc vs classical_mc_budget_matched, etc.), not hardcoded to
    one specific pair."""
    from scipy.stats import mannwhitneyu

    errors_a = np.array([abs(t.a_hat - true_value) for t in summary_a.trials])
    errors_b = np.array([abs(t.a_hat - true_value) for t in summary_b.trials])
    u_stat, p_value = mannwhitneyu(errors_a, errors_b, alternative="two-sided")

    med_a = float(np.median(errors_a))
    med_b = float(np.median(errors_b))

    return SignificanceTestResult(
        method_a=summary_a.method, method_b=summary_b.method,
        n_a=len(errors_a), n_b=len(errors_b),
        median_abs_error_a=med_a, median_abs_error_b=med_b,
        u_statistic=float(u_stat), p_value=float(p_value),
        a_significantly_more_precise=bool(p_value < alpha and med_a < med_b),
        alpha=alpha,
    )


__all__ = [
    "TrialResult",
    "EstimatorSummary",
    "QaeVsMcComparisonResult",
    "SignificanceTestResult",
    "run_classical_mc_trials",
    "run_iqae_trials",
    "compare_qae_vs_mc_ideal",
    "compare_qae_vs_mc_noisy",
    "paired_significance_test",
    "EPSILON_IDEAL",
    "EPSILON_NOISY",
    "ALPHA",
    "SHOTS",
]
