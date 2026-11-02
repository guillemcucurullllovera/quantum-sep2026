"""The standing classical-competitor gate for amplitude-estimation proposals
-- QUANTUM-1B Track E, EXP-E1.

**What this is and why it exists.** `algorithms/qae_iqae/erosion_audit.py`
already measures, with executed numbers, where AE does NOT beat the best
classical competitor on this book (QMC convergence slope, importance-
sampling variance reduction, Grover-Rudolph loading cost). Its own
`deliverables/FUTURE_DIRECTIONS.md` (2026-07-23, written by a prior phase)
names exactly one thing worth building next in this lane: *"generalise the
erosion audit into a standing rule... no quantum routine is proposed for
our book until it clears this gate... v1 exists; this promotes it to a
standing precondition."* This module is that promotion -- it does NOT
reimplement the audit (every number here comes from calling
`erosion_audit.py`'s existing, already-tested functions), it wraps the
audit's descriptive numbers into a structured, pre-committed PASS/FAIL
verdict, in the same spirit `core.evidence.compute_evidence_level` turns
raw fields into a verdict rather than leaving evidence-level claims as
free-typed assertions.

**Pre-committed thresholds (written before being applied to any new
integrand here, per this whole project's "never choose the metric after
seeing the result" convention):**
- QMC edge check: FAIL (classical already competitive) if the measured
  Sobol-QMC convergence slope `zeta_qmc` is at or beyond -0.9 (i.e.
  within 10% of the ideal -1.0 AE-matching rate) -- AE's whole quadratic
  pitch is moot if classical QMC already converges at close to the same
  asymptotic rate.
- Importance-sampling check: FAIL if the measured variance-reduction
  factor is >= 10x -- AE's edge is over PLAIN MC; a 10x+ classical
  variance reduction from IS alone means AE would need an enormous
  absolute query advantage just to catch up to the IS baseline, let alone
  beat it.
- Loading-cost check: FAIL if BOTH curves' power-law SCALING EXPONENT
  (the slope of log(cost) vs log(1/epsilon), fit across the full
  `audit_herbert_loading` epsilon grid) is within [0.8, 1.2] of 1.0 --
  Herbert's actual point is that AE's oracle-query count and the
  classical Grover-Rudolph loading cost are the SAME ASYMPTOTIC ORDER
  (both O(1/eps)), not that they are numerically close at one epsilon.
  **A same-epsilon ratio is the WRONG check here and was tried first in
  this module's own development** (AE's bound carries an extra
  `log(1/eps)` factor that inflates its constant ~77x at eps=1e-2 even
  though the scaling order genuinely matches) -- caught by this module's
  own test suite before being shipped, not discovered later.
- **Gate verdict: FAIL (do not build) if ANY of the three checks fail.**
  All three must independently clear for PASS -- this is deliberately
  strict, matching the audit's own finding that all three erode the edge
  on this book today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from algorithms.qae_iqae.erosion_audit import (
    audit_herbert_loading,
    audit_importance_sampling,
    audit_qmc_vs_mc,
)

ZETA_QMC_FAIL_THRESHOLD = -0.9
VARIANCE_REDUCTION_FAIL_THRESHOLD = 10.0
LOADING_EXPONENT_FAIL_LOW = 0.8
LOADING_EXPONENT_FAIL_HIGH = 1.2
LOADING_GATE_EPSILON = 1e-2


class GateVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    name: str
    verdict: GateVerdict
    measured_value: float
    threshold_description: str
    note: str


@dataclass(frozen=True)
class ClassicalCompetitorGateResult:
    integrand_label: str
    overall_verdict: GateVerdict
    checks: tuple[CheckResult, ...]
    raw_audit: dict

    def summary(self) -> str:
        lines = [f"Classical-competitor gate for {self.integrand_label!r}: {self.overall_verdict.value}"]
        for c in self.checks:
            lines.append(f"  [{c.verdict.value}] {c.name}: {c.measured_value:.4f} ({c.threshold_description})")
        return "\n".join(lines)


def _check_qmc_edge(qmc_result: dict) -> CheckResult:
    zeta = qmc_result["zeta_qmc"]
    verdict = GateVerdict.FAIL if zeta <= ZETA_QMC_FAIL_THRESHOLD else GateVerdict.PASS
    return CheckResult(
        name="qmc_convergence_edge",
        verdict=verdict,
        measured_value=zeta,
        threshold_description=f"FAIL if zeta_qmc <= {ZETA_QMC_FAIL_THRESHOLD} (classical QMC already near-AE-rate)",
        note="Sobol-QMC convergence slope vs the smooth payoff's true value; AE's O(1/N) pitch is moot "
             "if classical QMC already converges near that rate.",
    )


def _check_importance_sampling(is_result: dict) -> CheckResult:
    factor = is_result["variance_reduction_factor"]
    verdict = GateVerdict.FAIL if factor >= VARIANCE_REDUCTION_FAIL_THRESHOLD else GateVerdict.PASS
    return CheckResult(
        name="importance_sampling_erosion",
        verdict=verdict,
        measured_value=factor,
        threshold_description=f"FAIL if variance_reduction_factor >= {VARIANCE_REDUCTION_FAIL_THRESHOLD}x",
        note="AE's quadratic edge is over PLAIN MC; a large classical IS variance-reduction factor "
             "erodes that edge exactly where tail risk lives.",
    )


def _loglog_slope(rows: list[dict], value_key: str) -> float:
    """Fit log(value) vs log(1/epsilon) across `rows` and return the slope
    -- the empirical scaling exponent. A value growing as O(1/eps)^p has
    slope p; both curves here are claimed O(1/eps) (p=1) by Herbert 2021's
    argument, which this fits directly rather than assuming."""
    import math

    xs = [math.log(1.0 / r["epsilon"]) for r in rows]
    ys = [math.log(r[value_key]) for r in rows]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den != 0 else float("nan")


def _check_loading_cost(loading_result: dict, epsilon: float) -> CheckResult:
    rows = loading_result["rows"]
    ae_slope = _loglog_slope(rows, "ae_oracle_queries_order")
    gr_slope = _loglog_slope(rows, "grover_rudolph_classical_cdf_ops_order")
    # "same order" means BOTH slopes sit near 1.0 (O(1/eps)) -- report the
    # one further from 1.0 as the binding measured value, since either
    # curve deviating from O(1/eps) would break Herbert's "same order"
    # claim on its own.
    binding = max((ae_slope, gr_slope), key=lambda s: abs(s - 1.0))
    both_near_one = all(
        LOADING_EXPONENT_FAIL_LOW <= s <= LOADING_EXPONENT_FAIL_HIGH for s in (ae_slope, gr_slope)
    )
    verdict = GateVerdict.FAIL if both_near_one else GateVerdict.PASS
    return CheckResult(
        name="grover_rudolph_loading_cost",
        verdict=verdict,
        measured_value=binding,
        threshold_description=(
            f"FAIL if both AE-query and classical-loading log-log scaling exponents lie in "
            f"[{LOADING_EXPONENT_FAIL_LOW}, {LOADING_EXPONENT_FAIL_HIGH}] (both ~O(1/eps)); "
            f"measured ae_slope={ae_slope:.3f}, gr_slope={gr_slope:.3f}"
        ),
        note="Grover-Rudolph classical CDF-loading cost vs AE's own oracle-query count both scale as "
             "O(1/eps) (Herbert 2021) -- checked here as matching log-log slopes across the audit's "
             "epsilon grid, not a same-epsilon ratio (which is dominated by AE's extra log(1/eps) "
             "factor and would misleadingly suggest the costs differ in ORDER when they don't).",
    )


def evaluate_classical_competitor_gate(
    *,
    integrand_label: str,
    qmc_kwargs: dict | None = None,
    is_kwargs: dict | None = None,
    loading_epsilon: float = LOADING_GATE_EPSILON,
    seed: int = 0,
) -> ClassicalCompetitorGateResult:
    """Run all three erosion-audit checks (reusing `erosion_audit.py`
    verbatim) and return a structured PASS/FAIL gate verdict. `qmc_kwargs`/
    `is_kwargs` pass through to `audit_qmc_vs_mc`/`audit_importance_sampling`
    for a caller who wants to point the audit at a different payoff/barrier
    shape than the defaults (BS bull-call-spread / crash-barrier) -- the
    THRESHOLDS above are fixed regardless of which integrand is audited,
    per this module's whole point (a standing precondition, not a
    per-integrand negotiation)."""
    qmc_result = audit_qmc_vs_mc(seed=seed, **(qmc_kwargs or {}))
    is_result = audit_importance_sampling(seed=seed + 1, **(is_kwargs or {}))
    loading_result = audit_herbert_loading(epsilons=(loading_epsilon, 1e-3, 1e-4))

    checks = (
        _check_qmc_edge(qmc_result),
        _check_importance_sampling(is_result),
        _check_loading_cost(loading_result, loading_epsilon),
    )
    overall = GateVerdict.FAIL if any(c.verdict == GateVerdict.FAIL for c in checks) else GateVerdict.PASS

    return ClassicalCompetitorGateResult(
        integrand_label=integrand_label,
        overall_verdict=overall,
        checks=checks,
        raw_audit={"A_qmc_vs_mc": qmc_result, "B_importance_sampling": is_result, "C_herbert_loading": loading_result},
    )


__all__ = [
    "GateVerdict",
    "CheckResult",
    "ClassicalCompetitorGateResult",
    "evaluate_classical_competitor_gate",
    "ZETA_QMC_FAIL_THRESHOLD",
    "VARIANCE_REDUCTION_FAIL_THRESHOLD",
    "LOADING_EXPONENT_FAIL_LOW",
    "LOADING_EXPONENT_FAIL_HIGH",
]
