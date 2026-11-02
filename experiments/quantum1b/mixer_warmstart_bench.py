"""Track F, EXP-F1 -- QAOA parameter-initialization / warm-start benchmark.

**Why this exists and what it does NOT do.** `docs/quantum/QUANTUM1B_ESTATE_INVENTORY.md`
(lane B) found the estate already has `algorithms/qaoa_qubo/gen2`'s
cardinality-preserving XY-mixer/Dicke-state ansatz AND
`algorithms_extended/qaoa_initialization`'s INTERP (Zhou 2020) and
linear-ramp parameter-transfer strategies -- neither ever benchmarked
against Track B's own CVaR-QAOA engine (`experiments/quantum1b/cvar_qaoa.py`,
standard RX-mixer, cold-start-random).

**The XY/Dicke-mixer comparison is NOT attempted here, and that is a
deliberate, documented decision, not an oversight:** `cvar_qaoa.py`'s own
module docstring already explains why Gen2's XY-mixer was not reused to
build Track B in the first place -- it is cardinality-preserving
(assumes a FIXED Hamming weight via Dicke-state preparation,
`QAOAConfig.target_k`), which is structurally incompatible with this
package's QUBO encoding (`qubo_encoding.py`), whose SLACK qubits encode
inequality-constraint remainders and must range freely, not sit at a
fixed weight. Forcing a fixed-weight mixer onto those slack bits would
not be a fair "same problem, different ansatz" comparison -- it would
silently change which problem is being solved. This is reported here as
an honest structural non-comparison, not silently skipped: see "What this
experiment does NOT claim" below.

**What IS benchmarked:** three parameter-initialization strategies for
Track B's own existing RX-mixer CVaR-QAOA engine, all operating purely on
the `(gamma, beta)` angle arrays (so none of them has the XY-mixer's
fixed-weight incompatibility):

1. `cold_start_random` -- the engine's own existing default (uniform
   random in `(0, pi/2)`, 5 seeds), the baseline this whole comparison is
   against.
2. `linear_ramp` -- `algorithms_extended.qaoa_initialization.linear_ramp_initialization`,
   a deterministic adiabatic-inspired schedule (gamma ramps up, beta ramps
   down across layers). Deterministic -> one run per (instance, p), not
   five.
3. `interp` -- `algorithms_extended.qaoa_initialization.interpolate_to_next_depth`
   (Zhou et al. 2020), which extrapolates an OPTIMIZED depth-p solution to
   an initial guess for depth p+1. Only meaningful p=1 -> p=2 (this
   package's own ladder depth), one INTERP run per p=1 cold-start seed
   (5 total, paired with the cold-start baseline's own 5 p=1 results so
   the comparison at p=2 is against the SAME five p=1 starting points).

**Reuses, does not reimplement:** `experiments.quantum1b.cvar_qaoa.run_expectation_qaoa`
(now with an added optional `initial_params` override -- backward-
compatible, see that module's own docstring), `exact_distribution_metrics`
for true-objective decoding, `experiments.quantum1b.classical_exact.solve_exact_milp`
(Track A) for the ground truth, and
`experiments.quantum1b.hardware_readiness.frozen_controls.FROZEN_CONTROLS`
for the instance set (already pinned, already has the exact optimum
computed -- reusing it here also means this benchmark is running on
exactly the same instances the Hardware Day frozen-control submission
would use, a nice side benefit, not the primary reason for the choice).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms_extended.qaoa_initialization import (
    interpolate_to_next_depth,
    linear_ramp_initialization,
)
from experiments.quantum1b.cvar_qaoa import exact_distribution_metrics, run_expectation_qaoa
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.hardware_readiness.frozen_controls import FROZEN_CONTROLS
from experiments.quantum1b.qubo_encoding import build_qubo

MAXITER = 200  # matches EXP-B1's own convention
SEEDS = (0, 1, 2, 3, 4)  # matches EXP-B1's own convention


@dataclass(frozen=True)
class StrategyRunResult:
    strategy: str  # "cold_start_random" | "linear_ramp" | "interp"
    family: str
    n_candidates: int
    p: int
    seed: int | None  # None for linear_ramp (deterministic, no seed)
    n_evaluations: int
    final_training_objective: float
    best_feasible_objective: float | None
    objective_gap: float | None  # (exact_optimum - best_feasible_objective) / |exact_optimum|
    feasible_fraction: float
    runtime_s: float


def _objective_gap(exact_optimum: float, best_feasible: float | None) -> float | None:
    if best_feasible is None or exact_optimum == 0:
        return None
    return (exact_optimum - best_feasible) / abs(exact_optimum)


def _run_one(
    strategy: str, qubo, instance, *, p: int, seed: int | None, exact_optimum: float,
    initial_params: np.ndarray | None,
) -> tuple[StrategyRunResult, np.ndarray]:
    """Returns (result, best_params) -- best_params is needed by the
    INTERP strategy (extrapolated from a p=1 cold-start's OWN optimized
    params), not just reported."""
    r = run_expectation_qaoa(
        qubo, p=p, seed=(seed if seed is not None else 0), maxiter=MAXITER, initial_params=initial_params
    )
    m = exact_distribution_metrics(r, qubo, instance)
    result = StrategyRunResult(
        strategy=strategy, family=instance.family, n_candidates=len(instance.candidates), p=p, seed=seed,
        n_evaluations=r.n_evaluations, final_training_objective=r.final_training_objective,
        best_feasible_objective=m.best_feasible_objective,
        objective_gap=_objective_gap(exact_optimum, m.best_feasible_objective),
        feasible_fraction=m.feasible_fraction, runtime_s=r.runtime_s,
    )
    return result, r.best_params


def run_strategy_comparison(
    *, family: str, n_candidates: int, seed: int, exact_optimum: float,
) -> dict[str, list[StrategyRunResult]]:
    """Runs all three strategies on one instance, returns a dict keyed by
    strategy name. `family`/`n_candidates`/`seed` identify the INSTANCE
    (reused from `generate_instance`, matching Track B's own convention);
    `exact_optimum` is Track A's already-computed ground truth for that
    exact instance (callers pass `FROZEN_CONTROLS[i].expected_exact_optimum`,
    not recomputed here, to stay consistent with the frozen-control pin)."""
    instance = generate_instance(family, n_candidates, seed=seed)
    qubo = build_qubo(instance)

    cold_start_p1: list[StrategyRunResult] = []
    cold_start_p1_params: list[np.ndarray] = []
    for s in SEEDS:
        res, params = _run_one(
            "cold_start_random", qubo, instance, p=1, seed=s, exact_optimum=exact_optimum, initial_params=None
        )
        cold_start_p1.append(res)
        cold_start_p1_params.append(params)

    cold_start_p2: list[StrategyRunResult] = []
    for s in SEEDS:
        res, _ = _run_one(
            "cold_start_random", qubo, instance, p=2, seed=s, exact_optimum=exact_optimum, initial_params=None
        )
        cold_start_p2.append(res)

    linear_ramp_results: list[StrategyRunResult] = []
    for p in (1, 2):
        gammas, betas = linear_ramp_initialization(p)
        x0 = np.concatenate([gammas, betas])
        res, _ = _run_one(
            "linear_ramp", qubo, instance, p=p, seed=None, exact_optimum=exact_optimum, initial_params=x0
        )
        linear_ramp_results.append(res)

    interp_results: list[StrategyRunResult] = []
    for s, p1_params in zip(SEEDS, cold_start_p1_params):
        gammas1, betas1 = p1_params[:1], p1_params[1:]
        gammas2, betas2 = interpolate_to_next_depth(gammas1, betas1)
        x0 = np.concatenate([gammas2, betas2])
        res, _ = _run_one(
            "interp", qubo, instance, p=2, seed=s, exact_optimum=exact_optimum, initial_params=x0
        )
        interp_results.append(res)

    return {
        "cold_start_random_p1": cold_start_p1,
        "cold_start_random_p2": cold_start_p2,
        "linear_ramp": linear_ramp_results,
        "interp_p2": interp_results,
    }


def run_frozen_controls_comparison() -> dict[str, dict[str, list[StrategyRunResult]]]:
    """Runs `run_strategy_comparison` for every instance in
    `FROZEN_CONTROLS` -- the actual EXP-F1 sweep."""
    out = {}
    for control in FROZEN_CONTROLS:
        key = f"{control.family}_n{control.n_candidates}"
        out[key] = run_strategy_comparison(
            family=control.family, n_candidates=control.n_candidates, seed=control.seed,
            exact_optimum=control.expected_exact_optimum,
        )
    return out


__all__ = [
    "StrategyRunResult",
    "MAXITER",
    "SEEDS",
    "run_strategy_comparison",
    "run_frozen_controls_comparison",
]
