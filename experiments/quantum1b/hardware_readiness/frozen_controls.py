"""Frozen classical control set for Hardware Day.

A fixed, small, PRE-COMMITTED list of (family, n_candidates, seed)
instances -- the exact same instances a real Hardware Day run would submit
unchanged -- together with their deterministic fingerprints and Track A's
already-computed exact optimum. This is the concrete artifact
"frozen classical controls" in the overnight directive's checklist refers
to: not a new classical solver (Track A already has one, `READY` per
`docs/quantum/QUANTUM1B_PHASE1_CLOSEOUT.md`), but a frozen, checkable SET
OF INSTANCES so nobody picks a different, easier instance at Hardware Day
time than what was used for the simulator-evidence comparison.

`verify_frozen_controls()` regenerates each instance from
`experiments.quantum1b.generators.generate_instance` and Track A's own
`classical_exact.solve_exact_milp`, and checks the fingerprint AND the
exact optimum still match what is recorded below -- catching silent drift
if `generators.py`/`qubo_encoding.py`/`classical_exact.py` ever change in
a way that would make "the same instance" quietly become a different one.

**Also pins Track B's own simulator baseline** (expectation-QAOA and
CVaR-QAOA best true-objective, at EXP-B1's own `p=1, alpha=0.5, seed=0,
maxiter=200` convention) for each frozen instance -- this is what
`docs/quantum/QUANTUM1B_HARDWARE_DAY_RUNBOOK.md` step 2 needs ("compare
real-hardware ... against ... Track B's own simulator result for the
identical instance") ready-computed rather than something a Hardware Day
operator has to remember to (re)compute correctly under time pressure.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.quantum1b.classical_exact import solve_exact_milp
from experiments.quantum1b.cvar_qaoa import (
    exact_distribution_metrics,
    run_cvar_qaoa,
    run_expectation_qaoa,
)
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.qubo_encoding import build_qubo

SIMULATOR_BASELINE_P = 1
SIMULATOR_BASELINE_ALPHA = 0.5  # EXP-B1's convention for the CVaR variant
SIMULATOR_BASELINE_SEED = 0
SIMULATOR_BASELINE_MAXITER = 200


@dataclass(frozen=True)
class FrozenControlInstance:
    family: str
    n_candidates: int
    seed: int
    expected_fingerprint: str
    expected_n_qubits: int
    expected_exact_optimum: float
    expected_simulator_expectation_qaoa_best_objective: float = float("nan")
    expected_simulator_cvar_qaoa_best_objective: float = float("nan")


# Deliberately tiny and deliberately reused from EXP-B1's own already-
# validated ladder (same family/n/seed cells EXP-B1 already ran and
# reported) -- Hardware Day should never be the first time these specific
# instances are seen, and this set is small enough (3 instances) to be
# cheap even on real hardware shot budgets.
FROZEN_CONTROL_SET: tuple[FrozenControlInstance, ...] = (
    FrozenControlInstance(
        family="independent", n_candidates=4, seed=0,
        expected_fingerprint="",  # filled by `_freeze()` below at import time -- see note
        expected_n_qubits=0,
        expected_exact_optimum=0.0,
    ),
    FrozenControlInstance(
        family="correlated", n_candidates=4, seed=0,
        expected_fingerprint="",
        expected_n_qubits=0,
        expected_exact_optimum=0.0,
    ),
    FrozenControlInstance(
        family="tight_budget", n_candidates=4, seed=0,
        expected_fingerprint="",
        expected_n_qubits=0,
        expected_exact_optimum=0.0,
    ),
)


def _freeze() -> tuple[FrozenControlInstance, ...]:
    """Computes the actual expected fingerprint/n_qubits/exact_optimum for
    each entry in `FROZEN_CONTROL_SET` ONCE, at module import time, from
    the current code -- this is what actually gets "frozen" into this
    module's `FROZEN_CONTROLS` (the placeholders above document intent;
    this function computes the real pinned values so they don't have to be
    hand-copied and kept in sync manually)."""
    frozen = []
    for spec in FROZEN_CONTROL_SET:
        inst = generate_instance(spec.family, spec.n_candidates, seed=spec.seed)
        qubo = build_qubo(inst)
        exact = solve_exact_milp(inst)
        exp_metrics, cvar_metrics = _simulator_baseline(inst, qubo)
        frozen.append(
            FrozenControlInstance(
                family=spec.family,
                n_candidates=spec.n_candidates,
                seed=spec.seed,
                expected_fingerprint=inst.fingerprint(),
                expected_n_qubits=qubo.n_qubits(),
                expected_exact_optimum=exact.objective,
                expected_simulator_expectation_qaoa_best_objective=exp_metrics.best_feasible_objective
                if exp_metrics.best_feasible_objective is not None
                else float("nan"),
                expected_simulator_cvar_qaoa_best_objective=cvar_metrics.best_feasible_objective
                if cvar_metrics.best_feasible_objective is not None
                else float("nan"),
            )
        )
    return tuple(frozen)


def _simulator_baseline(inst, qubo):
    """Runs EXP-B1's own convention (see module-level constants) once for
    expectation-QAOA and once for CVaR-QAOA, returning each run's
    `exact_distribution_metrics` (the same post-hoc evaluator EXP-B1 itself
    uses -- true constrained objective, never the QUBO's own claimed
    cost)."""
    exp_result = run_expectation_qaoa(
        qubo, p=SIMULATOR_BASELINE_P, seed=SIMULATOR_BASELINE_SEED, maxiter=SIMULATOR_BASELINE_MAXITER
    )
    exp_metrics = exact_distribution_metrics(exp_result, qubo, inst)

    cvar_result = run_cvar_qaoa(
        qubo, p=SIMULATOR_BASELINE_P, alpha=SIMULATOR_BASELINE_ALPHA,
        seed=SIMULATOR_BASELINE_SEED, maxiter=SIMULATOR_BASELINE_MAXITER,
    )
    cvar_metrics = exact_distribution_metrics(cvar_result, qubo, inst, cvar_alpha=SIMULATOR_BASELINE_ALPHA)

    return exp_metrics, cvar_metrics


FROZEN_CONTROLS: tuple[FrozenControlInstance, ...] = _freeze()
"""Computed fresh at import time from the CURRENT code -- NOT what
`verify_frozen_controls()` compares against by default any more (fixed
2026-10-01, remediating independent-review finding P1-5 -- see
`load_committed_snapshot()` below). Because this is recomputed on every
import, a change to `generators.py`/`qubo_encoding.py`/`classical_exact.py`
changes `FROZEN_CONTROLS` itself on the next import too, which is exactly
why comparing it against itself could never catch drift. `FROZEN_CONTROLS`
remains useful as the thing `generate_frozen_controls_snapshot.py` writes
to disk, and as an explicit `against=` argument for a caller that
deliberately wants "does current code agree with itself" rather than
"does current code agree with the committed snapshot" -- but
`verify_frozen_controls()`'s default argument no longer is this."""


_SNAPSHOT_PATH = Path(__file__).parent / "frozen_controls_snapshot.json"

_SNAPSHOT_FIELDS = (
    "family", "n_candidates", "seed", "expected_fingerprint", "expected_n_qubits",
    "expected_exact_optimum", "expected_simulator_expectation_qaoa_best_objective",
    "expected_simulator_cvar_qaoa_best_objective",
)


def write_snapshot(
    controls: tuple[FrozenControlInstance, ...] = FROZEN_CONTROLS,
    path: Path | str = _SNAPSHOT_PATH,
) -> None:
    """Writes `controls` to a committed JSON snapshot file. Deliberately
    NOT called automatically anywhere -- regenerating the snapshot is a
    conscious, reviewed act (run
    `generate_frozen_controls_snapshot.py` and commit the result), never an
    automatic side effect of importing this module or running tests."""
    payload = {
        "_comment": (
            "Committed frozen-controls snapshot. verify_frozen_controls() "
            "compares a FRESH recomputation from current code against THIS "
            "FILE by default -- regenerate only via "
            "generate_frozen_controls_snapshot.py, with an explicit commit "
            "and review, never automatically."
        ),
        "frozen_controls": [{k: v for k, v in asdict(c).items() if k in _SNAPSHOT_FIELDS} for c in controls],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_committed_snapshot(path: Path | str = _SNAPSHOT_PATH) -> tuple[FrozenControlInstance, ...]:
    """Loads the committed frozen-controls snapshot from disk.

    **Added 2026-10-01, remediating independent-review finding P1-5**
    ("'Frozen' classical controls are not actually frozen; the
    drift-detection mechanism cannot catch the failure mode it is named
    for"). Previously `verify_frozen_controls()` defaulted to comparing a
    fresh recomputation against `FROZEN_CONTROLS` -- but `FROZEN_CONTROLS`
    is ITSELF a fresh recomputation from the same import of the same code,
    so the check was tautological: it could only ever fail if a single
    process produced non-deterministic results, never if
    `generators.py`/`qubo_encoding.py`/`classical_exact.py` changed between
    when the controls were chosen and when Hardware Day runs the check.
    Loading from this committed, independently-versioned file instead means
    a later semantic change to those modules changes what gets
    RECOMPUTED, while this file stays exactly as committed -- so the two
    sides of the comparison can actually disagree, which is the entire
    point of a drift check."""
    data = json.loads(Path(path).read_text())
    return tuple(
        FrozenControlInstance(**{k: d[k] for k in _SNAPSHOT_FIELDS}) for d in data["frozen_controls"]
    )


@dataclass(frozen=True)
class FrozenControlVerification:
    family: str
    n_candidates: int
    seed: int
    fingerprint_matches: bool
    n_qubits_matches: bool
    exact_optimum_matches: bool
    simulator_baseline_matches: bool = True

    @property
    def all_match(self) -> bool:
        return (
            self.fingerprint_matches
            and self.n_qubits_matches
            and self.exact_optimum_matches
            and self.simulator_baseline_matches
        )


def verify_frozen_controls(
    against: tuple[FrozenControlInstance, ...] | None = None,
    *,
    tolerance: float = 1e-6,
    check_simulator_baseline: bool = True,
) -> list[FrozenControlVerification]:
    """Regenerates every instance in `against` fresh from current code and
    checks it still matches. Returns one `FrozenControlVerification` per
    entry; callers should assert `all(v.all_match for v in results)`
    before trusting the frozen set is still frozen.

    **`against` now defaults to `load_committed_snapshot()` (the committed
    `frozen_controls_snapshot.json` file), not `FROZEN_CONTROLS` -- fixed
    2026-10-01, remediating P1-5.** Comparing against `FROZEN_CONTROLS`
    (recomputed at every import, from the same code this function itself
    uses to recompute `inst`/`qubo`/`exact`) could never detect drift: both
    sides would always move together. Comparing against the committed
    snapshot file can: the file only changes when someone deliberately runs
    `generate_frozen_controls_snapshot.py` and commits the result, so a
    later semantic change to `generators.py`/`qubo_encoding.py`/
    `classical_exact.py` now actually produces a mismatch here. A caller
    that explicitly wants the old self-consistency check (e.g. "did this
    process's own computation behave deterministically") can still pass
    `against=FROZEN_CONTROLS` explicitly.

    `check_simulator_baseline=False` skips re-running Track B's QAOA
    optimizer (COBYLA is not bitwise-deterministic-across-scipy-versions
    in the same strict sense as the exact solvers, so this check is looser
    -- see the tolerance note in the implementation -- and meaningfully
    slower; callers that only care about the frozen INSTANCE itself, not
    its simulator baseline, can skip it)."""
    if against is None:
        against = load_committed_snapshot()
    results = []
    for spec in against:
        inst = generate_instance(spec.family, spec.n_candidates, seed=spec.seed)
        qubo = build_qubo(inst)
        exact = solve_exact_milp(inst)

        simulator_baseline_matches = True
        has_pinned_baseline = not math.isnan(spec.expected_simulator_expectation_qaoa_best_objective)
        if check_simulator_baseline and has_pinned_baseline:
            exp_metrics, cvar_metrics = _simulator_baseline(inst, qubo)
            exp_val = exp_metrics.best_feasible_objective
            cvar_val = cvar_metrics.best_feasible_objective
            # COBYLA is deterministic given a fixed seed/maxiter/scipy
            # version, so an exact re-run SHOULD match -- but this is a
            # looser tolerance than the exact-solver checks above, since a
            # future scipy upgrade could shift COBYLA's exact iterate path
            # without that being "the frozen instance drifted" in the
            # sense this module cares about most (fingerprint/exact
            # optimum). A mismatch here is still reported, not hidden, but
            # is a weaker signal than a fingerprint/exact-optimum mismatch.
            simulator_baseline_matches = (
                exp_val is not None
                and cvar_val is not None
                and abs(exp_val - spec.expected_simulator_expectation_qaoa_best_objective) <= tolerance
                and abs(cvar_val - spec.expected_simulator_cvar_qaoa_best_objective) <= tolerance
            )

        results.append(
            FrozenControlVerification(
                family=spec.family,
                n_candidates=spec.n_candidates,
                seed=spec.seed,
                fingerprint_matches=(inst.fingerprint() == spec.expected_fingerprint),
                n_qubits_matches=(qubo.n_qubits() == spec.expected_n_qubits),
                exact_optimum_matches=(abs(exact.objective - spec.expected_exact_optimum) <= tolerance),
                simulator_baseline_matches=simulator_baseline_matches,
            )
        )
    return results


__all__ = [
    "FrozenControlInstance",
    "FROZEN_CONTROL_SET",
    "FROZEN_CONTROLS",
    "FrozenControlVerification",
    "verify_frozen_controls",
    "load_committed_snapshot",
    "write_snapshot",
]
