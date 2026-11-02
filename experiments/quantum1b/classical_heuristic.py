"""Simulated-annealing fallback for `OptionSelectionInstance`.

Documented fallback for Track A's classical comparator when
`classical_exact.solve_exact_milp`'s Fortet linearization (`O(n^2)`
auxiliary `y_ij` variables for dense pairwise-risk instances) becomes
impractical within EXP-A2's 120s-per-solve compute budget -- see
docs/quantum/QUANTUM1B_PREREGISTRATION.md EXP-A2's stopping rule
("on timeout, ... fall back to simulated_annealing_qubo ..., explicitly
labeled heuristic (not exact) in all downstream reporting").

Unlike `classical_exact`, this module DOES go through
`qubo_encoding.build_qubo` (the penalty-method encoding, same one Track B's
CVaR-QAOA uses) -- `simulated_annealing_qubo` only knows how to minimize
`x^T Q x`, not the instance's native linear/quadratic-constrained form. The
QUBO's own claimed objective is never trusted: every sampled bitstring is
decoded back to instance-space via `selection_from_bits` and re-scored
through the real `instance.evaluate_selection`, exactly like every other
solver in this package.
"""
from __future__ import annotations

import time

from classical.optimization.simulated_annealing import simulated_annealing_qubo
from experiments.quantum1b.classical_exact import ClassicalSolveResult
from experiments.quantum1b.instance import OptionSelectionInstance, evaluate_selection
from experiments.quantum1b.qubo_encoding import build_qubo, selection_from_bits


def solve_simulated_annealing(
    instance: OptionSelectionInstance,
    *,
    n_iters: int = 20000,
    n_restarts: int = 5,
    seed: int = 0,
) -> ClassicalSolveResult:
    """Run `simulated_annealing_qubo` `n_restarts` times (seeds
    `seed, seed+1, ..., seed+n_restarts-1`), decode and re-score each
    sample against the TRUE constrained objective, and return the best
    FEASIBLE one found. If no restart produced a feasible selection, the
    best (infeasible) one found is returned instead with `feasible=False`
    and that fact stated plainly in `solver_info` -- never silently
    promoted to look like a valid answer."""
    t0 = time.perf_counter()
    encoding = build_qubo(instance)

    best_feasible = None
    best_overall = None
    per_restart: list[dict] = []

    for r in range(n_restarts):
        run_seed = seed + r
        bits, qubo_obj = simulated_annealing_qubo(encoding.Q, n_iters=n_iters, seed=run_seed)
        decoded = selection_from_bits(encoding, instance, bits)
        ev = evaluate_selection(instance, decoded)
        per_restart.append(
            {
                "seed": run_seed,
                "qubo_minimized_value": float(qubo_obj) + encoding.offset,
                "true_objective": ev.objective,
                "feasible": ev.feasible,
            }
        )
        if ev.feasible and (best_feasible is None or ev.objective > best_feasible.objective):
            best_feasible = ev
        if best_overall is None or ev.objective > best_overall.objective:
            best_overall = ev

    runtime = time.perf_counter() - t0

    if best_feasible is not None:
        chosen = best_feasible
        note = (
            f"best FEASIBLE selection across {n_restarts} restarts "
            f"(seeds {seed}..{seed + n_restarts - 1})"
        )
    else:
        chosen = best_overall
        note = (
            f"NO feasible selection found in any of {n_restarts} restarts "
            f"(seeds {seed}..{seed + n_restarts - 1}) -- returning the best "
            "objective found across restarts honestly labeled feasible=False, "
            "not silently accepted as a valid answer."
        )

    return ClassicalSolveResult(
        selection=chosen.selection,
        objective=chosen.objective,
        feasible=chosen.feasible,
        method="simulated_annealing",
        solver_info={
            "n_restarts": n_restarts,
            "n_iters": n_iters,
            "seeds": list(range(seed, seed + n_restarts)),
            "qubit_count": encoding.n_qubits(),
            "penalty_lambda": encoding.penalty_lambda,
            "encoded_constraints": list(encoding.encoded_constraints),
            "excluded_constraints": list(encoding.excluded_constraints),
            "note": note,
            "per_restart": per_restart,
        },
        runtime_s=runtime,
    )
