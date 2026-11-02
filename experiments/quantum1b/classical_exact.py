"""Exact classical solvers for `OptionSelectionInstance` -- Track A's
primary comparator (docs/quantum/QUANTUM1B_PLAN.md section 2).

Two independent exact methods, cross-validated against each other
(EXP-A1, see `docs/quantum/QUANTUM1B_PREREGISTRATION.md`):

- `solve_exact_enumeration`: exhaustive 2^n search over the TRUE
  constrained objective (`instance.evaluate_selection`), ground truth for
  n small enough to enumerate.
- `solve_exact_milp`: the FULL constraint surface of `OptionSelectionInstance`
  (risk_budget, max_positions, max_per_underlying, max_concentration,
  max_directional_exposure, incompatible_with, liquidity_floor) encoded as
  real linear `scipy.optimize.milp` (HiGHS) constraints -- not penalties --
  plus an exact Fortet/Glover linearization of the quadratic
  `risk_aversion * pairwise_risk` term. This is the "structural fairness
  advantage" the plan documents explicitly: Track A's classical solver can
  use real hard constraints; Track B's QAOA must use penalties for the same
  constraints via `qubo_encoding.build_qubo` (which additionally drops
  max_per_underlying/max_concentration/max_directional_exposure for qubit-
  budget reasons -- this module does NOT go through that QUBO at all for
  its own exact solve, precisely to keep the exact classical comparator
  free of that qubit ceiling).

Both solvers never trust their own internal bookkeeping for the returned
`objective`/`feasible` fields -- the final selection is always re-scored
through `experiments.quantum1b.instance.evaluate_selection`, the single
ground-truth evaluator shared by every track (house convention, see
`qubo_encoding.py`'s module docstring: "the solver claims feasibility is
never trusted on its own").

**Performance note on `solve_exact_enumeration`:** a naive Python loop
calling `evaluate_selection` once per one of the 2^n bitstrings is too slow
to stay within EXP-A1's preregistered <5 minute compute budget at n=18
(measured ~8s/instance naive vs ~1.2s/instance vectorized, and EXP-A1 runs
45 n=18 instances). This module instead vectorizes the identical
constraint/objective arithmetic with numpy across all 2^n selections at
once to find the argmax, then re-scores ONLY that winning bitstring through
the real `evaluate_selection` -- so the number actually reported is always
literally the canonical evaluator's output, never a hand-rolled
reimplementation's own number. The vectorized arithmetic itself is
cross-checked against a literal per-bitstring `evaluate_selection` loop in
`tests/quantum1b/test_classical_exact.py` across every family.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

from experiments.quantum1b.instance import OptionSelectionInstance, evaluate_selection


@dataclass
class ClassicalSolveResult:
    selection: tuple[int, ...]  # full instance-space selection, length == instance.n()
    objective: float
    feasible: bool
    method: str  # "enumeration" | "milp_highs" | "simulated_annealing"
    solver_info: dict = field(default_factory=dict)
    runtime_s: float = 0.0


class MILPTimeoutError(RuntimeError):
    """Raised by `solve_exact_milp` when HiGHS reports a time-limit status
    (or, as a defensive fallback, when our own wall-clock guard trips for a
    scipy build that silently ignores the `time_limit` option) -- never
    hangs, never silently returns a possibly-suboptimal result mislabeled
    as exact. Callers (see EXP-A2's stopping rule) catch this and fall back
    to `classical_heuristic.solve_simulated_annealing`."""

    def __init__(self, message: str, *, runtime_s: float, solver_info: dict):
        super().__init__(message)
        self.runtime_s = runtime_s
        self.solver_info = solver_info


# ---------------------------------------------------------------------------
# Exhaustive enumeration (ground truth)
# ---------------------------------------------------------------------------


def _vectorized_enumeration_argmax(instance: OptionSelectionInstance) -> tuple[int, ...]:
    """Vectorized numpy search for the feasible-objective-maximizing 0/1
    selection, mirroring `instance.evaluate_selection`'s exact constraint
    math (same tolerances). Returns only the winning bitstring; the caller
    re-scores it through the real `evaluate_selection` for the authoritative
    objective/feasibility."""
    n = instance.n()
    cands = instance.candidates
    N = 1 << n

    idx_arr = np.arange(N, dtype=np.uint64)
    shifts = np.arange(n, dtype=np.uint64)
    bits = ((idx_arr[:, None] >> shifts) & 1).astype(np.int64)  # (N, n)

    premiums = np.array([c.premium_cost for c in cands], dtype=float)
    utilities = np.array([c.expected_utility for c in cands], dtype=float)

    total_premium = bits @ premiums
    n_positions = bits.sum(axis=1)

    feasible = np.ones(N, dtype=bool)
    feasible &= total_premium <= instance.risk_budget + 1e-9

    if instance.max_positions is not None:
        feasible &= n_positions <= instance.max_positions

    if instance.max_per_underlying is not None:
        for underlying, cap in instance.max_per_underlying.items():
            mask = np.array([1.0 if c.underlying == underlying else 0.0 for c in cands])
            feasible &= (bits @ mask) <= cap

    if instance.max_concentration is not None and instance.risk_budget > 0:
        underlyings_present = sorted({c.underlying for c in cands})
        for underlying in underlyings_present:
            mask = np.array(
                [c.premium_cost if c.underlying == underlying else 0.0 for c in cands]
            )
            frac = (bits @ mask) / instance.risk_budget
            feasible &= frac <= instance.max_concentration + 1e-9

    if instance.max_directional_exposure is not None:
        signs = np.array([c.direction_sign() * c.premium_cost for c in cands])
        directional = bits @ signs
        feasible &= np.abs(directional) <= instance.max_directional_exposure + 1e-9

    if instance.liquidity_floor is not None:
        bad = np.array(
            [
                1.0
                if (c.liquidity_score is not None and c.liquidity_score < instance.liquidity_floor)
                else 0.0
                for c in cands
            ]
        )
        feasible &= (bits @ bad) == 0

    idx_map = instance.candidate_index()
    for c in cands:
        i = idx_map[c.candidate_id]
        for other_id in c.incompatible_with:
            j = idx_map.get(other_id)
            if j is None or j == i:
                continue
            violated = (bits[:, i] == 1) & (bits[:, j] == 1)
            feasible &= ~violated

    linear_utility = bits @ utilities
    quadratic_risk = np.zeros(N)
    if instance.pairwise_risk:
        premium_by_id = {c.candidate_id: c.premium_cost for c in cands}
        for i_id, j_id, w in instance.pairwise_risk:
            i = idx_map.get(i_id)
            j = idx_map.get(j_id)
            if i is None or j is None or i == j:
                continue
            quadratic_risk += w * premium_by_id[i_id] * premium_by_id[j_id] * (bits[:, i] * bits[:, j])
    quadratic_risk *= instance.risk_aversion

    objective = linear_utility - quadratic_risk
    masked_objective = np.where(feasible, objective, -np.inf)
    best_idx = int(np.argmax(masked_objective))
    return tuple(int(b) for b in bits[best_idx])


def solve_exact_enumeration(
    instance: OptionSelectionInstance, max_vars: int = 20
) -> ClassicalSolveResult:
    """Exhaustive 2^n search over the TRUE constrained objective (ground
    truth). Raises `ValueError` above `max_vars` rather than hang."""
    n = instance.n()
    if n > max_vars:
        raise ValueError(
            f"solve_exact_enumeration is exact 2^n enumeration only; n={n} "
            f"exceeds max_vars={max_vars}. Use solve_exact_milp (exact, no "
            "enumeration ceiling) or classical_heuristic.solve_simulated_annealing "
            "(heuristic fallback) instead."
        )

    t0 = time.perf_counter()
    winner = _vectorized_enumeration_argmax(instance)
    ev = evaluate_selection(instance, winner)
    runtime = time.perf_counter() - t0

    assert ev.feasible, (
        "solve_exact_enumeration found an infeasible argmax -- NO_TRADE "
        "(all-zero) is always feasible per instance.py's contract, so this "
        "indicates a bug in the vectorized search, not a genuinely "
        "infeasible instance."
    )

    return ClassicalSolveResult(
        selection=ev.selection,
        objective=ev.objective,
        feasible=ev.feasible,
        method="enumeration",
        solver_info={"n_variables": n, "n_enumerated": 1 << n},
        runtime_s=runtime,
    )


# ---------------------------------------------------------------------------
# Exact MIQP via scipy.optimize.milp (HiGHS) + Fortet/Glover linearization
# ---------------------------------------------------------------------------


def _liquidity_filter(instance: OptionSelectionInstance):
    """Same convention as `qubo_encoding.build_qubo`: candidates below the
    liquidity floor are dropped from the variable set entirely (a hard
    pre-filter, not a penalized/constrained variable)."""
    kept = [
        c
        for c in instance.candidates
        if instance.liquidity_floor is None
        or c.liquidity_score is None
        or c.liquidity_score >= instance.liquidity_floor
    ]
    return kept


def solve_exact_milp(
    instance: OptionSelectionInstance, time_limit_s: float = 120.0
) -> ClassicalSolveResult:
    """Exact MIQP-via-MILP solve of the FULL `OptionSelectionInstance`
    constraint surface using `scipy.optimize.milp` (HiGHS backend).

    The quadratic `risk_aversion * pairwise_risk` term is linearized
    exactly via Fortet/Glover auxiliary variables `y_ij` per
    `(i_id, j_id, w)` triple in `instance.pairwise_risk` (skipping any pair
    referencing a liquidity-filtered candidate or a self-pair, matching
    `qubo_encoding.build_qubo`'s own skip convention):

        y_ij <= x_i
        y_ij <= x_j
        y_ij >= x_i + x_j - 1
        0 <= y_ij <= 1

    These four inequalities pin `y_ij` to exactly `x_i * x_j` for any
    binary `x_i, x_j` regardless of the sign of the objective coefficient
    on `y_ij` (verified by exhaustive case check over the 4 combinations of
    `(x_i, x_j)`), so `y_ij` is declared *continuous* in `[0, 1]`
    (`integrality=0`) -- branching is needed only on the `x_i` binaries,
    which keeps the MIP tree small even as the number of pairwise-risk
    terms grows with `O(n^2)` (the EXP-A2 scaling stress case).

    Raises `MILPTimeoutError` (never hangs, never silently downgrades) if
    HiGHS reports a time-limit status, or if the scipy build in use doesn't
    support the `options={"time_limit": ...}` key (detected and guarded,
    not assumed).
    """
    t0 = time.perf_counter()
    n = instance.n()
    kept = _liquidity_filter(instance)
    dropped_ids = {c.candidate_id for c in instance.candidates} - {c.candidate_id for c in kept}
    idx = {c.candidate_id: i for i, c in enumerate(kept)}
    n_x = len(kept)

    quad_terms: list[tuple[int, int, float, float, float]] = []
    if instance.risk_aversion:
        premium = {c.candidate_id: c.premium_cost for c in kept}
        for i_id, j_id, w in instance.pairwise_risk:
            if i_id not in idx or j_id not in idx or i_id == j_id:
                continue
            quad_terms.append((idx[i_id], idx[j_id], w, premium[i_id], premium[j_id]))
    n_y = len(quad_terms)
    n_vars = n_x + n_y

    orig_index = instance.candidate_index()

    if n_x == 0:
        # Every candidate liquidity-filtered out -- NO_TRADE is the only
        # feasible selection, trivially optimal (objective 0.0).
        runtime = time.perf_counter() - t0
        selection = tuple([0] * n)
        ev = evaluate_selection(instance, selection)
        return ClassicalSolveResult(
            selection=ev.selection,
            objective=ev.objective,
            feasible=ev.feasible,
            method="milp_highs",
            solver_info={
                "scipy_version": scipy.__version__,
                "solver": "scipy.optimize.milp (HiGHS)",
                "status": "trivial_all_candidates_liquidity_filtered",
                "n_variables": 0,
                "n_x": 0,
                "n_y_auxiliary": 0,
                "dropped_candidates": sorted(dropped_ids),
                "time_limit_s": time_limit_s,
            },
            runtime_s=runtime,
        )

    # Objective: minimize -linear_utility + risk_aversion * w * p_i * p_j * y_ij
    # (equivalent to maximizing linear_utility - risk_aversion * quadratic_risk).
    c = np.zeros(n_vars)
    for i, cand in enumerate(kept):
        c[i] = -cand.expected_utility
    for k, (i, j, w, pi, pj) in enumerate(quad_terms):
        c[n_x + k] = instance.risk_aversion * w * pi * pj

    integrality = np.zeros(n_vars)
    integrality[:n_x] = 1  # x_i binary; y_ij continuous (pinned exactly by constraints)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

    constraints: list[LinearConstraint] = []

    def _row(coeffs: dict[int, float]) -> np.ndarray:
        r = np.zeros((1, n_vars))
        for pos, val in coeffs.items():
            r[0, pos] = val
        return r

    # risk_budget: sum(premium_i * x_i) <= risk_budget
    row = np.zeros((1, n_vars))
    for i, cand in enumerate(kept):
        row[0, i] = cand.premium_cost
    constraints.append(LinearConstraint(row, -np.inf, instance.risk_budget))

    # max_positions: sum(x_i) <= max_positions
    if instance.max_positions is not None:
        row = np.zeros((1, n_vars))
        row[0, :n_x] = 1.0
        constraints.append(LinearConstraint(row, -np.inf, float(instance.max_positions)))

    # max_per_underlying: per-underlying cardinality caps
    if instance.max_per_underlying is not None:
        for underlying, cap in instance.max_per_underlying.items():
            row = np.zeros((1, n_vars))
            any_present = False
            for i, cand in enumerate(kept):
                if cand.underlying == underlying:
                    row[0, i] = 1.0
                    any_present = True
            if any_present:
                constraints.append(LinearConstraint(row, -np.inf, float(cap)))

    # max_concentration: per-underlying premium spend <= max_concentration * risk_budget
    if instance.max_concentration is not None and instance.risk_budget > 0:
        underlyings_present = sorted({c.underlying for c in kept})
        bound = instance.max_concentration * instance.risk_budget
        for underlying in underlyings_present:
            row = np.zeros((1, n_vars))
            for i, cand in enumerate(kept):
                if cand.underlying == underlying:
                    row[0, i] = cand.premium_cost
            constraints.append(LinearConstraint(row, -np.inf, bound))

    # max_directional_exposure: -D <= sum(sign_i * premium_i * x_i) <= D
    if instance.max_directional_exposure is not None:
        row = np.zeros((1, n_vars))
        for i, cand in enumerate(kept):
            row[0, i] = cand.direction_sign() * cand.premium_cost
        D = instance.max_directional_exposure
        constraints.append(LinearConstraint(row, -D, D))

    # incompatible_with: x_i + x_j <= 1 for each unique unordered pair
    seen_pairs: set[tuple[int, int]] = set()
    for cand in kept:
        i = idx[cand.candidate_id]
        for other_id in cand.incompatible_with:
            j = idx.get(other_id)
            if j is None or j == i:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            row = _row({pair[0]: 1.0, pair[1]: 1.0})
            constraints.append(LinearConstraint(row, -np.inf, 1.0))

    # Fortet/Glover linearization: y_ij == x_i * x_j exactly for binary x_i, x_j.
    for k, (i, j, w, pi, pj) in enumerate(quad_terms):
        yk = n_x + k
        constraints.append(LinearConstraint(_row({yk: 1.0, i: -1.0}), -np.inf, 0.0))  # y<=x_i
        constraints.append(LinearConstraint(_row({yk: 1.0, j: -1.0}), -np.inf, 0.0))  # y<=x_j
        constraints.append(
            LinearConstraint(_row({i: 1.0, j: 1.0, yk: -1.0}), -np.inf, 1.0)
        )  # y>=x_i+x_j-1

    options = {"time_limit": float(time_limit_s), "mip_rel_gap": 0.0}
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)
    runtime = time.perf_counter() - t0

    solver_info = {
        "scipy_version": scipy.__version__,
        "solver": "scipy.optimize.milp (HiGHS)",
        "status": int(result.status),
        "message": result.message,
        "success": bool(result.success),
        "n_variables": n_vars,
        "n_x": n_x,
        "n_y_auxiliary": n_y,
        "dropped_candidates": sorted(dropped_ids),
        "time_limit_s": time_limit_s,
        "mip_gap": getattr(result, "mip_gap", None),
        "mip_node_count": getattr(result, "mip_node_count", None),
    }

    if result.status == 1 or (result.status != 0 and runtime >= time_limit_s * 0.95):
        raise MILPTimeoutError(
            f"solve_exact_milp: HiGHS hit the {time_limit_s}s time limit "
            f"(status={result.status}: {result.message!r}) on instance "
            f"{instance.instance_id!r} (n_vars={n_vars}).",
            runtime_s=runtime,
            solver_info=solver_info,
        )
    if result.status == 2:
        raise RuntimeError(
            f"solve_exact_milp: HiGHS reports INFEASIBLE for instance "
            f"{instance.instance_id!r}, but NO_TRADE (all-zero) is always "
            "feasible by construction -- this indicates a bug in the MILP "
            f"formulation, not a genuinely infeasible instance. status={result.status}, "
            f"message={result.message!r}"
        )
    if result.status not in (0,):
        raise RuntimeError(
            f"solve_exact_milp: unexpected HiGHS status={result.status} "
            f"({result.message!r}) for instance {instance.instance_id!r}."
        )

    selection = [0] * n
    for k, cand in enumerate(kept):
        selection[orig_index[cand.candidate_id]] = int(round(result.x[k]))
    selection_t = tuple(selection)

    ev = evaluate_selection(instance, selection_t)

    return ClassicalSolveResult(
        selection=ev.selection,
        objective=ev.objective,
        feasible=ev.feasible,
        method="milp_highs",
        solver_info=solver_info,
        runtime_s=runtime,
    )
