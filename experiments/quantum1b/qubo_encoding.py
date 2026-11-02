"""Penalty-method QUBO encoding of an OptionSelectionInstance.

Shared by Track A's classical comparator (for the MIQP->MILP
linearization of the quadratic risk term, and as the cross-check target
for its own exact solve) and Track B's CVaR-QAOA -- see
docs/quantum/QUANTUM1B_PLAN.md sections 1-3 for why this single function
is the fairness guarantee of the whole research program.

**Constraint-surface scoping, disclosed explicitly (not a silent
omission):** `build_qubo` encodes `risk_budget`, `max_positions`, and
`incompatible_with` pairs as hard constraints (exactly, via slack-bit
encoding -- see below), plus the quadratic co-selection risk term. It
does NOT encode `max_per_underlying`, `max_concentration`, or
`max_directional_exposure` -- adding those as further slack-bit
inequalities would grow the qubit count well past the ~16-qubit ceiling
the statevector-simulated QAOA engines can reach (see
`algorithms/qaoa_qubo/solvers.py::QiskitQAOASolver`'s own documented
n<=14 limit). Track A's classical comparator (`classical_exact.py`) does
NOT go through this function for its own exact solve -- it builds the
FULL constraint surface (every field on `OptionSelectionInstance`)
directly as real linear MILP constraints, which is both exact and free of
any qubit budget. This is a deliberate, disclosed asymmetry (Track A's
classical solver is structurally capable of more than Track B's simulated
QAOA is, at these qubit budgets) documented again in
`QUANTUM1B_CLASSICAL_BENCHMARK.md` and `QUANTUM1B_CVAR_QAOA.md` --
comparisons between the two tracks are only ever run on instances whose
`max_per_underlying`/`max_concentration`/`max_directional_exposure` are
left unset (`None`), so both tracks solve the literal same constraint set
on every instance actually used for an A-vs-B comparison; the omitted
fields remain available on `OptionSelectionInstance` for Track A's own
scaling stress tests (EXP-A2) where no quantum comparison is made.

`liquidity_floor` is handled as a hard pre-filter (disqualified
candidates dropped from the QUBO's variable set before encoding, recorded
in `QUBOEncoding.dropped_candidates`), not a penalty -- it needs no
qubits at all.

**Exact slack-bit encoding (Lucas 2014, "Ising formulations of many NP
problems," section on inequality constraints).** For `sum(a_i*x_i) <= B`
with non-negative integer-valued `a_i`/`B` (this module discretizes to
integer units of `resolution`, default 0.25, matching
`generators.py`'s candidate-value rounding so the discretization is exact
rather than approximate for every synthetic instance -- see
`_to_units`), introduce
binary slack bits `s_k` with weights `1, 2, 4, ..., 2^(m-1)`, plus one
final remainder bit, summing exactly to `B` -- every integer in `[0, B]`
is representable by some subset. The penalty `lam * h(x,s)^2` where
`h = sum(a_i*x_i) + sum(w_k*s_k) - B` is then EXACT, not approximate:
for any feasible `x` there exists an `s` making `h=0` (zero penalty,
verified in `tests/quantum1b/test_qubo_encoding.py`), and for any
infeasible `x` the minimum achievable `h^2` over `s` is strictly positive
and grows with the violation amount. This differs from the "unbalanced
penalization" family (Montanez-Barrera et al. 2022) which is slack-free
but leaks nonzero penalty into deeply-feasible points -- an earlier draft
of this module used that approach and a direct numerical check (all-zero,
i.e. NO_TRADE, picking up a large nonzero penalty) showed the leakage was
too large to trust without per-instance retuning; the slack-bit route is
more qubits but is exact and needs no per-instance calibration beyond
`resolution`.

Q follows the house convention (classical/optimization/brute_force.py::
qubo_objective, algorithms/qaoa_qubo/qubo_builder.py::qubo_to_ising):
minimize x^T Q x + offset, Q upper-triangular (diagonal = linear
coefficients, Q[i, j] for i<j = the total coefficient of x_i*x_j, lower
triangle left zero). Slack bits are appended as extra QUBO variables
after the candidate variables; `QUBOEncoding.candidate_ids` only names
the candidate-space variables (the first `len(candidate_ids)` rows/cols of
`Q`) -- slack variables are internal to the encoding and never appear in
a decoded selection (`selection_from_bits` slices them off).

Neither a QUBO solver nor QAOA's classical optimizer is trusted to
actually respect the encoded constraints -- the house convention already
established in `applications/portfolio/validation.py` ("the solver claims
feasibility is never trusted on its own") applies here too. Feasibility is
always re-checked post-hoc against the TRUE constraint set via
`instance.evaluate_selection`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.quantum1b.instance import OptionSelectionInstance


@dataclass
class QUBOEncoding:
    Q: np.ndarray
    offset: float
    candidate_ids: tuple[str, ...]  # order matching Q's first len(candidate_ids) rows/cols
    dropped_candidates: tuple[str, ...]
    n_slack_qubits: int
    penalty_lambda: float
    resolution: float
    encoded_constraints: tuple[str, ...] = field(default_factory=tuple)
    excluded_constraints: tuple[str, ...] = field(default_factory=tuple)

    def n_qubits(self) -> int:
        return len(self.candidate_ids) + self.n_slack_qubits


def _slack_bit_weights(capacity: int) -> list[int]:
    """Binary-expansion slack weights summing exactly to `capacity`
    (integer, >= 0): 1, 2, 4, ..., 2^(m-1), plus a final remainder bit, so
    every integer in [0, capacity] is representable by some subset."""
    if capacity <= 0:
        return []
    weights: list[int] = []
    total = 0
    w = 1
    while total + w <= capacity:
        weights.append(w)
        total += w
        w *= 2
    remainder = capacity - total
    if remainder > 0:
        weights.append(remainder)
    return weights


def _to_units(value: float, resolution: float) -> int:
    return int(round(value / resolution))


def _add_quadratic_inplace(Q: np.ndarray, i: int, j: int, coeff: float) -> None:
    if i == j:
        Q[i, i] += coeff
        return
    a, b = (i, j) if i < j else (j, i)
    Q[a, b] += coeff


def _slack_inequality_penalty(
    Q: np.ndarray,
    offset: float,
    var_coeffs: dict[int, int],
    capacity_units: int,
    slack_start: int,
    lam: float,
) -> tuple[float, int]:
    """Add lam * h(x,s)^2 to (Q, offset) in place, where
    h = sum(var_coeffs[i]*x_i) + sum(slack_weights[k]*s_k) - capacity_units
    (all integer units). Returns (updated offset, number of slack bits
    used). Q must already be large enough to hold the slack columns
    starting at `slack_start`."""
    weights = _slack_bit_weights(capacity_units)
    n_slack = len(weights)

    # terms: x-part (coeffs = var_coeffs), s-part (coeffs = weights at
    # indices slack_start..slack_start+n_slack-1), const = -capacity_units
    all_coeffs: dict[int, float] = dict(var_coeffs)
    for k, w in enumerate(weights):
        all_coeffs[slack_start + k] = float(w)
    const = -float(capacity_units)

    idxs = list(all_coeffs.keys())
    for i in idxs:
        gi = all_coeffs[i]
        _add_quadratic_inplace(Q, i, i, lam * (gi * gi + 2 * const * gi))
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            i, j = idxs[a], idxs[b]
            _add_quadratic_inplace(Q, i, j, lam * 2 * all_coeffs[i] * all_coeffs[j])
    offset += lam * const * const
    return offset, n_slack


def default_penalty_scale(instance: OptionSelectionInstance) -> float:
    """Penalty magnitude: proportional to the largest plausible objective
    swing (sum of |expected_utility|, in squared-unit terms since the
    slack encoding is exact and the penalty only needs to dominate, not
    be finely tuned) -- standard QUBO penalty-method guidance (Glover,
    Kochenberger & Du 2019, section 6)."""
    total_utility = sum(abs(c.expected_utility) for c in instance.candidates)
    return max(1.0, 4.0 * total_utility)


def build_qubo(
    instance: OptionSelectionInstance,
    *,
    penalty_lambda: float | None = None,
    resolution: float = 0.25,
) -> QUBOEncoding:
    lam = penalty_lambda if penalty_lambda is not None else default_penalty_scale(instance)

    kept = [
        c
        for c in instance.candidates
        if instance.liquidity_floor is None
        or c.liquidity_score is None
        or c.liquidity_score >= instance.liquidity_floor
    ]
    kept_ids = {c.candidate_id for c in kept}
    dropped = tuple(
        c.candidate_id for c in instance.candidates if c.candidate_id not in kept_ids
    )
    candidate_ids = tuple(c.candidate_id for c in kept)
    idx = {cid: i for i, cid in enumerate(candidate_ids)}
    n_candidates = len(kept)

    # Upper bound on total slack bits needed: risk_budget capacity units
    # (worst case) + max_positions capacity (small). Size Q generously
    # then trim unused trailing slack columns at the end.
    budget_units = max(_to_units(instance.risk_budget, resolution), 0)
    max_slack_estimate = len(_slack_bit_weights(budget_units))
    if instance.max_positions is not None:
        max_slack_estimate += len(_slack_bit_weights(int(instance.max_positions)))

    total_size = n_candidates + max_slack_estimate
    Q = np.zeros((total_size, total_size))
    offset = 0.0
    encoded: list[str] = []
    excluded: list[str] = []

    for c in kept:
        Q[idx[c.candidate_id], idx[c.candidate_id]] -= c.expected_utility

    if instance.risk_aversion:
        premium = {c.candidate_id: c.premium_cost for c in kept}
        for i_id, j_id, w in instance.pairwise_risk:
            if i_id not in idx or j_id not in idx or i_id == j_id:
                continue
            i, j = idx[i_id], idx[j_id]
            _add_quadratic_inplace(
                Q, i, j, instance.risk_aversion * w * premium[i_id] * premium[j_id]
            )

    next_slack = n_candidates

    # risk_budget: sum(premium_i * x_i) <= risk_budget (always encoded).
    var_coeffs = {idx[c.candidate_id]: _to_units(c.premium_cost, resolution) for c in kept}
    offset, used = _slack_inequality_penalty(Q, offset, var_coeffs, budget_units, next_slack, lam)
    next_slack += used
    encoded.append("risk_budget")

    if instance.max_positions is not None:
        var_coeffs = {idx[c.candidate_id]: 1 for c in kept}
        offset, used = _slack_inequality_penalty(
            Q, offset, var_coeffs, int(instance.max_positions), next_slack, lam
        )
        next_slack += used
        encoded.append("max_positions")

    seen_pairs: set[tuple[int, int]] = set()
    for c in kept:
        i = idx[c.candidate_id]
        for other_id in c.incompatible_with:
            j = idx.get(other_id)
            if j is None or j == i:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            # x_i + x_j <= 1: exact bilinear penalty, no slack needed
            # (the only violating combination is x_i=x_j=1).
            _add_quadratic_inplace(Q, pair[0], pair[1], lam)
    if seen_pairs:
        encoded.append("incompatible_with")

    for name, val in (
        ("max_per_underlying", instance.max_per_underlying),
        ("max_concentration", instance.max_concentration),
        ("max_directional_exposure", instance.max_directional_exposure),
    ):
        if val is not None:
            excluded.append(name)

    Q = Q[:next_slack, :next_slack]

    return QUBOEncoding(
        Q=Q,
        offset=offset,
        candidate_ids=candidate_ids,
        dropped_candidates=dropped,
        n_slack_qubits=next_slack - n_candidates,
        penalty_lambda=lam,
        resolution=resolution,
        encoded_constraints=tuple(encoded),
        excluded_constraints=tuple(excluded),
    )


def selection_from_bits(
    encoding: QUBOEncoding, instance: OptionSelectionInstance, bits
) -> list[int]:
    """Expand a QUBO-space bitstring (candidate bits followed by slack
    bits) back to a full instance-space selection vector over
    `instance.candidates` (dropped/liquidity-filtered candidates forced to
    0, slack bits discarded) -- the shape `instance.evaluate_selection`
    expects."""
    bits = list(bits)
    candidate_bits = bits[: len(encoding.candidate_ids)]
    by_id = dict(zip(encoding.candidate_ids, candidate_bits))
    return [by_id.get(c.candidate_id, 0) for c in instance.candidates]
