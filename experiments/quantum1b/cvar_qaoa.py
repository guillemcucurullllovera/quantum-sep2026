"""NumPy statevector-exact CVaR-QAOA engine for `QUBOEncoding` -- Track B
of QUANTUM-1B (docs/quantum/QUANTUM1B_PLAN.md section 3,
docs/quantum/QUANTUM1B_PREREGISTRATION.md EXP-B1/B2/B3).

**Why a new engine instead of reusing `algorithms/qaoa_qubo/solvers.py`'s
`NumpyQAOASolver` (Gen2):** Gen2 uses a cardinality-preserving XY-mixer
with Dicke-state initialization, which assumes a FIXED Hamming weight
(`QAOAConfig.target_k`). This module's QUBO
(`experiments.quantum1b.qubo_encoding.build_qubo`) includes SLACK qubits
whose bits must range freely (they encode inequality remainders, not a
cardinality choice) -- a fixed-weight mixer would be structurally wrong
for them. This module instead implements the STANDARD RX-mixer QAOA
(`|+>^n -> {exp(-i*gamma*H_cost), RX(2*beta)}^p -> measure`), matching the
gate/sign convention of `algorithms/qaoa_qubo/solvers.py::
QiskitQAOASolver._build_qiskit_qaoa_circuit` and `_cost_diagonal_from_ising`
exactly (reimplemented here, not imported, so this module has no import
dependency on a private helper -- the convention and resulting numbers are
identical, see `tests/quantum1b/test_cvar_qaoa.py`'s brute-force parity
check).

**Two sign/objective conventions are in play in this module, precisely
documented so they are never confused:**

1. **QUBO-space training objective** (used inside `run_expectation_qaoa`/
   `run_cvar_qaoa`'s COBYLA loop): operates on the QUBO's own penalized
   cost `x^T Q x + offset` (candidate bits AND slack bits), which this
   codebase's convention (`qubo_builder.py`, `classical/optimization/
   brute_force.py::qubo_objective`) always MINIMIZES. CVaR-alpha here
   therefore averages the BEST (LOWEST-cost) alpha-fraction of the
   distribution's probability mass -- `cvar_from_distribution(...,
   direction="min")`. This is the Barkoutsos & Nannicini (2020) convention
   verbatim, and matches `docs/quantum/QUANTUM1B_PLAN.md` section 3's own
   locked wording ("optimizer target is the mean of the BEST alpha-fraction
   of the measured-cost distribution"). (Note: an earlier draft of this
   module's calling task description used the phrase "worst-cost
   alpha-fraction" -- that phrasing is NOT followed here; the Plan
   document's explicit "best alpha-fraction" wording and the cited paper
   both say best/lowest-cost, and this is what is implemented.)
2. **Instance-space reporting objective** (used by `exact_distribution_metrics`,
   and by `experiments.quantum1b.metrics.evaluate_sample_set` for the
   finite-shot path): operates on the TRUE constrained objective
   (`instance.evaluate_selection(...).objective`), which this codebase
   always MAXIMIZES (higher utility is better). CVaR-alpha here averages
   the BEST (HIGHEST-objective) alpha-fraction --
   `cvar_from_distribution(..., direction="max")`, matching
   `metrics.evaluate_sample_set`'s own docstring exactly.

These are genuinely different numbers (QUBO cost includes large penalty
terms for infeasible slack/candidate combinations; the true objective does
not), so they are never interchanged silently -- every function below
documents which convention it returns.

**Exactness, not sampling, for the training objective and for EXP-B1's
primary metrics.** Both `run_expectation_qaoa` and `run_cvar_qaoa` compute
their COBYLA objective from the EXACT `2^n_qubits`-dimensional probability
distribution (`|amplitude|^2`), never from finite shot sampling --
matching `docs/quantum/QUANTUM1B_PLAN.md`'s "statevector-exact" framing and
keeping the result at evidence level B (ideal simulator). Finite-shot
sampling (`sample_bitstrings`) exists only for EXP-B2's shot-sensitivity
experiment, which is explicitly about shot noise (evidence level C).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from experiments.quantum1b.instance import OptionSelectionInstance, evaluate_selection
from experiments.quantum1b.qubo_encoding import QUBOEncoding, selection_from_bits
from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising


# ---------------------------------------------------------------------------
# Cost diagonal (Ising) -- vectorized reimplementation of
# algorithms/qaoa_qubo/solvers.py::_cost_diagonal_from_ising's exact bit
# convention (bit b of an outcome index = (idx >> b) & 1 is variable/qubit
# b, z = 1 - 2*bit), verified against that function to machine precision in
# tests/quantum1b/test_cvar_qaoa.py. Vectorized because the original
# per-bitstring Python loop is O(2^n * n^2) in pure Python -- fine at n=10
# (Qiskit's own documented ceiling) but this module reaches n_qubits up to
# ~17-18 (candidates + slack bits, see qubo_encoding.py), where the loop
# form becomes the dominant cost; the numpy form is the same arithmetic,
# vectorized.
# ---------------------------------------------------------------------------


DENSE_BITS_QUBIT_THRESHOLD = 18
"""Above this many qubits, `cost_diagonal_from_ising` switches from the
fully-vectorized `(dim, n)` broadcast form to an O(dim) per-qubit
accumulation loop -- see that function's docstring for why. 18 matches the
qubit ceiling every pre-existing caller (EXP-B1/EXP-B2, and
`QiskitQAOASolver`'s own documented convention) already operates under, so
this threshold changes no previously-exercised code path."""


def cost_diagonal_from_ising(
    J: np.ndarray, h: np.ndarray, *, dense_threshold_qubits: int = DENSE_BITS_QUBIT_THRESHOLD
) -> np.ndarray:
    """Return the length-`2^n` diagonal Ising cost `h.z + sum_{i<j} J[i,j] z_i z_j`
    for every computational-basis outcome, `z_i = 1 - 2*bit_i`, `bit_i =
    (idx >> i) & 1`. Matches `_cost_diagonal_from_ising` in
    `algorithms/qaoa_qubo/solvers.py` exactly (same convention).

    **Two implementations, bit-identical output, chosen by `n` vs
    `dense_threshold_qubits`** (the keyword exists so a test can force
    either path at a shared small `n`; production callers never pass it):

    1. `n <= dense_threshold_qubits`: the original fully-vectorized
       `(dim, n)` broadcast form (fast for the sizes every caller before
       2026-10-01 used this at -- EXP-B1/EXP-B2 top out at n_qubits<=18).
    2. `n > dense_threshold_qubits`: an O(dim) (not O(dim*n)) per-qubit /
       per-pair accumulation loop that never materializes a `(dim, n)`
       array. **This branch exists because of a real production incident,
       not preemptive hardening:** EXP-B3's extended entanglement-diagnostic
       ladder calls this function at n_qubits up to 23-25, where the
       original `(dim, n)` broadcast form allocates multiple simultaneous
       `(dim, n)` float64/int64 temporaries (`bits`, `z`, `zJ`) -- at
       n_qubits=23 that is already ~1.5 GiB PER temporary (~4-6 GiB live
       at once), and at n_qubits=25 it extrapolates to 20+ GiB, which is
       exactly what killed the process via the host OOM killer on a shared,
       concurrently-loaded machine (see
       `q10_qaoa_reproduction.py`'s module docstring for the full incident
       record). This branch is the actual fix, not a workaround: it makes
       the SAME function safe at the sizes EXP-B3 was already preregistered
       to reach (`n_qubits<=27` per its stopping rule), independent of how
       far any particular run's candidate ladder is trimmed.
       Bit-identical to branch 1 -- verified in
       `tests/quantum1b/test_cvar_qaoa.py::TestCostDiagonalMemoryBoundedPath`
       by calling both branches at the same (small) n via this keyword and
       asserting exact agreement, and by the existing independent
       loop-reference test (`test_cost_diagonal_matches_loop_reference`),
       which both branches must satisfy.
    """
    n = len(h)
    dim = 1 << n
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)

    if n <= dense_threshold_qubits:
        idx = np.arange(dim, dtype=np.int64)
        bit_positions = np.arange(n, dtype=np.int64)
        bits = (idx[:, None] >> bit_positions[None, :]) & 1  # (dim, n)
        z = (1 - 2 * bits).astype(np.float64)  # (dim, n), values in {-1, +1}

        diag = z @ h
        if n > 1:
            Jsym = J + J.T  # symmetric, Jsym[i,j] = J[i,j] for i<j (diag 0)
            zJ = z @ Jsym
            diag += 0.5 * np.sum(zJ * z, axis=1)
        return diag

    # --- O(dim) memory path: never materializes a (dim, n) array. ---
    idx = np.arange(dim, dtype=np.int64)

    def z_col(bit: int) -> np.ndarray:
        return (1 - 2 * ((idx >> bit) & 1)).astype(np.float64)

    diag = np.zeros(dim, dtype=np.float64)
    for b in range(n):
        if h[b] != 0.0:
            diag += h[b] * z_col(b)

    if n > 1:
        Jsym = J + J.T  # symmetric, Jsym[i,j] = J[i,j] for i<j (diag 0)
        # diag += 0.5 * sum_{i,j} Jsym[i,j] * z_i * z_j, restricted to the
        # nonzero upper triangle (Jsym's diagonal is 0 by construction, so
        # this equals sum_{i<j} Jsym[i,j] * z_i * z_j exactly, matching the
        # dense branch's `0.5 * sum(zJ * z, axis=1)` algebraically).
        # Deliberately no cross-pair caching of z_col results: at most a
        # couple of (dim,) arrays are ever live at once (strict O(dim)
        # peak memory regardless of n or how dense J is), trading a modest,
        # bounded amount of recomputation (each z_col recomputed at most
        # once per pair that touches it, O(n^2) calls worst case) for that
        # guarantee -- exactly the tradeoff this branch exists to make.
        pair_i, pair_j = np.nonzero(np.triu(Jsym, k=1))
        for i, j in zip(pair_i.tolist(), pair_j.tolist()):
            diag += Jsym[i, j] * z_col(i) * z_col(j)
    return diag


# ---------------------------------------------------------------------------
# Statevector QAOA engine -- Hadamard init, diagonal cost-phase, per-qubit
# RX mixer applied via tensor reshaping (never a dense 2^n x 2^n matrix).
# ---------------------------------------------------------------------------


def _apply_cost_phase(state: np.ndarray, cost_diag: np.ndarray, gamma: float) -> np.ndarray:
    return state * np.exp(-1j * gamma * cost_diag)


def _apply_rx_all_qubits(state: np.ndarray, n: int, beta: float) -> np.ndarray:
    """Apply RX(2*beta) = exp(-i*beta*X) to every qubit independently, via
    tensor reshape + per-axis `tensordot` -- O(n * 2^n), never builds a
    dense `2^n x 2^n` matrix (essential at n_qubits ~ 17-18; a dense mixer
    matrix there would be `2^34` entries, intractable).

    Axis convention: reshaping a flat length-`2^n` array into shape
    `(2,) * n` in numpy's default (C) order places variable/qubit `b`
    (defined by `bit_b = (idx >> b) & 1`, matching `cost_diagonal_from_ising`)
    at tensor axis `n - 1 - b` (axis 0 is the most-significant bit of the
    flat index, axis n-1 is the least-significant bit). This mapping is
    used consistently everywhere a qubit index needs a tensor axis."""
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    gate = np.array([[c, s], [s, c]], dtype=np.complex128)

    tensor = state.reshape((2,) * n)
    for qubit in range(n):
        axis = n - 1 - qubit
        tensor = np.tensordot(gate, tensor, axes=([1], [axis]))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def simulate_qaoa(
    J: np.ndarray,
    h: np.ndarray,
    p: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    cost_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Exact, ideal (no shot noise) standard-mixer QAOA statevector
    simulation: `|+>^n -> {exp(-i*gamma_l*H_cost), RX(2*beta_l)}_{l=1..p} -> |psi>`.
    Returns the final complex statevector (length `2^n`, `n = len(h)`).
    `cost_diag` may be precomputed (it does not depend on gamma/beta) and
    passed in to avoid recomputation across repeated calls with the same
    `(J, h)` (e.g. inside an optimizer loop)."""
    n = len(h)
    dim = 1 << n
    if cost_diag is None:
        cost_diag = cost_diagonal_from_ising(J, h)

    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    for layer in range(p):
        state = _apply_cost_phase(state, cost_diag, gamma[layer])
        state = _apply_rx_all_qubits(state, n, beta[layer])
    return state


# ---------------------------------------------------------------------------
# CVaR on a discrete probability distribution (exact, Rockafellar-Uryasev
# boundary interpolation so the result is continuous in alpha and in the
# underlying probabilities -- important since this is a COBYLA target).
# ---------------------------------------------------------------------------


def cvar_from_distribution(
    probs: np.ndarray, values: np.ndarray, alpha: float, *, direction: str = "min"
) -> float:
    """Exact (no sampling) CVaR-alpha of `values` under the discrete
    distribution `probs` (need not be pre-normalized; renormalized here
    defensively against floating-point drift).

    `direction="min"`: average the BEST (LOWEST) alpha-fraction of
    probability mass -- the QUBO-cost-MINIMIZATION training convention
    (Barkoutsos & Nannicini 2020; see module docstring point 1).
    `direction="max"`: average the BEST (HIGHEST) alpha-fraction --
    the instance-objective-MAXIMIZATION reporting convention, matching
    `experiments.quantum1b.metrics.evaluate_sample_set` (module docstring
    point 2).

    At `alpha=1.0` this reduces to the plain probability-weighted
    expectation regardless of `direction` (the entire distribution is
    included either way) -- verified in
    tests/quantum1b/test_cvar_qaoa.py::test_cvar_alpha_one_equals_expectation.

    Uses linear interpolation at the probability-mass boundary (the exact
    Rockafellar-Uryasev CVaR definition for a discrete distribution) so the
    result varies continuously with `alpha` and with `probs` -- required
    for it to be a well-behaved COBYLA optimization target.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")

    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    total = p.sum()
    if total <= 0:
        raise ValueError("cvar_from_distribution: probabilities sum to zero")
    p = p / total
    v = np.asarray(values, dtype=np.float64)

    order = np.argsort(v) if direction == "min" else np.argsort(-v)
    sorted_v = v[order]
    sorted_p = p[order]
    cum = np.cumsum(sorted_p)

    cutoff = int(np.searchsorted(cum, alpha, side="left"))
    cutoff = min(cutoff, len(sorted_v) - 1)
    mass_before = float(cum[cutoff - 1]) if cutoff > 0 else 0.0
    weighted = float(np.sum(sorted_p[:cutoff] * sorted_v[:cutoff]))
    remaining_mass = alpha - mass_before
    weighted += remaining_mass * sorted_v[cutoff]
    return weighted / alpha


# ---------------------------------------------------------------------------
# Run result + optimizer drivers
# ---------------------------------------------------------------------------


@dataclass
class QAOARunResult:
    best_params: np.ndarray  # length 2*p: [gamma_1..gamma_p, beta_1..beta_p]
    p: int
    alpha: float  # 1.0 for expectation-QAOA
    seed: int
    n_qubits: int
    n_evaluations: int
    runtime_s: float
    optimizer_status: str
    probabilities: np.ndarray  # length 2^n_qubits, |amplitude|^2 at best_params, exact/ideal
    objective_mode: str  # "expectation" | "cvar"
    final_training_objective: float  # QUBO-space (minimized) objective value at best_params
    J: np.ndarray = field(repr=False)
    h: np.ndarray = field(repr=False)
    offset: float = 0.0

    def two_qubit_gate_count(self) -> int:
        """Number of nonzero off-diagonal Ising `J` entries times `p` --
        the preregistered EXP-B1 `two_qubit_gate_count` metric (one RZZ-type
        entangling gate per nonzero `J[i,j]` per QAOA layer, matching
        `_build_qiskit_qaoa_circuit`'s `qc.rzz(...)` gate count exactly)."""
        n_pairs = int(np.sum(np.abs(np.triu(self.J, k=1)) > 1e-12))
        return n_pairs * self.p

    def statevector(self) -> np.ndarray:
        """Recompute the exact complex statevector at `best_params` (not
        stored directly to keep `QAOARunResult` cheaply serializable as
        real-valued `probabilities`; recomputation is deterministic and
        cheap -- needed by EXP-B3's entanglement-entropy diagnostic, which
        requires amplitudes, not just probabilities)."""
        gamma, beta = self.best_params[: self.p], self.best_params[self.p :]
        cost_diag = cost_diagonal_from_ising(self.J, self.h)
        return simulate_qaoa(self.J, self.h, self.p, gamma, beta, cost_diag=cost_diag)


def _run_qaoa(
    qubo: QUBOEncoding, *, p: int, alpha: float, seed: int, maxiter: int, mode: str,
    initial_params: np.ndarray | None = None,
) -> QAOARunResult:
    t0 = time.perf_counter()
    J, h, offset = qubo_to_ising(qubo.Q)
    n = len(h)
    cost_diag_ising = cost_diagonal_from_ising(J, h)
    qubo_cost_diag = cost_diag_ising + offset  # actual x^T Q x + offset per outcome

    if initial_params is None:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, np.pi / 2, 2 * p)  # matches QiskitQAOASolver.solve's x0 convention
    else:
        # Caller-supplied warm-start (e.g. INTERP/linear-ramp, Track F) --
        # must be exactly (gamma_1..gamma_p, beta_1..beta_p), this module's
        # own convention, not silently reinterpreted.
        x0 = np.asarray(initial_params, dtype=float)
        if x0.shape != (2 * p,):
            raise ValueError(
                f"initial_params must have shape ({2 * p},) (gamma_1..gamma_p, beta_1..beta_p), "
                f"got shape {x0.shape}"
            )

    eval_count = [0]

    def objective(params: np.ndarray) -> float:
        eval_count[0] += 1
        gamma, beta = params[:p], params[p:]
        state = simulate_qaoa(J, h, p, gamma, beta, cost_diag=cost_diag_ising)
        probs = np.abs(state) ** 2
        if mode == "expectation":
            return float(np.sum(probs * qubo_cost_diag))
        return cvar_from_distribution(probs, qubo_cost_diag, alpha, direction="min")

    result = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
    best_params = np.asarray(result.x, dtype=float)
    gamma, beta = best_params[:p], best_params[p:]
    final_state = simulate_qaoa(J, h, p, gamma, beta, cost_diag=cost_diag_ising)
    final_probs = np.abs(final_state) ** 2
    final_probs = final_probs / final_probs.sum()

    runtime = time.perf_counter() - t0
    status = getattr(result, "message", None) or str(getattr(result, "status", "unknown"))

    return QAOARunResult(
        best_params=best_params,
        p=p,
        alpha=alpha,
        seed=seed,
        n_qubits=n,
        n_evaluations=eval_count[0],
        runtime_s=runtime,
        optimizer_status=str(status),
        probabilities=final_probs,
        objective_mode=mode,
        final_training_objective=float(result.fun),
        J=J,
        h=h,
        offset=offset,
    )


def run_expectation_qaoa(
    qubo: QUBOEncoding, p: int, seed: int, maxiter: int = 200,
    *, initial_params: np.ndarray | None = None,
) -> QAOARunResult:
    """Baseline QAOA: COBYLA minimizes the EXACT (statevector) expectation
    of the QUBO cost, `sum(prob_x * (x^T Q x + offset))` over all `2^n`
    outcomes. Equivalent to `run_cvar_qaoa(..., alpha=1.0, ...)` (verified
    in tests/quantum1b/test_cvar_qaoa.py); kept as a separate named
    function per the preregistered API (`alpha=1.0` is reported as
    "expectation-QAOA" throughout EXP-B1).

    `initial_params`, if given, overrides the default `seed`-derived random
    cold-start with a caller-supplied warm-start (shape `(2*p,)`,
    `gamma_1..gamma_p, beta_1..beta_p`) -- added for Track F's mixer/
    warm-start benchmarking (`experiments/quantum1b/mixer_warmstart_bench.py`),
    optional and backward-compatible (every pre-existing caller, including
    all of EXP-B1, omits it and gets byte-identical behavior to before)."""
    return _run_qaoa(qubo, p=p, alpha=1.0, seed=seed, maxiter=maxiter, mode="expectation", initial_params=initial_params)


def run_cvar_qaoa(
    qubo: QUBOEncoding, p: int, alpha: float, seed: int, maxiter: int = 200,
    *, initial_params: np.ndarray | None = None,
) -> QAOARunResult:
    """CVaR-QAOA (Barkoutsos & Nannicini 2020): COBYLA minimizes the EXACT
    (statevector) CVaR-alpha of the QUBO cost -- the mean QUBO cost over the
    BEST (lowest-cost) alpha-fraction of probability mass. See module
    docstring for the min/max convention discussion.

    `initial_params`: see `run_expectation_qaoa`'s docstring -- identical
    optional warm-start override, backward-compatible."""
    return _run_qaoa(qubo, p=p, alpha=alpha, seed=seed, maxiter=maxiter, mode="cvar", initial_params=initial_params)


# ---------------------------------------------------------------------------
# Sampling (finite-shot, EXP-B2 only) and exact post-hoc instance-space
# metrics (EXP-B1's primary path).
# ---------------------------------------------------------------------------


def sample_bitstrings(
    qaoa_result: QAOARunResult, n_samples: int, seed: int
) -> list[tuple[int, ...]]:
    """Draw `n_samples` QUBO-space bitstrings (candidates then slack, full
    `n_qubits` length) from the trained circuit's EXACT probability
    distribution, deterministic RNG. Used only by EXP-B2 (shot
    sensitivity) -- EXP-B1's primary metrics use the exact distribution
    directly via `exact_distribution_metrics`, not finite sampling, to
    avoid unnecessary sampling noise in a B-level (ideal simulator)
    result."""
    rng = np.random.default_rng(seed)
    dim = len(qaoa_result.probabilities)
    n = qaoa_result.n_qubits
    probs = qaoa_result.probabilities / qaoa_result.probabilities.sum()
    idxs = rng.choice(dim, size=n_samples, p=probs)
    out = []
    for idx in idxs:
        bits = tuple(int((int(idx) >> b) & 1) for b in range(n))
        out.append(bits)
    return out


@dataclass
class ExactDistributionMetrics:
    """Mirrors `experiments.quantum1b.metrics.SampleSetMetrics` field for
    field, but computed from the QAOA's EXACT probability distribution
    (marginalized over the internal slack qubits, which never appear in a
    decoded selection -- see `_marginal_candidate_distribution`) rather
    than from a finite equal-weighted sample list. This is the
    "mathematically identical logic applied to the same
    top-alpha-fraction-by-objective convention" alternative to calling
    `evaluate_sample_set` directly that the module brief anticipates:
    `evaluate_sample_set`'s signature assumes equal-weighted samples, and
    enumerating all `2^n_qubits` (up to ~2^17) outcomes individually
    through it would be both needlessly slow (most of that space is
    distinct-slack-bit copies of the same decoded selection) and would
    reintroduce sampling-style discreteness where an exact answer is
    available. Marginalizing over slack first collapses the enumeration to
    `2^len(candidate_ids)` (<= 1024 across this experiment's ladder)."""

    n_samples: int  # = 2^n_candidates distinct decoded outcomes enumerated (not 2^n_qubits)
    feasible_fraction: float
    best_objective: float
    best_feasible_objective: float | None
    expected_objective: float
    cvar_objective: float
    constraint_violation_rate: float
    n_distinct_top_decile: int
    mean_risk_budget_utilization: float


def exact_distribution_metrics(
    qaoa_result: QAOARunResult,
    qubo: QUBOEncoding,
    instance: OptionSelectionInstance,
    *,
    cvar_alpha: float = 1.0,
) -> ExactDistributionMetrics:
    """EXP-B1's primary post-hoc evaluator: bridges the QAOA's exact
    QUBO-space probability distribution back to instance-space via
    `selection_from_bits` + `evaluate_selection` (the TRUE constrained
    objective, never the QAOA's own claimed cost), weighted by exact
    (marginalized) probability rather than sampled counts."""
    n_candidates = len(qubo.candidate_ids)
    n_slack = qubo.n_slack_qubits
    probs = qaoa_result.probabilities
    if len(probs) != 1 << (n_candidates + n_slack):
        raise ValueError(
            "exact_distribution_metrics: qaoa_result.probabilities length "
            f"{len(probs)} does not match qubo n_qubits {qubo.n_qubits()}"
        )
    marginal = probs.reshape(2 ** n_slack, 2 ** n_candidates).sum(axis=0)
    marginal = marginal / marginal.sum()

    n_combos = 2 ** n_candidates
    objectives = np.empty(n_combos)
    feasible_mask = np.empty(n_combos, dtype=bool)
    rbu = np.empty(n_combos)
    n_violations = np.empty(n_combos)
    selections: list[tuple[int, ...]] = []

    for combo in range(n_combos):
        bits = [(combo >> b) & 1 for b in range(n_candidates)]
        sel = selection_from_bits(qubo, instance, bits)
        ev = evaluate_selection(instance, sel)
        objectives[combo] = ev.objective
        feasible_mask[combo] = ev.feasible
        rbu[combo] = ev.risk_budget_utilization
        n_violations[combo] = len(ev.violations)
        selections.append(tuple(sel))

    feasible_fraction = float(np.sum(marginal[feasible_mask]))
    best_objective = float(np.max(objectives))
    if feasible_mask.any():
        feas_obj = np.where(feasible_mask, objectives, -np.inf)
        best_feasible_objective = float(np.max(feas_obj))
    else:
        best_feasible_objective = None
    expected_objective = float(np.sum(marginal * objectives))

    cvar_objective = cvar_from_distribution(marginal, objectives, cvar_alpha, direction="max")

    constraint_violation_rate = float(np.sum(marginal * n_violations))
    mean_rbu = float(np.sum(marginal * rbu))

    order = np.argsort(-objectives)
    sorted_p = marginal[order]
    cum = np.cumsum(sorted_p)
    decile_cutoff = int(np.searchsorted(cum, 0.10, side="left"))
    decile_cutoff = min(decile_cutoff, n_combos - 1) + 1
    top_decile_indices = order[:decile_cutoff]
    n_distinct_top_decile = len({selections[i] for i in top_decile_indices})

    return ExactDistributionMetrics(
        n_samples=n_combos,
        feasible_fraction=feasible_fraction,
        best_objective=best_objective,
        best_feasible_objective=best_feasible_objective,
        expected_objective=expected_objective,
        cvar_objective=cvar_objective,
        constraint_violation_rate=constraint_violation_rate,
        n_distinct_top_decile=n_distinct_top_decile,
        mean_risk_budget_utilization=mean_rbu,
    )
