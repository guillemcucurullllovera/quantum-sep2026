"""EXP-B3 -- Q10-style classical-simulability (CSIM) reproduction, extended
to CVaR-QAOA (docs/quantum/QUANTUM1B_PLAN.md section 4,
docs/quantum/QUANTUM1B_PREREGISTRATION.md EXP-B3,
experiments/quantum1b/registry/EXP-B3.json).

The existing Q10 audit (`docs/research/Q10_AUDIT__classical-simulability.md`)
classifies PQK-v2, QRC-10, and VQC-v5 as CSIM (classically simulable) by two
independent routes: (1) **provenance** -- every benchmark number in the
audit was produced by exact numpy statevector/state-space execution, no QPU,
so CSIM already holds definitionally, before any structural argument; and
(2) **structure** -- e.g. VQC-v5's exact tensor-network surrogate shows
every bipartition's Schmidt rank caps at chi<=16 for ALL parameters tested,
an O(n) classical twin.

This module reproduces exactly that two-route logic for CVaR-QAOA, which
the existing audit never covered:

1. **CSIM by provenance is immediate and unconditional at every n tested
   here** -- `experiments.quantum1b.cvar_qaoa.simulate_qaoa` IS a numpy
   statevector simulation; there is no separate "quantum execution" whose
   provenance could be checked -- the circuit's own reference implementation
   is the classical computation, identically to the Q10 audit's QRC-10/
   VQC-v5 provenance finding. This is reported as a structural/definitional
   fact, never reframed as if it were a new discovery.
2. **The structural/entanglement question is separate and genuinely
   measured, not assumed:** does CVaR-QAOA's bipartite entanglement stay
   bounded (a VQC-v5-style bounded-chi finding, which WOULD additionally
   support an exact tensor-network surrogate) or grow toward the
   volume-law maximum (which would not preclude CSIM-by-provenance at any
   n reachable by exact statevector simulation, but WOULD mean no cheaper
   bounded-chi surrogate is available either, once n exceeds the exact
   simulator's own memory ceiling)? `bipartite_entanglement_entropy`
   measures this directly at the middle bipartition, across n and p, on
   the SAME dense-family instance ladder EXP-A2 already uses as its own
   worst-case pairwise-risk-graph stress test (family="dense" -- entangling
   coupling density is maximal for this family, the honest worst case for
   "does entanglement stay bounded").
3. **The honest crossover point** -- the smallest total qubit count at
   which exact statevector construction alone exceeds the stopping rule's
   4 GiB memory cap -- is reported analytically
   (`theoretical_memory_crossover_n_qubits`), together with how far the
   ladder actually tested here (n_qubits up to ~25, see CANDIDATE_LADDER
   below) falls short of it. This is the honest answer to "what regime
   would make a future hardware experiment interesting": CSIM-by-provenance
   holds at every n tested in THIS phase by construction (this module never
   runs anything else), so no n reachable here is evidence of quantum
   advantage regardless of the measured entanglement curve.

**Compute-budget / host-resource discipline (EXP-B3's own preregistered
stopping rule, and this task's explicit instruction to respect a shared
host running other concurrent sessions):** the candidate-count ladder below
is capped at 14 (23 total qubits for the "dense" family, ~134 MB per
statevector -- measured directly with `build_qubo` before choosing this
ceiling) rather than pushed toward the theoretical n_qubits<=27 ceiling the
stopping rule technically permits. `n_qubits<=27` would require multi-GiB
statevectors on a machine concurrently running other work; the registered
stopping rule explicitly permits reporting the theoretical crossover point
"even if not reached empirically," which is exactly what is done here. This
is a disclosed, resource-driven scope choice, not a result-driven one --
entanglement growth above n_qubits=23 is extrapolated from the measured
trend, not fabricated.

**Second resource-driven reduction, logged 2026-10-01 (overnight recovery
pass), lowering the ceiling further from candidate-count 16 (25 qubits,
~537 MB/statevector) to candidate-count 14 (23 qubits, ~134 MB/statevector):**
an earlier attempt at the original 16-candidate ceiling was killed by the
host's OOM killer while this shared machine was concurrently running two
other heavy agent sessions (MODEL-2A, MACRO-4A) plus several unrelated
`claude` sessions (host `nproc`=16, load average observed at ~15.2 -- i.e.
already CPU-saturated independent of this job). A direct, isolated
reproduction of the single most expensive step at n_qubits=25 (the
bipartite-entanglement SVD of a reshaped (4096, 8192) complex128 matrix,
~537 MB) was run afterward under a hard `ulimit -v` cap and did NOT reproduce
a memory blow-up in isolation (RSS held ~1.1 GB, just slow under the host's
existing CPU contention) -- so the exact mechanism of the original OOM is
not fully pinned down (most likely candidate: accumulated thread-local BLAS
workspace / glibc malloc-arena fragmentation across the full 14-cell loop's
repeated large allocations under heavy concurrent multi-tenant load, not a
single pathological allocation), but it is not worth re-risking a shared
host to isolate further. The 16-candidate/25-qubit cell is dropped rather
than chased, `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
`NUMEXPR_NUM_THREADS` are pinned to a small fixed value for this run (see
`run_track_b_exp_b3.py`) to bound BLAS thread-local memory overhead, and the
actually-tested ladder tops out at 14 candidates / 23 qubits / ~134 MB --
comfortably below the memory scale implicated in the OOM by roughly 4x,
while still covering both the COBYLA-optimized regime (n_candidates<=10) and
the beyond-ceiling representative-fixed-angle regime (12, 14) the diagnostic
needs to show a trend. This does not change the CSIM-by-provenance
classification (definitional at every n, per point 1 above) and does not
change the analytically-computed theoretical crossover point (point 3,
`theoretical_memory_crossover_n_qubits`, independent of how far the ladder
is actually run). See `docs/quantum/QUANTUM1B_CVAR_QAOA.md` for the full
incident record.

Angle sourcing, exactly as EXP-B3.json's benchmark_problem field specifies:
for candidate counts already in EXP-B1's optimized ladder (n<=10), this
module RE-RUNS `cvar_qaoa.run_cvar_qaoa` at EXP-B1's own representative
(p, alpha, seed) convention (cheap and deterministic, the same
re-derivation pattern `run_track_b_exp_b2.py` already uses) rather than
trying to deserialize a 2^n-dimensional probability array out of
`track_b_exp_b1.json`. For candidate counts beyond EXP-B1's optimized
ladder (n=12, 14, 16), no fresh COBYLA optimization is run (per the
registry's own "no fresh optimization needed above n=10" -- optimizing a
23-25 qubit CVaR-QAOA circuit is exactly the kind of open-ended compute
this phase is not budgeted for); a single fixed, non-degenerate
representative angle pair is used instead, clearly labeled as such in every
output record, since this diagnostic measures the CIRCUIT'S entangling
structure (a property of the cost Hamiltonian and mixer, not of how well an
optimizer happened to tune it), not optimized solution quality.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising
from core.evidence import compute_evidence_level
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from experiments.quantum1b.cvar_qaoa import run_cvar_qaoa, simulate_qaoa
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.qubo_encoding import build_qubo
from experiments.quantum1b.registry_schema import load_registry

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_b_exp_b3.json"

BYTES_PER_AMPLITUDE = 16  # complex128
MEMORY_CAP_BYTES = 4 * (1024 ** 3)  # 4 GiB, EXP-B3's registered stopping-rule cap

FAMILY = "dense"  # SAME worst-case family EXP-A2 already uses for its own scaling stress test
CANDIDATE_LADDER = (4, 6, 8, 10, 12, 14)  # 16 dropped 2026-10-01: see module docstring OOM note
P_VALUES = (1, 2)
OPTIMIZED_MAX_N_CANDIDATES = 10  # matches EXP-B1's own optimized-ladder ceiling
REPRESENTATIVE_ALPHA = 0.5  # CVaR-QAOA; a representative choice for the STRUCTURAL diagnostic, not a re-derivation of EXP-B1's own per-cell winning alpha
REPRESENTATIVE_SEED = 0
REPRESENTATIVE_GAMMA = np.pi / 8  # fixed, non-degenerate, used ONLY when optimization is skipped (n_candidates > OPTIMIZED_MAX_N_CANDIDATES)
REPRESENTATIVE_BETA = np.pi / 8
SCHMIDT_RANK_REL_TOL = 1e-8
ENTROPY_PROB_TOL = 1e-12

_BACKEND = BackendDescriptor(
    name="numpy_statevector_exact", kind=BackendKind.SIMULATOR, num_qubits=25
)


def statevector_memory_bytes(n_qubits: int) -> int:
    """Exact memory (bytes) to hold ONE complex128 statevector of `n_qubits`
    qubits: `2**n_qubits * 16`."""
    return (1 << n_qubits) * BYTES_PER_AMPLITUDE


def theoretical_memory_crossover_n_qubits(cap_bytes: float = MEMORY_CAP_BYTES) -> int:
    """Smallest `n_qubits` at which `statevector_memory_bytes(n_qubits)`
    exceeds `cap_bytes` -- the honest, disclosed answer to "how far is the
    tested ladder from where exact classical simulation becomes
    infeasible," per EXP-B3's stopping rule."""
    n = 0
    while statevector_memory_bytes(n) <= cap_bytes:
        n += 1
    return n


@dataclass
class EntanglementDiagnostic:
    n_qubits: int
    cut: int
    schmidt_rank: int
    max_possible_schmidt_rank: int
    entanglement_entropy_bits: float
    max_possible_entropy_bits: float
    entropy_fraction_of_max: float


def bipartite_entanglement_entropy(
    state: np.ndarray, n_qubits: int, cut: int, *, rel_tol: float = SCHMIDT_RANK_REL_TOL
) -> EntanglementDiagnostic:
    """Exact bipartite entanglement entropy and Schmidt rank of `state`
    (length `2**n_qubits`, assumed normalized) across the bipartition that
    splits qubits `[0, cut)` from `[cut, n_qubits)`, via SVD of the
    state reshaped to a `(2**cut, 2**(n_qubits-cut))` matrix -- the
    standard Schmidt-decomposition route (singular values squared = Schmidt
    coefficients' probabilities; von Neumann entropy of that probability
    vector, base 2, in bits)."""
    dim_a = 1 << cut
    dim_b = 1 << (n_qubits - cut)
    matrix = state.reshape(dim_a, dim_b)
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    probs = singular_values ** 2
    total = probs.sum()
    probs = probs / total if total > 0 else probs  # defensive renormalization vs fp drift

    nz = probs[probs > ENTROPY_PROB_TOL]
    entropy = float(-np.sum(nz * np.log2(nz))) if len(nz) else 0.0

    max_sv = float(singular_values.max()) if singular_values.size else 0.0
    schmidt_rank = int(np.sum(singular_values > rel_tol * max_sv)) if max_sv > 0 else 0

    max_possible_rank = min(dim_a, dim_b)
    max_possible_entropy = float(np.log2(max_possible_rank)) if max_possible_rank > 1 else 0.0

    return EntanglementDiagnostic(
        n_qubits=n_qubits,
        cut=cut,
        schmidt_rank=schmidt_rank,
        max_possible_schmidt_rank=max_possible_rank,
        entanglement_entropy_bits=entropy,
        max_possible_entropy_bits=max_possible_entropy,
        entropy_fraction_of_max=(entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0,
    )


def _build_state(n_candidates: int, p: int) -> tuple[np.ndarray, int, str]:
    inst = generate_instance(FAMILY, n_candidates, seed=0)
    qubo = build_qubo(inst)
    n_qubits = qubo.n_qubits()

    if n_candidates <= OPTIMIZED_MAX_N_CANDIDATES:
        qaoa_result = run_cvar_qaoa(
            qubo, p=p, alpha=REPRESENTATIVE_ALPHA, seed=REPRESENTATIVE_SEED, maxiter=200
        )
        state = qaoa_result.statevector()
        angle_source = (
            f"cobyla_optimized (EXP-B1 convention: alpha={REPRESENTATIVE_ALPHA}, "
            f"seed={REPRESENTATIVE_SEED}, maxiter=200)"
        )
    else:
        J, h, _offset = qubo_to_ising(qubo.Q)
        gamma = np.full(p, REPRESENTATIVE_GAMMA)
        beta = np.full(p, REPRESENTATIVE_BETA)
        state = simulate_qaoa(J, h, p, gamma, beta)
        angle_source = (
            f"representative_fixed_angles (gamma={REPRESENTATIVE_GAMMA:.4f}, "
            f"beta={REPRESENTATIVE_BETA:.4f} per layer, NO optimization run -- "
            f"n_candidates={n_candidates} exceeds EXP-B1's optimized ladder "
            f"ceiling of {OPTIMIZED_MAX_N_CANDIDATES})"
        )
    return state, n_qubits, angle_source


def run_exp_b3(registry) -> dict:
    exp = registry["EXP-B3"]
    t0 = time.perf_counter()

    crossover_n_qubits = theoretical_memory_crossover_n_qubits()

    cells = []
    for n_candidates in CANDIDATE_LADDER:
        for p in P_VALUES:
            state, n_qubits, angle_source = _build_state(n_candidates, p)
            cut = n_qubits // 2
            diag = bipartite_entanglement_entropy(state, n_qubits, cut)
            cells.append(
                {
                    "family": FAMILY,
                    "n_candidates": n_candidates,
                    "p": p,
                    "n_qubits": n_qubits,
                    "angle_source": angle_source,
                    "statevector_memory_bytes": statevector_memory_bytes(n_qubits),
                    "cut": diag.cut,
                    "schmidt_rank": diag.schmidt_rank,
                    "max_possible_schmidt_rank": diag.max_possible_schmidt_rank,
                    "entanglement_entropy_bits": diag.entanglement_entropy_bits,
                    "max_possible_entropy_bits": diag.max_possible_entropy_bits,
                    "entropy_fraction_of_max": diag.entropy_fraction_of_max,
                }
            )

    total_runtime = time.perf_counter() - t0

    largest_n_qubits_tested = max(c["n_qubits"] for c in cells)
    largest_memory_tested_bytes = max(c["statevector_memory_bytes"] for c in cells)
    # Guarded against an empty list (fixed 2026-10-01, remediating
    # independent-review finding P3-11): `ExperimentResult.metrics` requires
    # a plain float (pydantic rejects `None`), so NaN remains the correct
    # "undefined" sentinel here -- but in normal/real use (both p=1 and p=2
    # in P_VALUES) this list is never empty, while a caller that restricts
    # P_VALUES to a single value (e.g. a fast smoke test) would otherwise
    # hit `np.mean([])`, which produces the SAME NaN but via an unguarded,
    # unasserted "Mean of empty slice" RuntimeWarning. Short-circuiting to
    # an explicit `float("nan")` keeps the value identical while making the
    # "undefined" case a deliberate branch instead of a warning-producing
    # accident.
    _p1_fractions = [c["entropy_fraction_of_max"] for c in cells if c["p"] == 1]
    _p2_fractions = [c["entropy_fraction_of_max"] for c in cells if c["p"] == 2]
    mean_entropy_fraction_p1 = float(np.mean(_p1_fractions)) if _p1_fractions else float("nan")
    mean_entropy_fraction_p2 = float(np.mean(_p2_fractions)) if _p2_fractions else float("nan")
    # A "bounded-chi surrogate" signature (VQC-v5-style) would show schmidt_rank
    # saturating at a constant well below max_possible_schmidt_rank as n_qubits
    # grows; this is measured, not assumed -- see docs/quantum/QUANTUM1B_CVAR_QAOA.md
    # for the interpretation of whatever this ladder actually shows.
    rank_saturation_check = [
        {
            "n_qubits": c["n_qubits"],
            "p": c["p"],
            "schmidt_rank": c["schmidt_rank"],
            "max_possible_schmidt_rank": c["max_possible_schmidt_rank"],
        }
        for c in cells
    ]

    result = ExperimentResult(
        metadata=RunMetadata(algorithm="cvar_qaoa_csim_reproduction", seed=REPRESENTATIVE_SEED, generation=None),
        config=ExperimentConfig(
            name="EXP-B3",
            algorithm="cvar_qaoa_csim_reproduction",
            application="quantum1b_track_b",
            backend=_BACKEND,
            shots=None,
            seed=REPRESENTATIVE_SEED,
        ),
        metrics={
            "largest_n_qubits_tested": float(largest_n_qubits_tested),
            "theoretical_memory_crossover_n_qubits": float(crossover_n_qubits),
            "mean_entropy_fraction_of_max_p1": mean_entropy_fraction_p1,
            "mean_entropy_fraction_of_max_p2": mean_entropy_fraction_p2,
            "total_runtime_s": total_runtime,
        },
        classical_baseline_metrics=None,
        financial_edge_claimed=False,
        notes=(
            "EXP-B3: CSIM by provenance at every n tested (exact numpy statevector "
            "execution is the circuit's own reference implementation). No claim of "
            "quantum advantage at any n in this ladder, per the registered prohibited "
            "conclusion."
        ),
    )
    computed_evidence_level = compute_evidence_level(result).value

    manifest = {
        "experiment_id": "EXP-B3",
        "hypothesis": exp.hypothesis,
        "family": FAMILY,
        "candidate_ladder": list(CANDIDATE_LADDER),
        "p_values": list(P_VALUES),
        "optimized_max_n_candidates": OPTIMIZED_MAX_N_CANDIDATES,
        "memory_cap_bytes": MEMORY_CAP_BYTES,
        "theoretical_memory_crossover_n_qubits": crossover_n_qubits,
        "largest_n_qubits_tested": largest_n_qubits_tested,
        "largest_memory_tested_bytes": largest_memory_tested_bytes,
        "csim_classification": "CSIM_BY_PROVENANCE_AT_EVERY_N_TESTED",
        "prohibited_conclusion": exp.prohibited_conclusion,
        "cells": cells,
        "rank_saturation_check": rank_saturation_check,
        "total_runtime_s": total_runtime,
        "compute_budget_s": 1200.0,
        "within_compute_budget": total_runtime < 1200.0,
        "preregistered_evidence_level": exp.evidence_level,
        "computed_evidence_level_via_core_evidence": computed_evidence_level,
        "experiment_result": json.loads(result.model_dump_json()),
    }
    return manifest


def main() -> dict:
    registry = load_registry()
    print("Running EXP-B3 (Q10-style CSIM reproduction for CVaR-QAOA)...")
    b3 = run_exp_b3(registry)
    print(
        f"EXP-B3 complete: largest_n_qubits_tested={b3['largest_n_qubits_tested']}, "
        f"theoretical_crossover_n_qubits={b3['theoretical_memory_crossover_n_qubits']}, "
        f"runtime={b3['total_runtime_s']:.1f}s"
    )

    full_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "EXP-B3": b3,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"Wrote results manifest to {RESULTS_PATH}")

    return full_manifest


if __name__ == "__main__":
    main()
