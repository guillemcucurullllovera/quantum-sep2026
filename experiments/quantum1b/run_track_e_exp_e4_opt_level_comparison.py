"""Track E runner: EXP-E4 follow-up -- does `optimization_level=1` transpile
IQAE's Grover-amplified ansatz better than the `optimization_level=0`
`run_track_e_exp_e4.py` measured?

**Why this exists.** `run_track_e_exp_e4.py`'s own docstring (corrected
2026-10-01, P1-2) explicitly states: "Whether `optimization_level=1` would
do even better here was not tested." This closes that stated next step.

**Why this is a standalone script, not an edit to
`algorithms.qae_iqae.iqae._grover_metadata`.** `_grover_metadata` is
shared, pre-existing, already-in-production infrastructure
(`run_barrier_demo` depends on it) living OUTSIDE
`experiments/quantum1b/`/`tests/quantum1b/`/`docs/quantum/` -- this
session's own established boundary (see
`docs/quantum/QUANTUM1B_ESTATE_INVENTORY.md`'s "Do NOT touch" list) is to
not modify files outside that scope without a specific, justified
integration point. Adding an `optimization_level` parameter there was
considered and rejected in favor of this small, self-contained
reimplementation of just the transpile-and-measure step (~15 lines),
mirroring the SAME pattern `calibrate_transpile_optimization_level.py`
already used for Track D's equivalent question -- a committed,
independently re-runnable measurement, not a change to shared code whose
other callers (and their own tests) this session has not audited.

**Resource discipline:** reuses `run_track_e_exp_e4.py`'s own calibrated
scope (`ks=(0,1,2,4,8,16,32,64,128)`, measured at ~5-15s per full sweep at
`optimization_level=0`) -- this script runs the IDENTICAL sweep twice (once
per optimization_level), so total cost is a small constant multiple of
`run_track_e_exp_e4.py`'s own already-measured, well-bounded cost.

Run with: `python3 -m experiments.quantum1b.run_track_e_exp_e4_opt_level_comparison`
from the repo root.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from algorithms.qae_iqae.barrier_circuits import build_A_barrier  # noqa: E402
from algorithms.qae_iqae.finance_integrands import build_crash_barrier  # noqa: E402
from algorithms.qae_iqae.iqae import _decompose  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_e_exp_e4_opt_level_comparison.json"

KS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
COMPUTE_BUDGET_S = 300.0


def _grover_metadata_at_opt_level(A, anc: int, fake, ks, optimization_level: int) -> dict:
    """Identical construction to `algorithms.qae_iqae.iqae._grover_metadata`
    (same `_decompose`, same `EstimationProblem`/`grover_operator` source of
    the Grover operator, same `seed_transpiler=7` convention) with
    `optimization_level` exposed as a parameter -- see module docstring for
    why this is a standalone copy, not an edit to the shared original."""
    from qiskit import transpile
    from qiskit_algorithms import EstimationProblem

    Ad = _decompose(A)
    problem = EstimationProblem(state_preparation=Ad, objective_qubits=[anc])
    Q = problem.grover_operator
    out = {}
    for k in ks:
        circ = Ad.copy()
        for _ in range(k):
            circ.compose(Q, inplace=True)
        circ.measure_all()
        t = transpile(circ, backend=fake, optimization_level=optimization_level, seed_transpiler=7)
        gc = {kk: int(vv) for kk, vv in t.count_ops().items()}
        twoq = sum(v for g, v in gc.items() if g in ("cx", "cz", "ecr", "rzz"))
        out[f"Q{k}A"] = {"depth": int(t.depth()), "twoq_gates": twoq, "total_ops": sum(gc.values())}
    return out


def _linear_fit(ks_list, depths):
    n = len(ks_list)
    mean_k = sum(ks_list) / n
    mean_d = sum(depths) / n
    slope = sum((k - mean_k) * (d - mean_d) for k, d in zip(ks_list, depths)) / sum(
        (k - mean_k) ** 2 for k in ks_list
    )
    intercept = mean_d - slope * mean_k
    return slope, intercept


def main() -> dict:
    t0 = time.perf_counter()
    print("Running EXP-E4 optimization_level comparison (0 vs 1), FakeKingston...")

    from qiskit_ibm_runtime.fake_provider import FakeKingston

    barrier = build_crash_barrier(n=3)
    A = build_A_barrier(barrier.probs, barrier.barrier_mask)
    anc = int(round(math.log2(len(barrier.probs))))
    fake = FakeKingston()

    by_opt_level = {}
    for opt_level in (0, 1):
        meta = _grover_metadata_at_opt_level(A, anc, fake, KS, optimization_level=opt_level)
        depths = [meta[f"Q{k}A"]["depth"] for k in KS]
        slope, intercept = _linear_fit(list(KS), depths)
        by_opt_level[str(opt_level)] = {
            "circuits": meta,
            "depth_vs_k_linear_fit": {"slope_depth_per_grover_op": slope, "intercept_depth": intercept},
        }
        for k, row in meta.items():
            print(f"  opt_level={opt_level} {k}: depth={row['depth']} twoq_gates={row['twoq_gates']}")

    depth_0 = by_opt_level["0"]["circuits"][f"Q{KS[-1]}A"]["depth"]
    depth_1 = by_opt_level["1"]["circuits"][f"Q{KS[-1]}A"]["depth"]
    slope_0 = by_opt_level["0"]["depth_vs_k_linear_fit"]["slope_depth_per_grover_op"]
    slope_1 = by_opt_level["1"]["depth_vs_k_linear_fit"]["slope_depth_per_grover_op"]

    total_runtime = time.perf_counter() - t0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-E4-opt-level-comparison": {
            "integrand_label": barrier.label,
            "backend": fake.name,
            "ks": list(KS),
            "by_optimization_level": by_opt_level,
            "comparison_at_largest_k": {
                "k": KS[-1],
                "depth_opt_level_0": depth_0,
                "depth_opt_level_1": depth_1,
                "opt_level_1_is_shallower_or_equal": depth_1 <= depth_0,
            },
            "slope_comparison": {
                "slope_depth_per_grover_op_opt_level_0": slope_0,
                "slope_depth_per_grover_op_opt_level_1": slope_1,
            },
            "conclusion": (
                f"optimization_level=1 produces a {'SHALLOWER or equal' if depth_1 <= depth_0 else 'DEEPER'} "
                f"circuit than optimization_level=0 at k={KS[-1]} for this IQAE Grover-amplified ansatz "
                f"({depth_1} vs {depth_0}). This closes run_track_e_exp_e4.py's own stated open question "
                "('whether optimization_level=1 would do even better here was not tested'). Consistent with "
                "the general pattern this session's P1-2 calibration found for Track B's QAOA circuit "
                "(optimization_level=1, the qiskit default, is not worse and is often better) -- but this is "
                "its own independent measurement on a structurally different circuit (Grover-amplified "
                "amplitude estimation, not QAOA), not a re-derivation of that Track B/D finding."
            ),
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAt k={KS[-1]}: opt_level=0 depth={depth_0}, opt_level=1 depth={depth_1}")
    print(f"EXP-E4 opt-level comparison complete in {total_runtime:.1f}s. Wrote {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
