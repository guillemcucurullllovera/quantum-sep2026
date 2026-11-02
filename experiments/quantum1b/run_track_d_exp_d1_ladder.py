"""Track D, EXP-D1 ladder extension -- characterizes FakeKingston
transpilation depth/gate-count inflation across a SMALL candidate ladder,
closing one of the disclosed gaps in
`docs/quantum/PRE_HARDWARE_QUANTUM_CAPSTONE.md` §4 item 6 ("EXP-D1's
depth/gate-count-inflation measurement is a single data point").

**CORRECTED 2026-10-01, remediating independent-review findings P1-2 and
P1-8** (`QUANTUM1B_INDEPENDENT_ADVERSARIAL_REVIEW_FINAL.md`). Two separate
errors in the original version of this script and its output, both found
and reproduced independently (by review passes, then by this session's own
follow-up) before being corrected here:

1. **P1-2 — the ">510s transpile wall at n_qubits=14, fixed by
   `optimization_level=0`" claim does not reproduce.** A committed,
   re-runnable calibration
   (`experiments/quantum1b/calibrate_transpile_optimization_level.py`) now
   backs this: `optimization_level=1` (the qiskit default) completes in
   ~3 seconds at this exact cell and produces a SHALLOWER circuit
   (depth≈375) than `optimization_level=0` (depth≈486). The original
   "sharp threshold... discovered empirically" framing is withdrawn.
   **What genuinely remains unresolved and still blocks this ladder's
   n_candidates=8 cell: a full 10-iteration COBYLA optimization against
   the FakeKingston-noisy backend at this depth did not complete within
   two bounded attempts (30 then 45 minutes)** — this is a
   NOISY-EXECUTION-time cost, not a transpile-time cost; see
   `run_track_d_exp_d1_n8.py` and `results/track_d_exp_d1_n8.json` for
   that attempt record, which carries its own separate caveat (P1-7,
   below).
2. **P1-8 — this script's first ladder row was mislabeled
   `n_candidates=4`.** The reused data
   (`results/track_d_exp_d1.json`) was actually generated from
   `generate_instance("independent", 3, seed=0)` (confirmed by
   fingerprint match) — `N_CANDIDATES_REUSED_FROM_D1 = 3` below, not 4, is
   now the correct, fingerprint-verified label. The depth/ratio numbers
   themselves were always correct (they came from the real measured run);
   only the generation-parameter label was wrong.

**Still does NOT re-run the original reused cell** (already completed and
recorded in `results/track_d_exp_d1.json` — re-running it would be exactly
the "redo expensive completed work unnecessarily" this session was told
not to do). This script runs n_candidates=6 fresh and records
n_candidates=8 as attempted-but-not-completed (for the noisy-execution-time
reason above, not a transpile reason).
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from experiments.quantum1b.generators import generate_instance  # noqa: E402
from experiments.quantum1b.qubo_encoding import build_qubo  # noqa: E402
from experiments.quantum1b.noise_validation import (  # noqa: E402
    fake_kingston_backend,
    ideal_backend,
    run_noise_validation_cell,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_d_exp_d1_ladder.json"

FAMILY = "independent"
SEED = 0
P = 1
MAXITER = 10  # reduced from EXP-D1's original 15 -- n=6 cells cost more per eval (larger transpiled depth)
SHOTS = 1024
COMPUTE_BUDGET_S = 900.0

N_CANDIDATES_REUSED_FROM_D1 = 3  # corrected from the original (wrong) "4" -- see module docstring, P1-8
N8_N_QUBITS = 14


def _result_to_dict(r) -> dict:
    d = asdict(r)
    d["best_params"] = list(map(float, r.best_params))
    return d


def main() -> dict:
    t0 = time.perf_counter()
    print(
        f"Running EXP-D1 ladder extension (n_candidates=6; n_candidates="
        f"{N_CANDIDATES_REUSED_FROM_D1} already done, n_candidates=8 not completed)..."
    )

    inst = generate_instance(FAMILY, 6, seed=SEED)
    qubo = build_qubo(inst)
    print(f"instance: family={FAMILY} n_candidates=6 n_qubits={qubo.n_qubits()}")

    print("running ideal (bare AerSimulator) cell, n_candidates=6...")
    ideal_result = run_noise_validation_cell(
        inst, qubo, backend_label="ideal", backend=ideal_backend(), p=P, maxiter=MAXITER, shots=SHOTS, seed=SEED
    )
    print(f"  ideal: depth={ideal_result.depth} 2q={ideal_result.two_qubit_gates} runtime={ideal_result.runtime_s:.1f}s")

    print("building FakeKingston-derived noisy simulator...")
    fk_backend = fake_kingston_backend()
    print("running fake_kingston cell, n_candidates=6...")
    fk_result = run_noise_validation_cell(
        inst, qubo, backend_label="fake_kingston", backend=fk_backend, p=P, maxiter=MAXITER, shots=SHOTS, seed=SEED
    )
    print(f"  fake_kingston: depth={fk_result.depth} 2q={fk_result.two_qubit_gates} runtime={fk_result.runtime_s:.1f}s")

    total_runtime = time.perf_counter() - t0

    # Reused verbatim from the already-completed run
    # (results/track_d_exp_d1.json) -- NOT re-executed. That run's actual
    # generation parameter, fingerprint-verified: n_candidates=3 (see
    # module docstring, P1-8 correction).
    with open(RESULTS_DIR / "track_d_exp_d1.json") as f:
        original = json.load(f)
    reused = original["EXP-D1"]
    assert reused["instance_n_candidates"] == N_CANDIDATES_REUSED_FROM_D1, (
        f"reused cell's recorded n_candidates ({reused['instance_n_candidates']}) does not match "
        f"N_CANDIDATES_REUSED_FROM_D1 ({N_CANDIDATES_REUSED_FROM_D1}) -- the P1-8 fix assumed this "
        "file's own recorded value; re-verify before trusting this ladder's first row."
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "correction_note": (
            "This manifest was regenerated 2026-10-01 remediating independent-review findings "
            "P1-2 (the original n_candidates=8 'transpile wall' claim did not reproduce -- see "
            "calibrate_transpile_optimization_level.py) and P1-8 (the first ladder row's "
            "n_candidates was mislabeled 4, corrected to the fingerprint-verified 3). The prior, "
            "superseded version of this file is preserved in git history, not deleted."
        ),
        "EXP-D1-ladder": {
            "family": FAMILY,
            "seed": SEED,
            "p": P,
            "shots": SHOTS,
            "ladder": [
                {
                    "n_candidates": N_CANDIDATES_REUSED_FROM_D1,
                    "n_qubits": reused["qubo_n_qubits"],
                    "source": "reused verbatim from results/track_d_exp_d1.json (not re-run)",
                    "ideal_depth": reused["ideal"]["depth"],
                    "ideal_two_qubit_gates": reused["ideal"]["two_qubit_gates"],
                    "fake_kingston_depth": reused["fake_kingston"]["depth"],
                    "fake_kingston_two_qubit_gates": reused["fake_kingston"]["two_qubit_gates"],
                    "depth_ratio_fake_over_ideal": reused["depth_ratio_fake_over_ideal"],
                    "status": "COMPLETED",
                },
                {
                    "n_candidates": 6,
                    "n_qubits": qubo.n_qubits(),
                    "maxiter": MAXITER,
                    "ideal_depth": ideal_result.depth,
                    "ideal_two_qubit_gates": ideal_result.two_qubit_gates,
                    "ideal_runtime_s": ideal_result.runtime_s,
                    "fake_kingston_depth": fk_result.depth,
                    "fake_kingston_two_qubit_gates": fk_result.two_qubit_gates,
                    "fake_kingston_runtime_s": fk_result.runtime_s,
                    "depth_ratio_fake_over_ideal": fk_result.depth / ideal_result.depth if ideal_result.depth else None,
                    "ideal_feasible_fraction": ideal_result.feasible_fraction,
                    "fake_kingston_feasible_fraction": fk_result.feasible_fraction,
                    "ideal_objective_gap": ideal_result.objective_gap,
                    "fake_kingston_objective_gap": fk_result.objective_gap,
                    "status": "COMPLETED",
                },
                {
                    "n_candidates": 8,
                    "n_qubits": N8_N_QUBITS,
                    "status": "NOT_COMPLETED",
                    "reason": (
                        "CORRECTED 2026-10-01 (P1-2): transpilation itself is NOT the bottleneck -- "
                        "optimization_level=1 (the qiskit default) transpiles this exact cell in "
                        "~3s, confirmed by a committed, re-runnable calibration "
                        "(calibrate_transpile_optimization_level.py). The real, still-unresolved "
                        "bottleneck is NOISY-EXECUTION time: a full 10-iteration COBYLA "
                        "optimization against the FakeKingston-noisy backend at this depth did not "
                        "complete within two bounded attempts (30 then 45 minutes) -- see "
                        "results/track_d_exp_d1_n8.json, which itself carries a P1-7 caveat about "
                        "its attempt-timestamp record's internal consistency. Not re-attempted "
                        "further here given this session's resource discipline."
                    ),
                    "ideal": None,
                    "fake_kingston": None,
                },
            ],
            "depth_inflation_finding": (
                "FakeKingston transpile-time depth inflation (fake_kingston depth / ideal depth) "
                "grows with n_qubits across the two genuinely completed points measured here: "
                f"n_qubits={reused['qubo_n_qubits']} -> {reused['depth_ratio_fake_over_ideal']:.1f}x, "
                f"n_qubits={qubo.n_qubits()} -> {(fk_result.depth / ideal_result.depth if ideal_result.depth else float('nan')):.1f}x. "
                "CORRECTED framing (P2-13): with only two single-seed, unreplicated points, this is "
                "directionally suggestive, not a statistically established trend -- the original "
                "'genuine trend, not noise' language overstated what two points can show. The "
                "separate, now-retracted 'sharp threshold at n_qubits=14' claim (P1-2) has been "
                "removed from this field entirely, not merely softened."
            ),
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"EXP-D1 ladder extension complete in {total_runtime:.1f}s. Wrote {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
