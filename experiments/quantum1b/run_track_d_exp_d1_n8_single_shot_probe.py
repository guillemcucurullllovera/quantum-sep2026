"""Track D runner: EXP-D1 n_candidates=8 (14 qubits) -- single-evaluation
wall-clock probe.

**Why this exists, and why it's safe where the prior two attempts were
not.** `run_track_d_exp_d1_n8.py` twice attempted a FULL 10-iteration
noisy-COBYLA optimization loop at this cell and was killed both times
(30min, then 45min bound) without completing -- see
`results/track_d_exp_d1_n8.json`. Its own `followup_diagnostic` SPECULATED
("almost certainly") that the elapsed time was consumed by repeated noisy
`backend.run()` calls inside COBYLA, without ever directly measuring a
single such call. Per the review's own explicit recommendation against
blindly re-chasing an already-twice-failed OPEN-ENDED job (a full
optimization loop with no fixed iteration-time bound), this script does
NOT re-attempt that loop. It instead measures exactly ONE noisy
`backend.run()` call's wall-clock cost -- a single, bounded operation that
cannot "hang" the way an iterative optimizer's convergence loop can -- to
replace speculation with a real data point, and to extrapolate (not
re-attempt) what a full 10-iteration loop would plausibly cost.

Run with: `python3 -m experiments.quantum1b.run_track_d_exp_d1_n8_single_shot_probe`
from the repo root. Expected cost: a single noisy execution at 14 qubits --
bounded in STRUCTURE (one `backend.run()` call, not an iterative loop), but
NOT bounded in TIME in practice.

**Actual outcome, 2026-10-01: this script's own `main()` did not complete
on this host either.** Two independently bounded CLI attempts (`timeout
280`, then `timeout 90` with unbuffered output) both terminated before the
single `backend.run()` call returned. The committed result
(`results/track_d_exp_d1_n8_single_shot_probe.json`) was assembled from
these two attempts' own logged output, NOT from a successful `main()`
return -- see that file's own `manifest_provenance` field for exactly what
was measured in which attempt. What WAS confirmed directly: backend build
(~11s) and transpilation (~2.6s, depth=376, matching the P1-2 calibration's
independently-measured depth for this exact cell almost exactly) are fast
and not the bottleneck. What was NOT measured: the single execution's exact
wall-clock cost, beyond a lower bound of >280s. This is itself a genuine,
useful finding -- it isolates the EXP-D1 n=8 bottleneck to noisy circuit
EXECUTION specifically (not transpilation, not backend construction),
replacing `run_track_d_exp_d1_n8.py`'s own prior speculation with a
bounded, direct (if incomplete) measurement -- without re-attempting the
open-ended full optimization loop that failed twice before, per the
independent review's own recommendation against blindly re-chasing an
already-failed job. `main()` itself is left unchanged (a real,
straightforwardly re-runnable measurement) for a future session with more
patience or a less-loaded host to complete.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_d_exp_d1_n8_single_shot_probe.json"

FAMILY = "independent"
N_CANDIDATES = 8
SEED = 0
GAMMA = [0.5]
BETA = [0.3]
SHOTS = 1024
OPTIMIZATION_LEVEL = 1
ASSUMED_COBYLA_ITERATIONS = 10  # matches run_track_d_exp_d1_n8.py's own maxiter


def main() -> dict:
    from qiskit import transpile

    from experiments.quantum1b.generators import generate_instance
    from experiments.quantum1b.hardware_readiness.lineage import build_qaoa_circuit_for_lineage
    from experiments.quantum1b.noise_validation import fake_kingston_backend
    from experiments.quantum1b.qubo_encoding import build_qubo

    print("Running EXP-D1 n_candidates=8 single-evaluation probe (safe, bounded -- not a COBYLA loop)...")

    inst = generate_instance(FAMILY, N_CANDIDATES, seed=SEED)
    qubo = build_qubo(inst)
    n_qubits = qubo.n_qubits()
    print(f"instance: family={FAMILY} n_candidates={N_CANDIDATES} n_qubits={n_qubits}")

    t0 = time.perf_counter()
    backend = fake_kingston_backend()
    backend_build_s = time.perf_counter() - t0
    print(f"backend built in {backend_build_s:.2f}s")

    qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=np.array(GAMMA), beta=np.array(BETA))
    qc.measure_all()

    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=OPTIMIZATION_LEVEL, seed_transpiler=SEED)
    transpile_s = time.perf_counter() - t0
    depth = int(tqc.depth())
    print(f"transpiled in {transpile_s:.2f}s, depth={depth}")

    t0 = time.perf_counter()
    job = backend.run(tqc, shots=SHOTS, seed_simulator=SEED)
    result = job.result()
    execution_s = time.perf_counter() - t0
    n_counts = sum(result.get_counts().values())
    print(f"single noisy execution: {execution_s:.2f}s ({n_counts} shots returned)")

    extrapolated_10_iter_s = execution_s * ASSUMED_COBYLA_ITERATIONS

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-D1-n8-single-shot-probe": {
            "instance": {
                "family": FAMILY, "n_candidates": N_CANDIDATES, "seed": SEED,
                "n_qubits": n_qubits, "fingerprint": inst.fingerprint(),
            },
            "backend": backend.name,
            "backend_build_s": backend_build_s,
            "transpile_s": transpile_s,
            "transpiled_depth": depth,
            "shots": SHOTS,
            "single_execution_s": execution_s,
            "assumed_cobyla_iterations": ASSUMED_COBYLA_ITERATIONS,
            "extrapolated_full_optimization_s": extrapolated_10_iter_s,
            "extrapolated_full_optimization_minutes": extrapolated_10_iter_s / 60.0,
            "conclusion": (
                f"A SINGLE noisy backend.run() call at this cell (n_qubits={n_qubits}, depth={depth}, "
                f"{SHOTS} shots) took {execution_s:.1f}s, measured directly (not estimated). Linearly "
                f"extrapolating to a {ASSUMED_COBYLA_ITERATIONS}-iteration COBYLA loop (the same maxiter "
                f"run_track_d_exp_d1_n8.py used) implies roughly {extrapolated_10_iter_s / 60.0:.1f} minutes "
                "for the full optimization -- this is an EXTRAPOLATION from one real measurement, not a "
                "rerun of the full loop (COBYLA's actual per-iteration cost can vary with the objective "
                "function's own call pattern, e.g. retries or line-search sub-calls, so this extrapolation "
                "should be treated as an order-of-magnitude estimate, not a precise prediction). This "
                "replaces run_track_d_exp_d1_n8.py's own prior SPECULATION ('almost certainly...consumed by "
                "the noisy backend.run() calls') with a real, direct, bounded measurement of exactly that "
                "quantity, without re-attempting the open-ended full loop that failed twice before."
            ),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nExtrapolated full optimization: ~{extrapolated_10_iter_s / 60.0:.1f} minutes")
    print(f"EXP-D1 n8 single-shot probe complete. Wrote {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
