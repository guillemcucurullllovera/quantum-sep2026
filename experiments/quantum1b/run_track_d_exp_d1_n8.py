"""Track D, EXP-D1 ladder, n_candidates=8 (14 qubits).

**CORRECTED 2026-10-01, remediating independent-review findings P1-2 and
P1-7.** The original version of this docstring claimed
`run_track_d_exp_d1_ladder.py` recorded this cell as `RESOURCE_BLOCKED`
because `optimization_level=1` transpile "did not complete within 8.5
minutes," and that `optimization_level=0` was "the fix." **That transpile-
wall claim does not reproduce** — a committed, re-runnable calibration
(`experiments/quantum1b/calibrate_transpile_optimization_level.py`) shows
`optimization_level=1` (the qiskit default) transpiles this exact cell in
~3s and produces a SHALLOWER circuit than `optimization_level=0`.
**The real, still-unresolved bottleneck this script's two recorded
attempts actually ran into is noisy-execution time** (the FakeKingston-
noisy `backend.run()` calls inside COBYLA), not transpilation — both
attempts below ran for a substantial wall-clock time without completing,
for reasons unrelated to `optimization_level`. Separately, the independent
review (finding P1-7) found this script's own recorded attempt timestamps
are internally inconsistent (attempt 2 appears to start only ~11 minutes
after attempt 1 was reportedly killed by a 30-minute limit) — see
`results/track_d_exp_d1_n8.json`'s own added `review_correction_note`
field for the full, disclosed caveat. **Neither attempt below was
re-run as part of this correction** — per the review's own
recommendation, chasing an unresolved multi-tens-of-minutes cost further
was not repeated; the honest state remains "attempted twice, not
completed, true cost unmeasured."

**Does NOT re-run n_candidates=3 or 6** (already completed, recorded in
`results/track_d_exp_d1.json` and `results/track_d_exp_d1_ladder.json`
respectively). This script only runs n_candidates=8.

Resource discipline: BLAS threads pinned, run as the only heavy job,
`free -h`/`uptime` checked immediately before launch, generous but firm
`ulimit -v` safety net, and a script-level compute budget with an honest
`within_compute_budget` check (not a silent truncation).
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
RESULTS_PATH = RESULTS_DIR / "track_d_exp_d1_n8.json"

FAMILY = "independent"
N_CANDIDATES = 8
SEED = 0
P = 1
MAXITER = 10  # same convention as the n_candidates=6 cell
SHOTS = 1024
FAKE_KINGSTON_OPT_LEVEL = 1  # CORRECTED from 0 (P1-2): opt_level=1 is the qiskit default, confirmed
# faster AND shallower than opt_level=0 at this cell -- see module docstring. This constant was
# never actually exercised to completion by either recorded attempt (both failed on noisy-execution
# time, unrelated to this parameter), so changing it does not retroactively alter any real result.
IDEAL_OPT_LEVEL = 1  # ideal/bare AerSimulator has no coupling-map constraint; opt_level=1 is cheap here regardless
COMPUTE_BUDGET_S = 2700.0  # 45 minutes -- generous vs the ~25-30 min estimate, an honest ceiling not a silent one


def _result_to_dict(r) -> dict:
    d = asdict(r)
    d["best_params"] = list(map(float, r.best_params))
    return d


def main() -> dict:
    t0 = time.perf_counter()
    print(f"Running EXP-D1 n_candidates={N_CANDIDATES} cell (fake_kingston at optimization_level={FAKE_KINGSTON_OPT_LEVEL})...")

    inst = generate_instance(FAMILY, N_CANDIDATES, seed=SEED)
    qubo = build_qubo(inst)
    print(f"instance: family={FAMILY} n_candidates={N_CANDIDATES} n_qubits={qubo.n_qubits()}")

    print("running ideal (bare AerSimulator) cell...")
    ideal_result = run_noise_validation_cell(
        inst, qubo, backend_label="ideal", backend=ideal_backend(),
        p=P, maxiter=MAXITER, shots=SHOTS, seed=SEED, optimization_level=IDEAL_OPT_LEVEL,
    )
    print(f"  ideal: depth={ideal_result.depth} 2q={ideal_result.two_qubit_gates} runtime={ideal_result.runtime_s:.1f}s")

    print("building FakeKingston-derived noisy simulator...")
    fk_backend = fake_kingston_backend()
    print(f"running fake_kingston cell at optimization_level={FAKE_KINGSTON_OPT_LEVEL} (this cell is the slow one -- expect ~20-30 min)...")
    fk_result = run_noise_validation_cell(
        inst, qubo, backend_label="fake_kingston", backend=fk_backend,
        p=P, maxiter=MAXITER, shots=SHOTS, seed=SEED, optimization_level=FAKE_KINGSTON_OPT_LEVEL,
    )
    print(f"  fake_kingston: depth={fk_result.depth} 2q={fk_result.two_qubit_gates} runtime={fk_result.runtime_s:.1f}s")

    total_runtime = time.perf_counter() - t0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "EXP-D1-n8": {
            "family": FAMILY,
            "n_candidates": N_CANDIDATES,
            "n_qubits": qubo.n_qubits(),
            "instance_fingerprint": inst.fingerprint(),
            "seed": SEED,
            "p": P,
            "maxiter": MAXITER,
            "shots": SHOTS,
            "ideal_optimization_level": IDEAL_OPT_LEVEL,
            "fake_kingston_optimization_level": FAKE_KINGSTON_OPT_LEVEL,
            "ideal": _result_to_dict(ideal_result),
            "fake_kingston": _result_to_dict(fk_result),
            "depth_ratio_fake_over_ideal": fk_result.depth / ideal_result.depth if ideal_result.depth else None,
            "resolution_note": (
                "CORRECTED 2026-10-01 (P1-2): this cell was never actually blocked by transpile "
                "time -- optimization_level=1 (used here, the qiskit default) transpiles this "
                "exact cell in ~3s, confirmed by calibrate_transpile_optimization_level.py. If "
                "this script completed successfully, the real bottleneck it had to get past was "
                "noisy-execution time (the FakeKingston backend.run() calls inside COBYLA), not "
                "transpilation -- see the module docstring for the full corrected account."
            ),
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"EXP-D1 n_candidates={N_CANDIDATES} cell complete in {total_runtime:.1f}s. Wrote {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
