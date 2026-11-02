"""Track D runner: EXP-D1 seed replication -- does the depth-inflation
ratio (FakeKingston-transpiled depth / ideal depth) genuinely grow with
n_qubits, or was the "12.6x at n=9 -> 20.4x at n=12" finding an artifact of
having exactly ONE single-seed instance per rung?

**Why this exists.** Independent-review finding P2-13 ("'Genuine trend,
not noise' asserted from 2 unreplicated, single-seed data points"): with
exactly two (n, ratio) pairs, each from a single seed/instance, "trend vs.
noise" is not statistically distinguishable. The review's own suggested
remediation: "Rerun EXP-D1's ladder with 3+ seeds per rung before
asserting a trend." This script does exactly that, for the two rungs that
actually completed (n_candidates=3 and 6 -- see
`results/track_d_exp_d1_ladder.json`; n_candidates=8 did not complete and
is out of scope here).

**Why this is cheap, unlike the original ladder measurements.** The
original EXP-D1 cells paid for a full noisy-COBYLA optimization loop
(repeated `backend.run()` calls) to report BOTH depth/gate-count AND
objective-quality metrics. Transpiled circuit DEPTH, however, is a
function of the circuit's GATE STRUCTURE (which qubits a p=1 QAOA circuit
entangles, determined by the QUBO's nonzero Ising coupling terms) and the
backend's coupling map -- NOT of the specific rotation ANGLES COBYLA would
have found. This script transpiles each seed's circuit ONCE at a single,
fixed, arbitrary (gamma, beta) (same convention as
`calibrate_transpile_optimization_level.py`), skipping the optimization
loop entirely -- a measurement of the SAME quantity (depth-inflation
ratio) the original ladder reported, at a small fraction of the cost,
making a genuine 5-seeds-per-rung replication practical on a shared host.

Run with: `python3 -m experiments.quantum1b.run_track_d_exp_d1_seed_replication`
from the repo root.
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
RESULTS_PATH = RESULTS_DIR / "track_d_exp_d1_seed_replication.json"

FAMILY = "independent"
N_CANDIDATES_RUNGS = (3, 6)  # the two EXP-D1 ladder rungs that actually completed
SEEDS = (0, 1, 2, 3, 4)  # 5 seeds per rung -- exceeds the review's own "3+" minimum
GAMMA = [0.5]
BETA = [0.3]
OPTIMIZATION_LEVEL = 1  # the qiskit default, corrected convention per P1-2
COMPUTE_BUDGET_S = 300.0


def main() -> dict:
    from qiskit import transpile

    from experiments.quantum1b.generators import generate_instance
    from experiments.quantum1b.hardware_readiness.lineage import build_qaoa_circuit_for_lineage
    from experiments.quantum1b.noise_validation import fake_kingston_backend, ideal_backend
    from experiments.quantum1b.qubo_encoding import build_qubo

    t0 = time.perf_counter()
    print("Running EXP-D1 seed replication (P2-13 remediation)...")

    ideal = ideal_backend()
    fake = fake_kingston_backend()

    rows = []
    for n_candidates in N_CANDIDATES_RUNGS:
        for seed in SEEDS:
            inst = generate_instance(FAMILY, n_candidates, seed=seed)
            qubo = build_qubo(inst)
            qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=np.array(GAMMA), beta=np.array(BETA))

            tqc_ideal = transpile(qc, ideal, optimization_level=OPTIMIZATION_LEVEL, seed_transpiler=seed)
            tqc_fake = transpile(qc, fake, optimization_level=OPTIMIZATION_LEVEL, seed_transpiler=seed)

            ideal_depth = int(tqc_ideal.depth())
            fake_depth = int(tqc_fake.depth())
            ratio = fake_depth / ideal_depth if ideal_depth else None

            rows.append(
                {
                    "n_candidates": n_candidates,
                    "n_qubits": qubo.n_qubits(),
                    "seed": seed,
                    "instance_fingerprint": inst.fingerprint(),
                    "ideal_depth": ideal_depth,
                    "fake_kingston_depth": fake_depth,
                    "depth_ratio_fake_over_ideal": ratio,
                }
            )
            print(f"  n_candidates={n_candidates} seed={seed}: ideal={ideal_depth} fake={fake_depth} ratio={ratio:.2f}")

    total_runtime = time.perf_counter() - t0

    # Per-rung aggregate statistics.
    by_n = {}
    for n_candidates in N_CANDIDATES_RUNGS:
        ratios = [r["depth_ratio_fake_over_ideal"] for r in rows if r["n_candidates"] == n_candidates]
        by_n[n_candidates] = {
            "n_seeds": len(ratios),
            "mean_ratio": float(np.mean(ratios)),
            "std_ratio": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
            "ratios": ratios,
        }

    # Mann-Whitney U test: is the n=6 rung's ratio distribution genuinely
    # higher than n=3's, or statistically indistinguishable at this sample
    # size? Same test convention EXP-E2/E3 already use in this estate.
    from scipy.stats import mannwhitneyu

    ratios_n3 = by_n[N_CANDIDATES_RUNGS[0]]["ratios"]
    ratios_n6 = by_n[N_CANDIDATES_RUNGS[1]]["ratios"]
    try:
        u_stat, p_value = mannwhitneyu(ratios_n6, ratios_n3, alternative="greater")
        mw_result = {"u_statistic": float(u_stat), "p_value": float(p_value), "computed": True}
    except ValueError as e:  # pragma: no cover -- defensive, reported not swallowed
        mw_result = {"u_statistic": None, "p_value": None, "computed": False, "reason": str(e)}

    significant = mw_result.get("computed") and mw_result.get("p_value") is not None and mw_result["p_value"] < 0.05

    conclusion = (
        f"With {len(SEEDS)} seeds per rung (vs. the original single-seed-per-rung measurement), "
        f"n_candidates=3 (9 qubits) has mean depth-inflation ratio {by_n[N_CANDIDATES_RUNGS[0]]['mean_ratio']:.2f} "
        f"(std={by_n[N_CANDIDATES_RUNGS[0]]['std_ratio']:.2f}), n_candidates=6 (12 qubits) has mean "
        f"{by_n[N_CANDIDATES_RUNGS[1]]['mean_ratio']:.2f} (std={by_n[N_CANDIDATES_RUNGS[1]]['std_ratio']:.2f}). "
        f"Mann-Whitney U (one-sided, n=6 ratios > n=3 ratios): p={mw_result.get('p_value')}. "
        + (
            "This IS now a statistically significant difference at p<0.05 -- the growth direction "
            "from the original 2-point measurement is corroborated with real replication, though "
            "this remains 2 rungs, not a full scaling-law fit across many n."
            if significant
            else "This is NOT statistically significant at p<0.05 given this sample size -- the "
            "original single-seed 12.6x->20.4x finding's DIRECTION may still be real, but this "
            "replication cannot confirm it at conventional significance; report as 'suggestive, "
            "not confirmed' until a larger sample or more rungs are measured."
        )
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-D1-seed-replication": {
            "family": FAMILY,
            "n_candidates_rungs": list(N_CANDIDATES_RUNGS),
            "seeds": list(SEEDS),
            "optimization_level": OPTIMIZATION_LEVEL,
            "rows": rows,
            "by_rung": by_n,
            "mann_whitney_u_n6_ratio_greater_than_n3_ratio": mw_result,
            "significant_at_p_lt_0.05": significant,
            "conclusion": conclusion,
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{conclusion}")
    print(f"EXP-D1 seed replication complete in {total_runtime:.1f}s. Wrote {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
