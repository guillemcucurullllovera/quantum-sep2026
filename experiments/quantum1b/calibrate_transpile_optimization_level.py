"""Committed, reproducible calibration: `qiskit.transpile`'s
`optimization_level=1` (the qiskit default) vs `optimization_level=0`
against FakeKingston, for Track B's QAOA circuit at n_candidates=8
(14 qubits).

**Why this script exists.** The independent adversarial review
(`QUANTUM1B_INDEPENDENT_ADVERSARIAL_REVIEW_FINAL.md`, finding P1-2) found
that the original claim behind this platform's `optimization_level=0`
runbook instruction -- that `optimization_level=1` "did not complete
within 510s" at this exact cell -- does not reproduce. Three independent
review passes, plus this session's own follow-up reproduction, all found
`optimization_level=1` completes in ~3 seconds and produces a SHALLOWER
circuit (depth≈375) than `optimization_level=0` (depth≈486, ~0.02-0.03s).
No committed, re-runnable script backed the original claim (a second,
related finding, P2-14) -- this script is the remediation for both: a
deterministic, version-controlled measurement anyone can re-run, replacing
prose-only "ad hoc calibration calls."

**What this does NOT investigate:** what actually caused the original
>510s observation (host contention, an unrelated hang, an environment
mismatch) remains genuinely unknown -- see
`docs/quantum/QUANTUM1B_CVAR_QAOA.md`/`QUANTUM1B_HARDWARE_READINESS.md`'s
corrected text for the full, honest account. This script only answers
"does `optimization_level=1` reproducibly work fine at this cell, under
current pinned dependencies, on this host" -- yes, confirmed below -- not
"why did a different run once take longer."

Run with: `python3 -m experiments.quantum1b.calibrate_transpile_optimization_level`
from the repo root. Fast (~15s total, dominated by the one-time
~11s FakeKingston noise-model/backend build) -- safe to run standalone.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "transpile_optimization_level_calibration.json"

FAMILY = "independent"
N_CANDIDATES = 8
SEED = 0
GAMMA = [0.5]
BETA = [0.3]
OPT_LEVELS = (1, 0)  # qiskit default first, then the level this platform's runbook currently recommends


def main() -> dict:
    from qiskit import transpile

    from experiments.quantum1b.generators import generate_instance
    from experiments.quantum1b.hardware_readiness.lineage import build_qaoa_circuit_for_lineage
    from experiments.quantum1b.noise_validation import fake_kingston_backend
    from experiments.quantum1b.qubo_encoding import build_qubo

    print("Running optimization_level calibration (P1-2 remediation)...")

    inst = generate_instance(FAMILY, N_CANDIDATES, seed=SEED)
    qubo = build_qubo(inst)
    n_qubits = qubo.n_qubits()
    print(f"instance: family={FAMILY} n_candidates={N_CANDIDATES} n_qubits={n_qubits}")

    t0 = time.perf_counter()
    backend = fake_kingston_backend()
    backend_build_s = time.perf_counter() - t0
    print(f"backend built in {backend_build_s:.2f}s")

    qc = build_qaoa_circuit_for_lineage(qubo.Q, p=1, gamma=np.array(GAMMA), beta=np.array(BETA))

    rows = []
    for opt_level in OPT_LEVELS:
        t0 = time.perf_counter()
        tqc = transpile(qc, backend, optimization_level=opt_level, seed_transpiler=0)
        dt = time.perf_counter() - t0
        gate_counts = {k: int(v) for k, v in tqc.count_ops().items()}
        row = {
            "optimization_level": opt_level,
            "transpile_time_s": dt,
            "depth": int(tqc.depth()),
            "gate_counts": gate_counts,
            "total_ops": sum(gate_counts.values()),
        }
        rows.append(row)
        print(f"  optimization_level={opt_level}: {dt:.2f}s depth={row['depth']}")

    by_level = {r["optimization_level"]: r for r in rows}
    level1_shallower_or_equal = by_level[1]["depth"] <= by_level[0]["depth"]
    level1_faster = by_level[1]["transpile_time_s"] < 60  # nowhere near the originally-claimed >510s

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "P1-2 remediation: committed, reproducible calibration replacing the "
            "non-reproducing 'optimization_level=1 hangs past 510s' claim."
        ),
        "instance": {"family": FAMILY, "n_candidates": N_CANDIDATES, "seed": SEED, "n_qubits": n_qubits,
                      "fingerprint": inst.fingerprint()},
        "backend": backend.name,
        "backend_build_s": backend_build_s,
        "rows": rows,
        "finding": {
            "level1_completed_well_under_claimed_510s": level1_faster,
            "level1_depth_shallower_or_equal_to_level0": level1_shallower_or_equal,
            "conclusion": (
                "optimization_level=1 (the qiskit default) completes quickly and produces a "
                "SHALLOWER circuit than optimization_level=0 at this exact cell, directly "
                "contradicting the original '>510s, sharp threshold' claim. See "
                "docs/quantum/QUANTUM1B_HARDWARE_READINESS.md's corrected EXP-D1 section for "
                "the full retraction and what remains genuinely unknown about the original "
                "observation."
            ),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote results manifest to {RESULTS_PATH}")
    return manifest


if __name__ == "__main__":
    main()
