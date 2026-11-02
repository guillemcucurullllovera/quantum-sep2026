"""Track E runner: EXP-E4 -- hardware-aware circuit-cost scaling for
IQAE's Grover-amplified ansatz, closing the "HARDWARE-AWARE COST" step of
the required research standard for Track E (EXP-E2/E3 cover IDEAL and
NOISE-AWARE simulation; this closes the next step before any real-hardware
consideration, matching what Track D's EXP-D1 already did for Track B's
QAOA circuits).

**Reuses, does not reimplement:** `algorithms.qae_iqae.iqae._grover_metadata`
-- already-existing, already-used-in-production (`run_barrier_demo`)
infrastructure that transpiles a Grover-amplified IQAE circuit against a
FakeBackend at `optimization_level=0`, that function's own pre-existing
convention (predates this session entirely), reused as-is here without
re-justifying it. `run_barrier_demo`'s own default only probes
`ks=(0, 1, 2)` -- far short of the regime EXP-E2's `epsilon=0.01` run
actually reaches (`mean_oracle_queries~16247`, implying substantial
Grover amplification across IQAE's adaptive rounds). This script extends
the SAME function to a wider, doubling `ks` range, calibrated first at a
single large k before committing to a full sweep.

**Correction, 2026-10-01 (independent-review finding P1-2):** an earlier
version of this docstring described `optimization_level=0` here as "the
exact fix Track D's own EXP-D1 followup diagnostic independently found
for QAOA circuits." **That Track D finding did not reproduce** under
independent review -- `optimization_level=1` was confirmed fast and
shallow for Track B's QAOA circuit, contradicting the original "fix"
claim entirely (see `docs/quantum/QUANTUM1B_HARDWARE_READINESS.md`'s
corrected EXP-D1 section). This script's own measurement (below) is
unaffected -- `optimization_level=0` genuinely does transpile this IQAE
circuit cleanly up to k=128 with no wall, which is this track's own,
independently-measured result -- but it is no longer cross-referenced as
confirming or matching Track D's claim, since there was nothing real on
that side to match. Whether `optimization_level=1` would do even better
here was not tested.

Resource discipline: a single large-k calibration (`ks=(0, 32, 128)`,
~5.4s measured) was run BEFORE committing to this script's full sweep, to
confirm `optimization_level=0` stays fast at this scale for THIS circuit
(confirmed -- see docs/quantum/QUANTUM1B_TRACK_E_AE_VS_MC.md). The full
sweep below is a strict superset of that calibration, safely bounded.
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
from algorithms.qae_iqae.iqae import _grover_metadata  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_e_exp_e4.json"

KS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
COMPUTE_BUDGET_S = 300.0  # 5 minutes -- generous vs the ~5-15s calibrated/measured cost


def main() -> dict:
    t0 = time.perf_counter()
    print("Running EXP-E4 (IQAE Grover-amplified circuit-cost scaling, FakeKingston)...")

    from qiskit_ibm_runtime.fake_provider import FakeKingston

    barrier = build_crash_barrier(n=3)
    A = build_A_barrier(barrier.probs, barrier.barrier_mask)
    anc = int(round(math.log2(len(barrier.probs))))
    fake = FakeKingston()

    meta = _grover_metadata(A, anc, fake, ks=KS)
    for k, row in meta.items():
        print(f"  {k}: depth={row['depth']} twoq_gates={row['twoq_gates']} total_ops={row['total_ops']}")

    # Fit a linear depth-vs-k trend (depth = intercept + slope*k) -- the
    # measured numbers show this is a clean linear relationship (one
    # Grover operator adds a roughly fixed depth increment), reported as a
    # fit, not assumed.
    ks_list = list(KS)
    depths = [meta[f"Q{k}A"]["depth"] for k in ks_list]
    n = len(ks_list)
    mean_k = sum(ks_list) / n
    mean_d = sum(depths) / n
    slope = sum((k - mean_k) * (d - mean_d) for k, d in zip(ks_list, depths)) / sum((k - mean_k) ** 2 for k in ks_list)
    intercept = mean_d - slope * mean_k

    total_runtime = time.perf_counter() - t0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-E4": {
            "integrand_label": barrier.label,
            "backend": fake.name,
            "backend_num_qubits": int(fake.num_qubits),
            "optimization_level": 0,
            "ks": list(KS),
            "circuits": meta,
            "depth_vs_k_linear_fit": {"slope_depth_per_grover_op": slope, "intercept_depth": intercept},
            "context_from_exp_e2": {
                "epsilon": 0.01,
                "mean_oracle_queries": 16247.5,
                "note": (
                    "EXP-E2's epsilon=0.01 ideal run used ~16247 total oracle queries summed across "
                    "IQAE's adaptive rounds, NOT the depth of one circuit -- the deepest SINGLE circuit "
                    "in that run (the last round's k) was not directly instrumented by this script, but "
                    "the measured linear depth-vs-k trend here (~1100 depth per Grover operator) means "
                    "even a modest k in the tens already implies five-digit circuit depth, and any k "
                    "anywhere near the hundreds (plausible for this epsilon) implies depths in the "
                    "100,000+ range -- clearly far beyond any near-term hardware's practical coherence "
                    "budget, independent of exactly which k IQAE's adaptive schedule used."
                ),
            },
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nLinear fit: depth ~= {intercept:.1f} + {slope:.1f} * k")
    print(f"EXP-E4 complete in {total_runtime:.1f}s. Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
