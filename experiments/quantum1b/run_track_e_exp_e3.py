"""Track E runner: EXP-E3 -- IQAE vs classical MC, repeated-trial
statistical comparison, FakeKingston-NOISY simulator, crash-barrier
integrand, loose epsilon (per FUTURE_DIRECTIONS.md's explicit
recommendation and this module's own calibration -- see
experiments/quantum1b/qae_vs_mc.py's module docstring).

Resource discipline: BLAS threads pinned, single job, calibrated
beforehand (epsilon=0.05 -> ~5-8s/trial under FakeKingston noise, plus a
one-time ~11s noise-model build) -- 10 trials x 2 methods is a bounded,
safe job (~2 minutes).

Run with: `python3 -m experiments.quantum1b.run_track_e_exp_e3` from the
repo root.
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

from algorithms.qae_iqae.finance_integrands import build_crash_barrier  # noqa: E402
from experiments.quantum1b.qae_vs_mc import compare_qae_vs_mc_noisy, paired_significance_test  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_e_exp_e3.json"

N_TRIALS = 10
SEED = 0
COMPUTE_BUDGET_S = 900.0  # 15 minutes -- generous vs the ~2 minute calibrated estimate


def _result_to_dict(r) -> dict:
    d = asdict(r)
    for key in ("iqae", "classical_mc"):
        d[key]["trials"] = [asdict(t) for t in getattr(r, key).trials]
    return d


def main() -> dict:
    t0 = time.perf_counter()
    print("Running EXP-E3 (IQAE vs classical MC, FakeKingston-noisy, crash barrier)...")

    barrier = build_crash_barrier(n=3)
    print(f"integrand: {barrier.label}, exact_grid_a={barrier.exact_grid_a:.4f}")

    result = compare_qae_vs_mc_noisy(barrier, n_trials=N_TRIALS, seed=SEED)

    print(f"IQAE (noisy): bias={result.iqae.bias:+.4f} rmse={result.iqae.rmse:.4f} "
          f"coverage={result.iqae.empirical_ci_coverage:.2f} mean_oracle_queries={result.iqae.mean_oracle_queries:.0f} "
          f"mean_runtime={result.iqae.mean_runtime_s:.2f}s")
    print(f"classical_mc: bias={result.classical_mc.bias:+.4f} rmse={result.classical_mc.rmse:.4f} "
          f"coverage={result.classical_mc.empirical_ci_coverage:.2f} "
          f"mean_runtime={result.classical_mc.mean_runtime_s:.4f}s")

    sig = paired_significance_test(result.iqae, result.classical_mc, result.true_value_numeric)
    print(f"Mann-Whitney (IQAE vs classical_mc, noisy): p={sig.p_value:.5f} "
          f"iqae_significantly_more_precise={sig.a_significantly_more_precise}")

    total_runtime = time.perf_counter() - t0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-E3": {
            **_result_to_dict(result),
            "significance": asdict(sig),
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"EXP-E3 complete in {total_runtime:.1f}s. Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
