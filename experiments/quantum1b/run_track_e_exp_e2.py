"""Track E runner: EXP-E2 -- IQAE vs classical MC, rigorous repeated-trial
statistical comparison, IDEAL simulator, crash-barrier integrand.

See experiments/quantum1b/qae_vs_mc.py for the full methodology.

Resource discipline: ideal-simulator-only (no FakeKingston noise model
build), calibrated beforehand (epsilon=0.01 -> ~4s/trial under ideal
sim) -- 30 trials x 2 methods is a ~2-3 minute job, safely bounded.

Run with: `python3 -m experiments.quantum1b.run_track_e_exp_e2` from the
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
from experiments.quantum1b.qae_vs_mc import compare_qae_vs_mc_ideal, paired_significance_test  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_e_exp_e2.json"

N_TRIALS = 30
SEED = 0
COMPUTE_BUDGET_S = 600.0  # 10 minutes -- generous vs the ~2-3 minute calibrated estimate


def _result_to_dict(r) -> dict:
    d = asdict(r)
    for key in ("iqae", "classical_mc", "classical_mc_budget_matched"):
        summary = getattr(r, key)
        if summary is not None:
            d[key]["trials"] = [asdict(t) for t in summary.trials]
    return d


def main() -> dict:
    t0 = time.perf_counter()
    print("Running EXP-E2 (IQAE vs classical MC, ideal simulator, crash barrier)...")

    barrier = build_crash_barrier(n=3)
    print(f"integrand: {barrier.label}, exact_grid_a={barrier.exact_grid_a:.4f}, exact_analytic={barrier.exact_analytic:.4f}")

    result = compare_qae_vs_mc_ideal(barrier, n_trials=N_TRIALS, seed=SEED)

    print(f"IQAE:                        bias={result.iqae.bias:+.4f} rmse={result.iqae.rmse:.4f} "
          f"coverage={result.iqae.empirical_ci_coverage:.2f} mean_oracle_queries={result.iqae.mean_oracle_queries:.0f} "
          f"mean_runtime={result.iqae.mean_runtime_s:.2f}s")
    print(f"classical_mc (UNMATCHED):   bias={result.classical_mc.bias:+.4f} rmse={result.classical_mc.rmse:.4f} "
          f"coverage={result.classical_mc.empirical_ci_coverage:.2f} mean_oracle_queries={result.classical_mc.mean_oracle_queries:.0f} "
          f"mean_runtime={result.classical_mc.mean_runtime_s:.4f}s  <- {result.shots} samples, NOT the same "
          f"query budget as IQAE; see classical_mc_budget_matched for the fair comparison")
    if result.classical_mc_budget_matched is not None:
        m = result.classical_mc_budget_matched
        print(f"classical_mc (MATCHED):     bias={m.bias:+.4f} rmse={m.rmse:.4f} "
              f"coverage={m.empirical_ci_coverage:.2f} mean_oracle_queries={m.mean_oracle_queries:.0f} "
              f"mean_runtime={m.mean_runtime_s:.4f}s  <- same total query count as IQAE's own mean "
              f"oracle_queries, the methodologically fair comparison")

    sig_unmatched = paired_significance_test(result.iqae, result.classical_mc, result.true_value_numeric)
    sig_matched = paired_significance_test(result.iqae, result.classical_mc_budget_matched, result.true_value_numeric)
    print(f"Mann-Whitney (IQAE vs classical_mc UNMATCHED): p={sig_unmatched.p_value:.5f} "
          f"iqae_significantly_more_precise={sig_unmatched.a_significantly_more_precise}")
    print(f"Mann-Whitney (IQAE vs classical_mc MATCHED):   p={sig_matched.p_value:.5f} "
          f"iqae_significantly_more_precise={sig_matched.a_significantly_more_precise}")

    total_runtime = time.perf_counter() - t0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-E2": {
            **_result_to_dict(result),
            "significance_vs_unmatched": asdict(sig_unmatched),
            "significance_vs_budget_matched": asdict(sig_matched),
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"EXP-E2 complete in {total_runtime:.1f}s. Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
