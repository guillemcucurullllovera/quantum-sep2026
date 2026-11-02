"""Track F runner: EXP-F1 -- QAOA warm-start/initialization-strategy
benchmark (cold-start-random vs linear-ramp vs INTERP) on
`FROZEN_CONTROLS`. See experiments/quantum1b/mixer_warmstart_bench.py for
the full methodology and why the XY/Dicke-mixer comparison is explicitly
NOT attempted.

Entirely classical-circuit simulation at tiny scale (9 qubits, the
FROZEN_CONTROLS size) -- fast, no special resource discipline needed
beyond this package's usual convention.

Run with: `python3 -m experiments.quantum1b.run_track_f_exp_f1` from the
repo root.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.quantum1b.mixer_warmstart_bench import run_frozen_controls_comparison

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_f_exp_f1.json"


def _group_stats(results: list) -> dict:
    gaps = [r.objective_gap for r in results if r.objective_gap is not None]
    evals = [r.n_evaluations for r in results]
    feas = [r.feasible_fraction for r in results]
    return {
        "n_runs": len(results),
        "n_runs_with_feasible_sample": len(gaps),
        "median_objective_gap": statistics.median(gaps) if gaps else None,
        "mean_objective_gap": statistics.fmean(gaps) if gaps else None,
        "median_n_evaluations": statistics.median(evals),
        "mean_n_evaluations": statistics.fmean(evals),
        "median_feasible_fraction": statistics.median(feas),
    }


def main() -> dict:
    t0 = time.perf_counter()
    print("Running EXP-F1 (QAOA warm-start/initialization-strategy benchmark on FROZEN_CONTROLS)...")

    per_instance = run_frozen_controls_comparison()

    manifest_instances = {}
    for instance_key, groups in per_instance.items():
        print(f"\n== {instance_key} ==")
        instance_manifest = {}
        for group_name, results in groups.items():
            stats = _group_stats(results)
            print(
                f"  {group_name:24s} n={stats['n_runs']} "
                f"median_gap={stats['median_objective_gap']} "
                f"median_evals={stats['median_n_evaluations']:.0f}"
            )
            instance_manifest[group_name] = {
                "stats": stats,
                "runs": [asdict(r) for r in results],
            }
        manifest_instances[instance_key] = instance_manifest

    total_runtime = time.perf_counter() - t0

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "EXP-F1": {
            "maxiter": 200,
            "note_xy_mixer": (
                "XY/Dicke-mixer (Gen2 QAOAv2) comparison deliberately NOT attempted -- "
                "structurally incompatible with this QUBO's free-ranging slack qubits. "
                "See mixer_warmstart_bench.py's module docstring and "
                "docs/quantum/QUANTUM1B_TRACK_F_MIXER_WARMSTART.md for the full reasoning."
            ),
            "instances": manifest_instances,
            "total_runtime_s": total_runtime,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nEXP-F1 complete in {total_runtime:.1f}s. Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
