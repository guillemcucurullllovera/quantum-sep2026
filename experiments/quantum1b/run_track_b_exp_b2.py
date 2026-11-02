"""Track B runner: EXP-B2 -- shot sensitivity
(docs/quantum/QUANTUM1B_PREREGISTRATION.md EXP-B2,
experiments/quantum1b/registry/EXP-B2.json).

Reads `experiments/quantum1b/results/track_b_exp_b1.json` (must be
generated first by `run_track_b_exp_b1.py`) to find the best
(family, n, p, alpha) cell -- "best" meaning the CVaR-QAOA (alpha in
{0.5, 0.2}) cell with the smallest MEDIAN objective_gap across its 5
optimizer-restart seeds. Within that cell, the specific seed achieving the
minimum objective_gap is picked as "its trained circuit" (the task brief
says "pick the best cell... then finite-shot-sample its trained circuit",
singular -- a cell has 5 seeded trained circuits, so a specific seed must
be chosen; the minimum-objective-gap seed within the winning cell is the
most defensible single choice, and is documented as such in the output
manifest).

The QAOA run for that exact (qubo, p, alpha, seed) is deterministic (see
tests/quantum1b/test_cvar_qaoa.py::TestDeterminism), so it is re-run here
(cheap, a single COBYLA optimization) to reconstruct the trained
`QAOARunResult` rather than trying to serialize a 2^n-dimensional
probability array through the EXP-B1 JSON manifest.

Metric: |CVaR_shots - CVaR_exact| vs shots in [256, 1024, 4096, 16384], 10
seeds per shot count (`sample_bitstrings` determinism seed, NOT the QAOA
optimizer-restart seed -- that is fixed to the winning cell's chosen seed
throughout this experiment).

Run with: `python3 -m experiments.quantum1b.run_track_b_exp_b2` from the
repo root, after `run_track_b_exp_b1.py` has produced
`results/track_b_exp_b1.json`.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import scipy

from core.evidence import compute_evidence_level
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from experiments.quantum1b.cvar_qaoa import (
    exact_distribution_metrics,
    run_cvar_qaoa,
    sample_bitstrings,
)
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.metrics import evaluate_sample_set
from experiments.quantum1b.qubo_encoding import build_qubo, selection_from_bits
from experiments.quantum1b.registry_schema import load_registry

RESULTS_DIR = Path(__file__).parent / "results"
EXP_B1_PATH = RESULTS_DIR / "track_b_exp_b1.json"
RESULTS_PATH = RESULTS_DIR / "track_b_exp_b2.json"

SHOT_GRID = (256, 1024, 4096, 16384)
SHOT_SEEDS = tuple(range(10))

_QUANTUM_BACKEND_SHOTS = BackendDescriptor(
    name="numpy_statevector_exact", kind=BackendKind.SIMULATOR, num_qubits=20
)


def _make_experiment_result(*, name, algorithm, seed, shots, metrics, notes):
    result = ExperimentResult(
        metadata=RunMetadata(algorithm=algorithm, seed=seed, generation=None),
        config=ExperimentConfig(
            name=name,
            algorithm=algorithm,
            application="quantum1b_track_b",
            backend=_QUANTUM_BACKEND_SHOTS,
            shots=shots,  # finite-shot sampling from the exact distribution -> evidence level C
            seed=seed,
        ),
        metrics=metrics,
        classical_baseline_metrics=None,
        financial_edge_claimed=False,
        notes=notes,
    )
    computed_level = compute_evidence_level(result)
    return result, computed_level.value


def _find_best_cell(exp_b1_manifest: dict) -> dict:
    cells = exp_b1_manifest["EXP-B1"]["cells"]
    grouped: dict[tuple, list[dict]] = {}
    for c in cells:
        if c["alpha"] == 1.0:
            continue  # EXP-B2 picks from CVaR-QAOA cells only (alpha < 1.0)
        key = (c["family"], c["n"], c["p"], c["alpha"])
        grouped.setdefault(key, []).append(c)

    best_key = None
    best_median_gap = None
    for key, runs in grouped.items():
        gaps = [r["objective_gap"] for r in runs if r["objective_gap"] is not None]
        if not gaps:
            continue
        m = median(gaps)
        if best_median_gap is None or m < best_median_gap:
            best_median_gap = m
            best_key = key

    if best_key is None:
        raise RuntimeError("EXP-B2: no CVaR-QAOA cell with a defined objective_gap found in EXP-B1 results")

    family, n, p, alpha = best_key
    runs = grouped[best_key]
    best_run = min(
        (r for r in runs if r["objective_gap"] is not None), key=lambda r: r["objective_gap"]
    )
    return {
        "family": family,
        "n": n,
        "p": p,
        "alpha": alpha,
        "seed": best_run["seed"],
        "cell_median_objective_gap": best_median_gap,
        "chosen_seed_objective_gap": best_run["objective_gap"],
        "n_seeds_in_cell": len(runs),
    }


def run_exp_b2(registry) -> dict:
    exp = registry["EXP-B2"]

    if not EXP_B1_PATH.exists():
        raise FileNotFoundError(
            f"EXP-B2 requires {EXP_B1_PATH} (run run_track_b_exp_b1.py first)"
        )
    with open(EXP_B1_PATH) as f:
        exp_b1_manifest = json.load(f)

    winner = _find_best_cell(exp_b1_manifest)
    print(
        f"EXP-B2: best cell family={winner['family']} n={winner['n']} p={winner['p']} "
        f"alpha={winner['alpha']} seed={winner['seed']} "
        f"(cell median gap={winner['cell_median_objective_gap']:.4f})"
    )

    t0 = time.perf_counter()
    inst = generate_instance(winner["family"], winner["n"], seed=0)
    qubo = build_qubo(inst)
    qaoa_result = run_cvar_qaoa(
        qubo, p=winner["p"], alpha=winner["alpha"], seed=winner["seed"], maxiter=200
    )

    exact_metrics = exact_distribution_metrics(
        qaoa_result, qubo, inst, cvar_alpha=winner["alpha"]
    )
    cvar_exact = exact_metrics.cvar_objective

    shot_cells = []
    for shots in SHOT_GRID:
        abs_errors = []
        for shot_seed in SHOT_SEEDS:
            samples = sample_bitstrings(qaoa_result, shots, seed=shot_seed)
            full_selections = [tuple(selection_from_bits(qubo, inst, s)) for s in samples]
            sampled_metrics = evaluate_sample_set(
                inst, full_selections, cvar_alpha=winner["alpha"]
            )
            abs_err = abs(sampled_metrics.cvar_objective - cvar_exact)
            abs_errors.append(abs_err)
            shot_cells.append(
                {
                    "shots": shots,
                    "shot_seed": shot_seed,
                    "cvar_shots": sampled_metrics.cvar_objective,
                    "cvar_exact": cvar_exact,
                    "abs_error": abs_err,
                }
            )
        print(
            f"  shots={shots:6d}: mean|CVaR_shots-CVaR_exact|="
            f"{np.mean(abs_errors):.5f}  std={np.std(abs_errors):.5f}"
        )

    total_runtime = time.perf_counter() - t0

    shots_summary = []
    for shots in SHOT_GRID:
        errs = [c["abs_error"] for c in shot_cells if c["shots"] == shots]
        shots_summary.append(
            {
                "shots": shots,
                "mean_abs_error": float(np.mean(errs)),
                "std_abs_error": float(np.std(errs)),
                "min_abs_error": float(np.min(errs)),
                "max_abs_error": float(np.max(errs)),
            }
        )

    monotonic_decreasing = all(
        shots_summary[i]["mean_abs_error"] >= shots_summary[i + 1]["mean_abs_error"] - 1e-9
        for i in range(len(shots_summary) - 1)
    )

    result, computed_evidence_level = _make_experiment_result(
        name="EXP-B2",
        algorithm="cvar_qaoa_shot_sensitivity",
        seed=winner["seed"],
        shots=SHOT_GRID[-1],
        metrics={
            f"mean_abs_error_shots_{s['shots']}": s["mean_abs_error"] for s in shots_summary
        }
        | {"total_runtime_s": total_runtime},
        notes=(
            f"EXP-B2 shot sensitivity on best cell "
            f"family={winner['family']} n={winner['n']} p={winner['p']} "
            f"alpha={winner['alpha']}: mean_abs_error monotonically "
            f"non-increasing with shots={monotonic_decreasing}. Descriptive "
            "only, no pass/fail gate per preregistration."
        ),
    )

    manifest = {
        "experiment_id": "EXP-B2",
        "hypothesis": exp.hypothesis,
        "winning_cell": winner,
        "cvar_exact": cvar_exact,
        "shot_grid": list(SHOT_GRID),
        "shot_seeds": list(SHOT_SEEDS),
        "shots_summary": shots_summary,
        "monotonic_decreasing_mean_error": monotonic_decreasing,
        "shot_cells": shot_cells,
        "total_runtime_s": total_runtime,
        "compute_budget_s": 900.0,
        "within_compute_budget": total_runtime < 900.0,
        "preregistered_evidence_level": exp.evidence_level,
        "computed_evidence_level_via_core_evidence": computed_evidence_level,
        "experiment_result": json.loads(result.model_dump_json()),
        "prohibited_conclusion": exp.prohibited_conclusion,
    }
    return manifest


def main() -> dict:
    registry = load_registry()
    print("Running EXP-B2 (shot sensitivity)...")
    b2 = run_exp_b2(registry)
    print(f"EXP-B2 complete: runtime={b2['total_runtime_s']:.1f}s")

    full_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "EXP-B2": b2,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"Wrote results manifest to {RESULTS_PATH}")

    return full_manifest


if __name__ == "__main__":
    main()
