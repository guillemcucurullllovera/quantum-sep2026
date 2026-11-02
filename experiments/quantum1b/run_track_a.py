"""Track A runner: executes the two preregistered classical-comparator
experiments (docs/quantum/QUANTUM1B_PREREGISTRATION.md, mirrored at
experiments/quantum1b/registry/EXP-A1.json and EXP-A2.json) exactly as
specified, and writes a results manifest to
`experiments/quantum1b/results/track_a_results.json`.

EXP-A1 -- classical comparator exactness: exhaustive enumeration (ground
truth) vs the Fortet-linearized MILP, all 9 synthetic families, n in
[4,6,8,10,12,14,16,18], seeds 0-4 (360 (family, n, seed) cells). PASS iff
100% agree within tolerance 1e-6 on objective value and feasible-set
membership.

EXP-A2 -- classical comparator scaling: family="dense" (worst case for the
O(n^2) Fortet linearization), n in [8,12,16,20,24], seeds 0-2 (15 cells).
120s wall-clock cap per solve; on a HiGHS timeout, falls back to
simulated_annealing_qubo (n_iters=20000, 5 restarts, seeds 0-4), labeled
heuristic. READY iff every n<=16 solves exactly within budget.

Run with: `python3 -m experiments.quantum1b.run_track_a` from the repo
root (the local research checkout).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import scipy

from core.evidence import compute_evidence_level
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from experiments.quantum1b.classical_exact import (
    MILPTimeoutError,
    solve_exact_enumeration,
    solve_exact_milp,
)
from experiments.quantum1b.classical_heuristic import solve_simulated_annealing
from experiments.quantum1b.generators import FAMILIES, generate_instance
from experiments.quantum1b.registry_schema import load_registry

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_a_results.json"

_CLASSICAL_BACKEND = BackendDescriptor(name="classical_highs", kind=BackendKind.SIMULATOR, num_qubits=0)


def _make_experiment_result(
    *, name: str, algorithm: str, seed: int, metrics: dict, notes: str
) -> tuple[ExperimentResult, str]:
    """Build a core.schemas.ExperimentResult for a pure-classical Track A
    result. There is no quantum backend concept for an analytical MILP/
    enumeration solve, so (per QUANTUM1B_PLAN.md section 2's own framing
    of this as evidence level A, "exact classical numerics, no execution
    uncertainty") we use BackendKind.SIMULATOR with num_qubits=0 and
    shots=None purely as the closest-fit convention within core.schemas'
    existing (quantum-backend-shaped) model -- documented, not asserted
    silently. `core.evidence.compute_evidence_level` itself has no "A"
    branch reachable from a plain SIMULATOR backend (its only paths from
    SIMULATOR are B/C/D depending on noise_model/shots) -- we call it
    anyway (using the machinery verbatim, as instructed) and record its
    computed value alongside the preregistration's own authoritative "A"
    label; see QUANTUM1B_CLASSICAL_BENCHMARK.md for the full discussion of
    this discrepancy.
    """
    result = ExperimentResult(
        metadata=RunMetadata(algorithm=algorithm, seed=seed, generation=None),
        config=ExperimentConfig(
            name=name,
            algorithm=algorithm,
            application="quantum1b_track_a",
            backend=_CLASSICAL_BACKEND,
            shots=None,
            seed=seed,
        ),
        metrics=metrics,
        classical_baseline_metrics=None,
        financial_edge_claimed=False,
        notes=notes,
    )
    computed_level = compute_evidence_level(result)
    return result, computed_level.value


def run_exp_a1(registry) -> dict:
    exp = registry["EXP-A1"]
    sizes = [4, 6, 8, 10, 12, 14, 16, 18]
    seeds = list(exp.seeds)
    assert seeds == [0, 1, 2, 3, 4], f"EXP-A1 registry seeds drifted: {seeds}"

    t0 = time.perf_counter()
    cells = []
    n_agree = 0
    max_abs_diff = 0.0

    for family in FAMILIES:
        for n in sizes:
            for seed in seeds:
                inst = generate_instance(family, n, seed=seed)
                r_enum = solve_exact_enumeration(inst)
                r_milp = solve_exact_milp(inst)
                abs_diff = abs(r_enum.objective - r_milp.objective)
                feasibility_agree = r_enum.feasible == r_milp.feasible
                objective_agree = abs_diff < 1e-6
                agree = feasibility_agree and objective_agree
                if agree:
                    n_agree += 1
                max_abs_diff = max(max_abs_diff, abs_diff)
                cells.append(
                    {
                        "family": family,
                        "n": n,
                        "seed": seed,
                        "instance_id": inst.instance_id,
                        "fingerprint": inst.fingerprint(),
                        "enum_objective": r_enum.objective,
                        "milp_objective": r_milp.objective,
                        "enum_feasible": r_enum.feasible,
                        "milp_feasible": r_milp.feasible,
                        "abs_diff": abs_diff,
                        "agree": agree,
                        "enum_runtime_s": r_enum.runtime_s,
                        "milp_runtime_s": r_milp.runtime_s,
                        "n_y_auxiliary": r_milp.solver_info["n_y_auxiliary"],
                    }
                )

    total_runtime = time.perf_counter() - t0
    n_cells = len(cells)
    passed = n_agree == n_cells

    result, computed_evidence_level = _make_experiment_result(
        name="EXP-A1",
        algorithm="classical_exact_milp_vs_enumeration",
        seed=0,
        metrics={
            "n_cells": float(n_cells),
            "n_agree": float(n_agree),
            "fraction_agree": n_agree / n_cells,
            "max_abs_objective_diff": max_abs_diff,
            "total_runtime_s": total_runtime,
        },
        notes=(
            f"EXP-A1 classical comparator exactness: {n_agree}/{n_cells} cells agree "
            f"within 1e-6. PASS={passed}."
        ),
    )

    return {
        "experiment_id": "EXP-A1",
        "hypothesis": exp.hypothesis,
        "families": list(FAMILIES),
        "sizes": sizes,
        "seeds": seeds,
        "n_cells": n_cells,
        "n_agree": n_agree,
        "fraction_agree": n_agree / n_cells,
        "max_abs_objective_diff": max_abs_diff,
        "pass": passed,
        "total_runtime_s": total_runtime,
        "compute_budget_s": 300.0,
        "within_compute_budget": total_runtime < 300.0,
        "preregistered_evidence_level": exp.evidence_level,
        "computed_evidence_level_via_core_evidence": computed_evidence_level,
        "experiment_result": json.loads(result.model_dump_json()),
        "cells": cells,
    }


def run_exp_a2(registry) -> dict:
    exp = registry["EXP-A2"]
    sizes = [8, 12, 16, 20, 24]
    seeds = list(exp.seeds)
    assert seeds == [0, 1, 2], f"EXP-A2 registry seeds drifted: {seeds}"
    family = "dense"
    time_limit_s = 120.0

    t0 = time.perf_counter()
    cells = []
    any_timeout = False
    exact_ns: set[int] = set()

    for n in sizes:
        for seed in seeds:
            inst = generate_instance(family, n, seed=seed)
            cell = {
                "family": family,
                "n": n,
                "seed": seed,
                "instance_id": inst.instance_id,
                "time_limit_s": time_limit_s,
            }
            try:
                r = solve_exact_milp(inst, time_limit_s=time_limit_s)
                cell.update(
                    {
                        "timed_out": False,
                        "method": r.method,
                        "runtime_s": r.runtime_s,
                        "n_y_auxiliary": r.solver_info["n_y_auxiliary"],
                        "highs_status": r.solver_info["status"],
                        "objective": r.objective,
                        "feasible": r.feasible,
                    }
                )
                exact_ns.add(n)
            except MILPTimeoutError as e:
                any_timeout = True
                cell.update(
                    {
                        "timed_out": True,
                        "milp_runtime_s": e.runtime_s,
                        "milp_solver_info": e.solver_info,
                    }
                )
                r_sa = solve_simulated_annealing(inst, n_iters=20000, n_restarts=5, seed=0)
                cell.update(
                    {
                        "method": "simulated_annealing_fallback",
                        "runtime_s": r_sa.runtime_s,
                        "objective": r_sa.objective,
                        "feasible": r_sa.feasible,
                        "sa_solver_info": r_sa.solver_info,
                    }
                )
            cells.append(cell)

    total_runtime = time.perf_counter() - t0
    # READY iff every n<=16 solved exactly (no timeout) within budget.
    required_exact_ns = {n for n in sizes if n <= 16}
    ready = required_exact_ns.issubset(exact_ns)
    largest_exact_n = max(exact_ns) if exact_ns else None

    result, computed_evidence_level = _make_experiment_result(
        name="EXP-A2",
        algorithm="classical_exact_milp_scaling",
        seed=0,
        metrics={
            "n_cells": float(len(cells)),
            "any_timeout": float(any_timeout),
            "largest_exact_n": float(largest_exact_n) if largest_exact_n is not None else -1.0,
            "total_runtime_s": total_runtime,
        },
        notes=(
            f"EXP-A2 classical comparator scaling: READY={ready}, "
            f"largest_exact_n={largest_exact_n}, any_timeout={any_timeout}."
        ),
    )

    return {
        "experiment_id": "EXP-A2",
        "hypothesis": exp.hypothesis,
        "family": family,
        "sizes": sizes,
        "seeds": seeds,
        "time_limit_s": time_limit_s,
        "any_timeout": any_timeout,
        "largest_exact_n": largest_exact_n,
        "ready": ready,
        "total_runtime_s": total_runtime,
        "compute_budget_s": 1800.0,
        "within_compute_budget": total_runtime < 1800.0,
        "preregistered_evidence_level": exp.evidence_level,
        "computed_evidence_level_via_core_evidence": computed_evidence_level,
        "experiment_result": json.loads(result.model_dump_json()),
        "cells": cells,
    }


def main() -> dict:
    registry = load_registry()
    print("Running EXP-A1 (classical comparator exactness)...")
    a1 = run_exp_a1(registry)
    print(
        f"EXP-A1: {a1['n_agree']}/{a1['n_cells']} cells agree "
        f"(PASS={a1['pass']}), runtime={a1['total_runtime_s']:.1f}s"
    )

    print("Running EXP-A2 (classical comparator scaling)...")
    a2 = run_exp_a2(registry)
    print(
        f"EXP-A2: largest_exact_n={a2['largest_exact_n']}, any_timeout={a2['any_timeout']}, "
        f"READY={a2['ready']}, runtime={a2['total_runtime_s']:.1f}s"
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "EXP-A1": a1,
        "EXP-A2": a2,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
