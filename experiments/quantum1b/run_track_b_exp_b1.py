"""Track B runner: EXP-B1 -- expectation-QAOA vs CVaR-QAOA objective gap
(docs/quantum/QUANTUM1B_PREREGISTRATION.md EXP-B1,
experiments/quantum1b/registry/EXP-B1.json).

Ladder: n in [4,6,8,10] (already reduced from the original [4..14] draft --
see QUANTUM1B_PREREGISTRATION.md "Preregistration deviations" #1, discovered
during qubo_encoding.py implementation, BEFORE any QAOA run -- this script
does not introduce a new reduction, it runs the already-disclosed reduced
ladder). Families: independent, correlated, tight_budget, degenerate,
concentrated, conflicting_constraints (one instance per family per n, seed 0
for instance generation). Sweep: p in [1,2], alpha in [1.0, 0.5, 0.2], 5
optimizer-restart seeds [0..4], COBYLA maxiter=200 -- 6*4*2*3*5 = 720 runs.

A pre-flight timing calibration (see this module's `_calibrate`) measured
the full 720-run sweep at ~4-5 minutes wall-clock (max n_qubits=17 across
the whole ladder; COBYLA converges in ~35-45 evaluations per run well
before the maxiter=200 cap in nearly every cell) -- well inside the
preregistered 4-hour budget, so NO further reduction (seeds/families/p) was
applied beyond the already-disclosed n-ladder reduction. This is recorded
explicitly in the output manifest's `compute_budget` section.

Ground truth: Track A's `classical_exact.solve_exact_milp` on the IDENTICAL
`OptionSelectionInstance` (same `qubo_encoding.build_qubo` call Track B's
QAOA itself uses to build its Ising Hamiltonian).

Run with: `python3 -m experiments.quantum1b.run_track_b_exp_b1` from the
repo root (the local research checkout), using
`.venv/bin/python3` (the interpreter
with pydantic/scipy already verified present -- see
docs/quantum/QUANTUM1B_CVAR_QAOA.md).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import scipy
from scipy.stats import wilcoxon

from core.evidence import compute_evidence_level
from core.schemas import (
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from experiments.quantum1b.classical_exact import solve_exact_milp
from experiments.quantum1b.cvar_qaoa import (
    exact_distribution_metrics,
    run_cvar_qaoa,
    run_expectation_qaoa,
)
from experiments.quantum1b.generators import generate_instance
from experiments.quantum1b.metrics import approximation_ratio, objective_gap
from experiments.quantum1b.qubo_encoding import build_qubo
from experiments.quantum1b.registry_schema import load_registry

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_b_exp_b1.json"

FAMILIES = (
    "independent",
    "correlated",
    "tight_budget",
    "degenerate",
    "concentrated",
    "conflicting_constraints",
)
SIZES = (4, 6, 8, 10)
P_VALUES = (1, 2)
ALPHAS = (1.0, 0.5, 0.2)
SEEDS = (0, 1, 2, 3, 4)
MAXITER = 200
COMPUTE_BUDGET_S = 4 * 3600.0

NEGLIGIBLE_FEASIBLE_PROBABILITY_THRESHOLD = 0.05
"""**Added 2026-10-01, remediating independent-review finding P2-18**
("EXP-B1's classifier code does not implement the full registered NO_GO
criterion" -- EXP-B1.json registers NO_GO as "no improvement, underperforms
expectation-QAOA, OR negligible feasible-probability across most cells,"
but the code previously only ever reached NO_GO via the first two clauses
-- there was no independent branch for "CVaR ties/improves on objective
gap while feasible-probability collapses toward 0 in most cells," which
would have been misclassified RESEARCH_ONLY instead of the registered
NO_GO. Chosen deliberately far below the 0.5 "majority_feasible" bar used
for the PROMISING_FOR_FURTHER_TESTING criterion -- 0.5 is "works more than
half the time," 0.05 is "effectively never produces a feasible sample,"
the actual "negligible" the registered rule describes. Did not change
EXP-B1's own actual reported classification (its real measured feasible
probabilities, ~0.245-0.256, are nowhere near negligible by this or any
reasonable threshold) -- this closes a latent classifier gap, confirmed
via a synthetic ladder engineered to trigger it (see
tests/quantum1b/test_run_track_b_exp_b1.py)."""

_QUANTUM_BACKEND = BackendDescriptor(
    name="numpy_statevector_exact", kind=BackendKind.SIMULATOR, num_qubits=20
)


def classify_cvar_qaoa_result(
    *, cvar_improves_or_ties: bool, significant: bool, majority_feasible: bool, majority_feasible_negligible: bool
) -> str:
    """Implements EXP-B1.json's registered `pass_fail_interpretation`
    exactly:

    - PROMISING_FOR_FURTHER_TESTING: improves-or-ties, significant
      (Wilcoxon p<0.05), AND majority-feasible (>=0.5 in most cells).
    - RESEARCH_ONLY: "directionally consistent but non-significant
      improvement, or works only on a subset of families."
    - NO_GO: "no improvement, underperforms expectation-QAOA, OR
      negligible feasible-probability across most cells."

    **Extracted into its own function 2026-10-01, remediating
    independent-review finding P2-18** ("EXP-B1's classifier code does not
    implement the full registered NO_GO criterion") -- the
    `majority_feasible_negligible` parameter and its NO_GO branch are the
    actual fix; the extraction itself is what makes this testable in
    isolation without re-running the full statevector sweep (see
    `tests/quantum1b/test_run_track_b_exp_b1.py`, including a synthetic
    scenario engineered to trigger the previously-missing branch:
    cvar_improves_or_ties=True with majority_feasible_negligible=True,
    which the OLD code would have misclassified RESEARCH_ONLY)."""
    if majority_feasible_negligible:
        return "NO_GO"
    if cvar_improves_or_ties and significant and majority_feasible:
        return "PROMISING_FOR_FURTHER_TESTING"
    if cvar_improves_or_ties and not (significant and majority_feasible):
        return "RESEARCH_ONLY"
    return "NO_GO"


def _make_experiment_result(*, name, algorithm, seed, metrics, notes):
    result = ExperimentResult(
        metadata=RunMetadata(algorithm=algorithm, seed=seed, generation=None),
        config=ExperimentConfig(
            name=name,
            algorithm=algorithm,
            application="quantum1b_track_b",
            backend=_QUANTUM_BACKEND,
            shots=None,  # exact statevector, no shot noise -> evidence level B
            seed=seed,
        ),
        metrics=metrics,
        classical_baseline_metrics=None,
        financial_edge_claimed=False,
        notes=notes,
    )
    computed_level = compute_evidence_level(result)
    return result, computed_level.value


def _run_one_cell(instance, qubo, opt_objective, *, p, alpha, seed):
    if alpha == 1.0:
        qaoa_result = run_expectation_qaoa(qubo, p=p, seed=seed, maxiter=MAXITER)
    else:
        qaoa_result = run_cvar_qaoa(qubo, p=p, alpha=alpha, seed=seed, maxiter=MAXITER)

    dist = exact_distribution_metrics(qaoa_result, qubo, instance, cvar_alpha=alpha)

    used_infeasible_best = dist.best_feasible_objective is None
    qaoa_best_for_gap = (
        dist.best_feasible_objective
        if dist.best_feasible_objective is not None
        else dist.best_objective
    )

    gap = objective_gap(opt_objective, qaoa_best_for_gap)
    ratio = (
        approximation_ratio(opt_objective, dist.best_feasible_objective)
        if dist.best_feasible_objective is not None
        else None
    )

    return {
        "p": p,
        "alpha": alpha,
        "seed": seed,
        "n_qubits": qaoa_result.n_qubits,
        "n_evaluations": qaoa_result.n_evaluations,
        "runtime_s": qaoa_result.runtime_s,
        "optimizer_status": qaoa_result.optimizer_status,
        "two_qubit_gate_count": qaoa_result.two_qubit_gate_count(),
        "objective_gap": gap,
        "used_infeasible_best_for_gap": used_infeasible_best,
        "best_sample_objective": dist.best_objective,
        "best_feasible_objective": dist.best_feasible_objective,
        "feasible_sample_probability": dist.feasible_fraction,
        "cvar_objective": dist.cvar_objective,
        "expected_objective": dist.expected_objective,
        "approximation_ratio": ratio,
        "constraint_violation_rate": dist.constraint_violation_rate,
        "risk_budget_utilization": dist.mean_risk_budget_utilization,
        "solution_diversity": dist.n_distinct_top_decile,
    }


def _calibrate() -> dict:
    """Time a handful of representative cells (smallest/largest n, p=2)
    before committing to the full sweep, per the preregistration's own
    anticipated calibration step."""
    samples = [
        ("independent", 4),
        ("degenerate", 10),  # worst case: largest n_qubits (17) in the ladder
        ("conflicting_constraints", 10),
    ]
    t0 = time.perf_counter()
    for fam, n in samples:
        inst = generate_instance(fam, n, seed=0)
        qubo = build_qubo(inst)
        run_expectation_qaoa(qubo, p=2, seed=0, maxiter=MAXITER)
        run_cvar_qaoa(qubo, p=2, alpha=0.2, seed=0, maxiter=MAXITER)
    elapsed = time.perf_counter() - t0
    # 6 cell-equivalent runs sampled above (2 per instance x 3 instances);
    # extrapolate to the full 720-run sweep conservatively (worst-case
    # n_qubits=17 instance included in the sample).
    per_run = elapsed / 6.0
    extrapolated_total = per_run * (len(FAMILIES) * len(SIZES) * len(P_VALUES) * len(ALPHAS) * len(SEEDS))
    return {
        "sample_cells": samples,
        "sample_elapsed_s": elapsed,
        "per_run_s_estimate": per_run,
        "extrapolated_total_s": extrapolated_total,
        "within_budget": extrapolated_total < COMPUTE_BUDGET_S,
    }


def run_exp_b1(registry) -> dict:
    exp = registry["EXP-B1"]

    calibration = _calibrate()
    print(
        f"Calibration: extrapolated total {calibration['extrapolated_total_s']:.1f}s "
        f"(budget {COMPUTE_BUDGET_S:.0f}s) -> within_budget={calibration['within_budget']}"
    )
    reduction_applied = not calibration["within_budget"]
    # Per the preregistration's documented fallback, if the calibration
    # extrapolation exceeded budget we would reduce scope HERE, before
    # running the full sweep, and record exactly what was cut. The
    # measured calibration came in far under budget (~minutes, not hours),
    # so no additional reduction beyond the already-disclosed n-ladder
    # reduction (n in [4,6,8,10], see module docstring) was necessary.
    seeds = list(SEEDS)
    p_values = list(P_VALUES)
    alphas = list(ALPHAS)
    families = list(FAMILIES)
    sizes = list(SIZES)

    t0 = time.perf_counter()
    cells = []
    instance_meta = {}

    for family in families:
        for n in sizes:
            inst = generate_instance(family, n, seed=0)
            qubo = build_qubo(inst)
            milp = solve_exact_milp(inst)
            opt_objective = milp.objective
            instance_meta[(family, n)] = {
                "instance_id": inst.instance_id,
                "fingerprint": inst.fingerprint(),
                "n_candidates_kept": len(qubo.candidate_ids),
                "n_slack_qubits": qubo.n_slack_qubits,
                "n_qubits": qubo.n_qubits(),
                "milp_objective": opt_objective,
                "milp_feasible": milp.feasible,
                "milp_runtime_s": milp.runtime_s,
            }
            for p in p_values:
                for alpha in alphas:
                    for seed in seeds:
                        cell = _run_one_cell(
                            inst, qubo, opt_objective, p=p, alpha=alpha, seed=seed
                        )
                        cell["family"] = family
                        cell["n"] = n
                        cells.append(cell)

    total_runtime = time.perf_counter() - t0
    print(f"EXP-B1 sweep complete: {len(cells)} cells in {total_runtime:.1f}s")

    # ------------------------------------------------------------------
    # Per-(family, n) aggregation: median objective_gap and median
    # feasible_sample_probability for expectation-QAOA (alpha=1.0) and for
    # EACH CVaR alpha, across (p, seed) -- 10 runs per (family, n, alpha).
    # ------------------------------------------------------------------
    fn_keys = [(f, n) for f in families for n in sizes]
    per_cell_alpha_gap = {}  # (family, n, alpha) -> list of gaps (None excluded)
    per_cell_alpha_feasprob = {}

    for c in cells:
        key = (c["family"], c["n"], c["alpha"])
        per_cell_alpha_gap.setdefault(key, []).append(c["objective_gap"])
        per_cell_alpha_feasprob.setdefault(key, []).append(c["feasible_sample_probability"])

    def _median_ignoring_none(values):
        vals = [v for v in values if v is not None]
        return median(vals) if vals else None

    cell_summaries = []
    excluded_cells = []
    expectation_gaps = []
    cvar_best_gaps = []
    cvar_best_alphas = []
    feasible_majority_flags = []
    feasible_negligible_flags = []

    for family, n in fn_keys:
        exp_gap = _median_ignoring_none(per_cell_alpha_gap[(family, n, 1.0)])
        exp_feasprob = _median_ignoring_none(per_cell_alpha_feasprob[(family, n, 1.0)])

        alpha_candidates = {}
        for alpha in (0.5, 0.2):
            g = _median_ignoring_none(per_cell_alpha_gap[(family, n, alpha)])
            fp = _median_ignoring_none(per_cell_alpha_feasprob[(family, n, alpha)])
            alpha_candidates[alpha] = (g, fp)

        # "CVaR-QAOA (best alpha)" per cell: whichever of {0.5, 0.2} has
        # the smaller median objective_gap (None treated as +inf -- worse).
        def _key(a):
            g = alpha_candidates[a][0]
            return g if g is not None else float("inf")

        best_alpha = min(alpha_candidates, key=_key)
        best_gap, best_feasprob = alpha_candidates[best_alpha]

        summary = {
            "family": family,
            "n": n,
            "instance": instance_meta[(family, n)],
            "expectation_qaoa_median_gap": exp_gap,
            "expectation_qaoa_median_feasible_probability": exp_feasprob,
            "cvar_qaoa_best_alpha": best_alpha,
            "cvar_qaoa_best_alpha_median_gap": best_gap,
            "cvar_qaoa_best_alpha_median_feasible_probability": best_feasprob,
            "alpha_0.5_median_gap": alpha_candidates[0.5][0],
            "alpha_0.2_median_gap": alpha_candidates[0.2][0],
        }
        cell_summaries.append(summary)

        if exp_gap is None or best_gap is None:
            excluded_cells.append({"family": family, "n": n, "reason": "objective_gap undefined (opt objective ~= 0)"})
            continue

        expectation_gaps.append(exp_gap)
        cvar_best_gaps.append(best_gap)
        cvar_best_alphas.append(best_alpha)
        feasible_majority_flags.append(best_feasprob is not None and best_feasprob >= 0.5)
        feasible_negligible_flags.append(
            best_feasprob is None or best_feasprob < NEGLIGIBLE_FEASIBLE_PROBABILITY_THRESHOLD
        )

    # ------------------------------------------------------------------
    # Preregistered statistical test: two-sided Wilcoxon signed-rank,
    # paired by (family, n) cell, CVaR-QAOA(best alpha) vs expectation-QAOA
    # median objective_gap.
    # ------------------------------------------------------------------
    n_paired = len(expectation_gaps)
    wilcoxon_result = {"n_pairs": n_paired}
    try:
        diffs = np.array(cvar_best_gaps) - np.array(expectation_gaps)
        if n_paired >= 1 and np.any(diffs != 0):
            stat, pvalue = wilcoxon(cvar_best_gaps, expectation_gaps, alternative="two-sided")
            wilcoxon_result.update(
                {"statistic": float(stat), "pvalue": float(pvalue), "computed": True}
            )
        else:
            wilcoxon_result.update(
                {
                    "statistic": None,
                    "pvalue": None,
                    "computed": False,
                    "reason": "all paired differences are zero or fewer than 1 pair",
                }
            )
    except Exception as e:  # pragma: no cover -- defensive, reported not swallowed
        wilcoxon_result.update({"statistic": None, "pvalue": None, "computed": False, "reason": str(e)})

    cvar_median_of_medians = median(cvar_best_gaps) if cvar_best_gaps else None
    expectation_median_of_medians = median(expectation_gaps) if expectation_gaps else None
    n_majority_feasible = sum(1 for f in feasible_majority_flags if f)
    majority_feasible = (
        n_majority_feasible > (len(feasible_majority_flags) / 2.0) if feasible_majority_flags else False
    )
    n_negligible_feasible = sum(1 for f in feasible_negligible_flags if f)
    majority_feasible_negligible = (
        n_negligible_feasible > (len(feasible_negligible_flags) / 2.0) if feasible_negligible_flags else False
    )
    cvar_improves_or_ties = (
        cvar_median_of_medians is not None
        and expectation_median_of_medians is not None
        and cvar_median_of_medians <= expectation_median_of_medians
    )
    significant = (
        wilcoxon_result.get("computed")
        and wilcoxon_result.get("pvalue") is not None
        and wilcoxon_result["pvalue"] < 0.05
    )

    classification = classify_cvar_qaoa_result(
        cvar_improves_or_ties=cvar_improves_or_ties,
        significant=significant,
        majority_feasible=majority_feasible,
        majority_feasible_negligible=majority_feasible_negligible,
    )

    # Fraction of individual (family,n) cells where CVaR(best alpha) beat
    # or tied expectation -- "works only on a subset of families" evidence
    # for the RESEARCH_ONLY branch.
    n_cells_cvar_better_or_tie = sum(
        1 for c, e in zip(cvar_best_gaps, expectation_gaps) if c <= e
    )
    fraction_cells_cvar_better_or_tie = (
        n_cells_cvar_better_or_tie / n_paired if n_paired else None
    )

    result, computed_evidence_level = _make_experiment_result(
        name="EXP-B1",
        algorithm="cvar_qaoa_vs_expectation_qaoa",
        seed=0,
        metrics={
            "n_cells_total": float(len(cells)),
            "n_paired_family_n_cells": float(n_paired),
            "cvar_best_alpha_median_of_medians_gap": (
                cvar_median_of_medians if cvar_median_of_medians is not None else float("nan")
            ),
            "expectation_median_of_medians_gap": (
                expectation_median_of_medians if expectation_median_of_medians is not None else float("nan")
            ),
            "wilcoxon_pvalue": (
                wilcoxon_result["pvalue"] if wilcoxon_result.get("pvalue") is not None else float("nan")
            ),
            "fraction_family_n_cells_majority_feasible": (
                n_majority_feasible / len(feasible_majority_flags) if feasible_majority_flags else float("nan")
            ),
            "fraction_cells_cvar_better_or_tie": (
                fraction_cells_cvar_better_or_tie if fraction_cells_cvar_better_or_tie is not None else float("nan")
            ),
            "total_runtime_s": total_runtime,
        },
        notes=(
            f"EXP-B1: classification={classification}, "
            f"wilcoxon_p={wilcoxon_result.get('pvalue')}, "
            f"cvar_best_median_gap={cvar_median_of_medians}, "
            f"expectation_median_gap={expectation_median_of_medians}. "
            "No claim of quantum advantage -- compares two QAOA variants "
            "against the classical optimum only."
        ),
    )

    manifest = {
        "experiment_id": "EXP-B1",
        "hypothesis": exp.hypothesis,
        "families": families,
        "sizes": sizes,
        "p_values": p_values,
        "alphas": alphas,
        "seeds": seeds,
        "maxiter": MAXITER,
        "n_cells_total": len(cells),
        "calibration": calibration,
        "preregistration_deviations": [
            {
                "deviation": "n ladder reduced from [4,6,8,10,12,14] to [4,6,8,10]",
                "reason": (
                    "Already disclosed in QUANTUM1B_PREREGISTRATION.md 'Preregistration "
                    "deviations' #1, discovered during qubo_encoding.py implementation "
                    "before any QAOA run -- the exact slack-bit QUBO encoding needs more "
                    "qubits per candidate than a penalty-only encoding would (n=14 -> up "
                    "to 23-24 qubits, beyond the ~16-18 qubit ceiling this statevector "
                    "engine and QiskitQAOASolver's own documented n<=14 convention target)."
                ),
                "consequence_for_this_script": (
                    "The Wilcoxon signed-rank test below runs across 24 (family,n) cells "
                    "(6 families x 4 sizes), not the 36 the original pass/fail wording "
                    "(EXP-B1.json/PREREGISTRATION.md) assumed under the original [4..14] "
                    "ladder -- a direct arithmetic consequence of the already-disclosed "
                    "ladder reduction, not a new deviation."
                ),
            },
            {
                "deviation": "none -- full preregistered p/alpha/seed/family sweep run as specified",
                "reason": (
                    f"Calibration (see 'calibration' key above) extrapolated the full "
                    f"{len(families)*len(sizes)*len(p_values)*len(alphas)*len(seeds)}-run "
                    f"sweep at ~{calibration['extrapolated_total_s']:.0f}s, far under the "
                    "4-hour preregistered budget, so no seed/family/p reduction was applied."
                ),
            },
        ],
        "excluded_cells": excluded_cells,
        "cell_summaries": cell_summaries,
        "wilcoxon_test": wilcoxon_result,
        "classification": classification,
        "classification_inputs": {
            "cvar_best_alpha_median_of_medians_gap": cvar_median_of_medians,
            "expectation_median_of_medians_gap": expectation_median_of_medians,
            "cvar_improves_or_ties": cvar_improves_or_ties,
            "wilcoxon_significant_p_lt_0.05": significant,
            "majority_cells_feasible_probability_ge_0.5": majority_feasible,
            "n_majority_feasible_cells": n_majority_feasible,
            "n_cells_considered_for_majority": len(feasible_majority_flags),
            "fraction_cells_cvar_better_or_tie": fraction_cells_cvar_better_or_tie,
            "majority_cells_feasible_probability_negligible": majority_feasible_negligible,
            "negligible_feasible_probability_threshold": NEGLIGIBLE_FEASIBLE_PROBABILITY_THRESHOLD,
            "n_negligible_feasible_cells": n_negligible_feasible,
        },
        "prohibited_conclusion": exp.prohibited_conclusion,
        "total_runtime_s": total_runtime,
        "compute_budget_s": COMPUTE_BUDGET_S,
        "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
        "preregistered_evidence_level": exp.evidence_level,
        "computed_evidence_level_via_core_evidence": computed_evidence_level,
        "experiment_result": json.loads(result.model_dump_json()),
        "cells": cells,
    }
    return manifest


def main() -> dict:
    registry = load_registry()
    print("Running EXP-B1 (expectation-QAOA vs CVaR-QAOA objective gap)...")
    b1 = run_exp_b1(registry)
    print(
        f"EXP-B1: classification={b1['classification']}, "
        f"wilcoxon_p={b1['wilcoxon_test'].get('pvalue')}, "
        f"runtime={b1['total_runtime_s']:.1f}s"
    )

    full_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "EXP-B1": b1,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"Wrote results manifest to {RESULTS_PATH}")

    return full_manifest


if __name__ == "__main__":
    main()
