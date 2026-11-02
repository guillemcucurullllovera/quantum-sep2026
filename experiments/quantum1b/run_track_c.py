"""EXP-C1 runner -- QRC vs ESN fair-comparison harness on the synthetic
`synthetic_realized_vol_proxy` task.

Executes the grid documented in `qrc_esn_harness.py` (readout_dims,
seeds, alpha_grid as specified by `QUANTUM1B_PREREGISTRATION.md` /
`experiments/quantum1b/registry/EXP-C1.json`, with the one logged
readout_dims deviation -- see `qrc_esn_harness`'s module docstring for
why `[4, 8, 16]` became `[4, 6, 8]`) and writes
`experiments/quantum1b/results/track_c_results.json`.

This script never touches real financial data. `financial_edge_claimed`
is always `False`. Real-financial-task execution is a separate, explicit
BLOCKED guard -- see `qrc_esn_harness.run_real_financial_task`.

Run with: `python3 -m experiments.quantum1b.run_track_c` from the
worktree root.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from core.evidence import compute_evidence_level
from core.schemas import BackendDescriptor, BackendKind, ExperimentConfig, ExperimentResult, RunMetadata
from experiments.quantum1b.qrc_esn_harness import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_READOUT_DIMS,
    DEFAULT_SEEDS,
    DEFAULT_SERIES_LEN,
    DEFAULT_TRAIN_FRAC,
    compare_qrc_vs_esn,
    paired_significance,
    serialize_comparison,
)

RESULTS_PATH = Path(__file__).parent / "results" / "track_c_results.json"

PREREGISTRATION_DEVIATIONS = [
    {
        "field": "readout_dims",
        "preregistered": [4, 8, 16],
        "actual": list(DEFAULT_READOUT_DIMS),
        "reason": (
            "D=16 requires a 16-total-qubit QuantumReservoir full density-"
            "matrix simulation (2**16 x 2**16 complex128, ~68.7GB for rho "
            "alone) -- a hard memory/compute infeasibility on any machine "
            "available to this task, not a budget-trimming choice. D=6 "
            "inserted to keep the preregistered 3-point cardinality while "
            "keeping D=4 and D=8 from the original grid. See "
            "qrc_esn_harness.py module docstring and "
            "docs/quantum/QUANTUM1B_QRC_ESN.md for the empirical timing "
            "table that motivated this before any EXP-C1 result existed."
        ),
    },
    {
        "field": "readout_dimension_matching_methodology",
        "preregistered": (
            "QRC's n2 hidden qubits set so 4^n2 >= D readout features "
            "project down to D via the same ridge head dimensionality"
        ),
        "actual": (
            "D = n1 + n2 (total qubit count), with n1=1 fixed and "
            "n2=D-1. QuantumReservoir's public sequence_features/"
            "windowed_features API returns only n1+n2 Z-expectation "
            "scalars per timestep, never the full 4^n2-dimensional "
            "rho_hidden (confirmed by reading reservoir.py/"
            "reservoir_math.py and cross-checked against the Q10 audit's "
            "explicit finding that 'readout touches only diagonal "
            "populations' despite rho_hidden's internal 4^n2 dimension). "
            "No public API exists to extract a 4^n2-sized readout, and "
            "reservoir.py is not modified to add one."
        ),
        "reason": "Discovered while reading the actual class, before running EXP-C1, per the task's own instruction to verify rather than assume.",
    },
]


def main() -> dict:
    t0 = time.time()
    comparison = compare_qrc_vs_esn(
        readout_dims=DEFAULT_READOUT_DIMS,
        seeds=DEFAULT_SEEDS,
        alpha_grid=DEFAULT_ALPHA_GRID,
        n_points=DEFAULT_SERIES_LEN,
        train_frac=DEFAULT_TRAIN_FRAC,
    )
    wall_time = time.time() - t0

    significance_by_D = {
        str(D): paired_significance(comparison["paired_esn_minus_qrc_mse"][D]) for D in comparison["readout_dims"]
    }

    metrics: dict[str, float] = {"wall_time_s": wall_time}
    for D in comparison["readout_dims"]:
        metrics[f"esn_mean_oos_mse_D{D}"] = float(np.mean([r.oos_mse for r in comparison["esn"][D]]))
        metrics[f"esn_std_oos_mse_D{D}"] = float(np.std([r.oos_mse for r in comparison["esn"][D]]))
        metrics[f"qrc_mean_oos_mse_D{D}"] = float(np.mean([r.oos_mse for r in comparison["qrc"][D]]))
        metrics[f"qrc_std_oos_mse_D{D}"] = float(np.std([r.oos_mse for r in comparison["qrc"][D]]))
        metrics[f"esn_mean_memory_capacity_D{D}"] = float(
            np.mean([r.extra["memory_capacity"]["total"] for r in comparison["esn"][D]])
        )
        metrics[f"qrc_mean_memory_capacity_D{D}"] = float(
            np.mean([r.extra["memory_capacity"]["total"] for r in comparison["qrc"][D]])
        )
    metrics["persistence_mean_oos_mse"] = float(np.mean([r.oos_mse for r in comparison["persistence"]]))
    metrics["ridge_raw_mean_oos_mse"] = float(np.mean([r.oos_mse for r in comparison["ridge_raw"]]))

    backend = BackendDescriptor(
        name="numpy_dense_reservoir_and_classical",
        kind=BackendKind.SIMULATOR,
        num_qubits=max(DEFAULT_READOUT_DIMS),
    )
    exp_config = ExperimentConfig(
        name="EXP-C1_qrc_vs_esn_synthetic",
        algorithm="qrc_vs_esn_harness",
        application="track_c_synthetic_vol",
        backend=backend,
        shots=None,
        seed=DEFAULT_SEEDS[0],
        params={
            "readout_dims": list(DEFAULT_READOUT_DIMS),
            "seeds": list(DEFAULT_SEEDS),
            "alpha_grid": list(DEFAULT_ALPHA_GRID),
            "n_points": DEFAULT_SERIES_LEN,
            "train_frac": DEFAULT_TRAIN_FRAC,
        },
    )
    metadata = RunMetadata(algorithm="qrc_vs_esn_harness", generation=None, seed=DEFAULT_SEEDS[0])

    classical_baseline_metrics = {
        "persistence_mean_oos_mse": metrics["persistence_mean_oos_mse"],
        "ridge_raw_mean_oos_mse": metrics["ridge_raw_mean_oos_mse"],
        **{f"esn_mean_oos_mse_D{D}": metrics[f"esn_mean_oos_mse_D{D}"] for D in comparison["readout_dims"]},
    }

    result = ExperimentResult(
        metadata=metadata,
        config=exp_config,
        metrics=metrics,
        classical_baseline_metrics=classical_baseline_metrics,
        financial_edge_claimed=False,
        notes=(
            "EXP-C1: synthetic-task-only QRC-vs-ESN fair comparison. "
            "financial_edge_claimed is always False here. RESEARCH_ONLY is "
            "the maximum reachable classification regardless of outcome -- "
            "see docs/quantum/QUANTUM1B_QRC_ESN.md. Real-financial-task "
            "execution is explicitly BLOCKED, see "
            "qrc_esn_harness.run_real_financial_task."
        ),
    )
    evidence_level = compute_evidence_level(result)

    output = {
        "experiment_id": "EXP-C1",
        "evidence_level": evidence_level.value,
        "financial_edge_claimed": result.financial_edge_claimed,
        "wall_time_s": wall_time,
        "preregistration_deviations": PREREGISTRATION_DEVIATIONS,
        "config": json.loads(exp_config.model_dump_json()),
        "metadata": json.loads(metadata.model_dump_json()),
        "metrics": metrics,
        "significance_esn_vs_qrc_by_readout_dim": significance_by_D,
        "comparison": serialize_comparison(comparison),
        "real_financial_execution_status": "BLOCKED",
        "real_financial_execution_reason": (
            "See experiments/quantum1b/qrc_esn_harness.RealFinancialExecutionBlocked "
            "and run_real_financial_task(); no licensed OHLCV for "
            "SPY/QQQ/TLT/IEF/GLD/USO with confirmed research rights, per "
            "Model Lab readiness gate D4 ruling "
            "(research-corpus/model-lab/work/readiness/READINESS_BRIEF.md)."
        ),
        "research_only_ceiling": (
            "Per EXP-C1's pass/fail rule, RESEARCH_ONLY is the maximum "
            "reachable classification from this phase's results, "
            "regardless of outcome -- a structural ceiling, not an "
            "outcome of the measurement, since no real financial-task "
            "evidence exists."
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2, sort_keys=False))
    return output


if __name__ == "__main__":
    out = main()
    summary = {k: v for k, v in out.items() if k != "comparison"}
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")
