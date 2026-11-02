"""Track D runner: EXP-D1 -- FakeBackend noise-aware CVaR-QAOA-shaped
circuit validation (hardware-readiness, not Track B science -- see
`experiments/quantum1b/noise_validation.py`'s module docstring for why
these are separate).

**Host-resource discipline (same convention as `run_track_b_exp_b3.py`):**
BLAS thread counts pinned before numpy/scipy import; run as the only heavy
job at a time; one small instance only. **Corrected 2026-10-01, remediating
independent-review finding P3-12:** this docstring previously stated a
PRE-RUN calibration ESTIMATE ("~7s per COBYLA evaluation... ~123s total")
that was never reconciled against the ACTUAL measured result after the run
completed -- the real result (`results/track_d_exp_d1.json`) shows 15/15
evaluations at 4.63s/eval, 69.4s total fake_kingston runtime, ~44% lower
than the pre-run estimate. No practical consequence either way (ample
budget margin both pre- and post-measurement) -- see
`docs/quantum/QUANTUM1B_HARDWARE_READINESS.md` for the full measured
record.

Run with: `python3 -m experiments.quantum1b.run_track_d_exp_d1` from the
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

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from core.evidence import compute_evidence_level  # noqa: E402
from core.schemas import (  # noqa: E402
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from experiments.quantum1b.generators import generate_instance  # noqa: E402
from experiments.quantum1b.qubo_encoding import build_qubo  # noqa: E402
from experiments.quantum1b.noise_validation import (  # noqa: E402
    fake_kingston_backend,
    ideal_backend,
    run_noise_validation_cell,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "track_d_exp_d1.json"

FAMILY = "independent"
N_CANDIDATES = 3
SEED = 0
P = 1
MAXITER = 15
SHOTS = 1024
COMPUTE_BUDGET_S = 900.0  # 15 minutes -- generous vs the ~150s calibrated estimate


def _result_to_dict(r) -> dict:
    d = asdict(r)
    d["best_params"] = list(map(float, r.best_params))
    return d


def main() -> dict:
    t0 = time.perf_counter()
    print("Running EXP-D1 (FakeBackend noise-aware CVaR-QAOA-shaped circuit validation)...")

    inst = generate_instance(FAMILY, N_CANDIDATES, seed=SEED)
    qubo = build_qubo(inst)
    print(f"instance: family={FAMILY} n_candidates={N_CANDIDATES} n_qubits={qubo.n_qubits()}")

    print("running ideal (bare AerSimulator) cell...")
    ideal_result = run_noise_validation_cell(
        inst, qubo, backend_label="ideal", backend=ideal_backend(), p=P, maxiter=MAXITER, shots=SHOTS, seed=SEED
    )
    print(
        f"  ideal: depth={ideal_result.depth} 2q_gates={ideal_result.two_qubit_gates} "
        f"feasible_fraction={ideal_result.feasible_fraction:.3f} gap={ideal_result.objective_gap} "
        f"runtime={ideal_result.runtime_s:.1f}s"
    )

    print("building FakeKingston-derived noisy simulator (one-time ~11s)...")
    fk_backend = fake_kingston_backend()
    print("running fake_kingston (noise + real device coupling map) cell...")
    fk_result = run_noise_validation_cell(
        inst, qubo, backend_label="fake_kingston", backend=fk_backend, p=P, maxiter=MAXITER, shots=SHOTS, seed=SEED
    )
    print(
        f"  fake_kingston: depth={fk_result.depth} 2q_gates={fk_result.two_qubit_gates} "
        f"feasible_fraction={fk_result.feasible_fraction:.3f} gap={fk_result.objective_gap} "
        f"runtime={fk_result.runtime_s:.1f}s"
    )

    total_runtime = time.perf_counter() - t0

    ideal_er = ExperimentResult(
        metadata=RunMetadata(algorithm="qaoa_noise_validation", seed=SEED),
        config=ExperimentConfig(
            name="EXP-D1-ideal",
            algorithm="qaoa_noise_validation",
            application="quantum1b_track_d",
            backend=BackendDescriptor(name="aer_simulator_ideal", kind=BackendKind.SIMULATOR, num_qubits=qubo.n_qubits()),
            shots=SHOTS,
            seed=SEED,
        ),
        metrics={
            "objective_gap": ideal_result.objective_gap if ideal_result.objective_gap is not None else -1.0,
            "feasible_fraction": ideal_result.feasible_fraction,
            "circuit_depth": float(ideal_result.depth),
            "two_qubit_gates": float(ideal_result.two_qubit_gates),
        },
        financial_edge_claimed=False,
        notes="EXP-D1 ideal-backend cell -- hardware-readiness engineering validation, not Track B science.",
    )
    fk_er = ExperimentResult(
        metadata=RunMetadata(algorithm="qaoa_noise_validation", seed=SEED),
        config=ExperimentConfig(
            name="EXP-D1-fake_kingston",
            algorithm="qaoa_noise_validation",
            application="quantum1b_track_d",
            backend=BackendDescriptor(
                name="FakeKingston", kind=BackendKind.FAKE, num_qubits=156, calibration_source="fake_backend:FakeKingston"
            ),
            shots=SHOTS,
            seed=SEED,
        ),
        metrics={
            "objective_gap": fk_result.objective_gap if fk_result.objective_gap is not None else -1.0,
            "feasible_fraction": fk_result.feasible_fraction,
            "circuit_depth": float(fk_result.depth),
            "two_qubit_gates": float(fk_result.two_qubit_gates),
        },
        financial_edge_claimed=False,
        notes="EXP-D1 FakeKingston (device-calibrated noise + real coupling map) cell.",
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "EXP-D1": {
            "instance_family": FAMILY,
            "instance_n_candidates": N_CANDIDATES,
            "instance_fingerprint": inst.fingerprint(),
            "qubo_n_qubits": qubo.n_qubits(),
            "p": P,
            "maxiter": MAXITER,
            "shots": SHOTS,
            "seed": SEED,
            "ideal": _result_to_dict(ideal_result),
            "fake_kingston": _result_to_dict(fk_result),
            "ideal_evidence_level": compute_evidence_level(ideal_er).value,
            "fake_kingston_evidence_level": compute_evidence_level(fk_er).value,
            "depth_ratio_fake_over_ideal": (
                fk_result.depth / ideal_result.depth if ideal_result.depth else None
            ),
            "total_runtime_s": total_runtime,
            "compute_budget_s": COMPUTE_BUDGET_S,
            "within_compute_budget": total_runtime < COMPUTE_BUDGET_S,
            "ideal_experiment_result": json.loads(ideal_er.model_dump_json()),
            "fake_kingston_experiment_result": json.loads(fk_er.model_dump_json()),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"EXP-D1 complete in {total_runtime:.1f}s. Wrote results manifest to {RESULTS_PATH}")

    return manifest


if __name__ == "__main__":
    main()
