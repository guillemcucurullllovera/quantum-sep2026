"""Quantum Lab end-to-end prototype: classical -> QUBO -> QAOA/CVaR-QAOA ->
noise-validated simulator -> comparison report, on the candidate-trade /
finite derivative-risk-budget combinatorial optimization problem
(`OptionSelectionInstance` -- Track A/B's existing AMC-shaped discrete
option-structure selection problem).

**Why this instance, and why no new science.** This assembles PIECES THAT
ALREADY EXIST (Track A's `classical_exact`, Track B's `cvar_qaoa`, Track
D's `noise_validation`) into ONE coherent, runnable, demonstrable
end-to-end pipeline on a single frozen instance -- it deliberately does
NOT run a new sweep, does NOT claim a new finding, and reuses
`FROZEN_CONTROLS[0]` (family="independent", n_candidates=4, seed=0) exactly
as `hardware_readiness.frozen_controls` already pins it, so this prototype
is anchored to the SAME instance the Hardware Day runbook would submit
first, not a bespoke one invented for this script.

**What this IS:** a working demonstration that the full pipeline runs
end-to-end on synthetic/frozen data, entirely on simulators, with the
classical exact solver as the benchmark every quantum stage is measured
against -- exactly what the Command Center needs to show as "the Quantum
Lab's flagship demonstrable experiment."

**What this is NOT:** a new preregistered experiment, a quantum-advantage
claim, or a financial-edge claim. `financial_edge_claimed=False`
throughout. No real IBM hardware, no real credentials used anywhere in
this module.

Run with: `python3 -m experiments.quantum1b.quantum_lab.candidate_trade_prototype`
from the repo root.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import json  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from core.evidence import compute_evidence_level  # noqa: E402
from core.schemas import (  # noqa: E402
    BackendDescriptor,
    BackendKind,
    ExperimentConfig,
    ExperimentResult,
    RunMetadata,
)
from experiments.quantum1b.classical_exact import solve_exact_milp  # noqa: E402
from experiments.quantum1b.cvar_qaoa import (  # noqa: E402
    exact_distribution_metrics,
    run_cvar_qaoa,
    run_expectation_qaoa,
)
from experiments.quantum1b.generators import generate_instance  # noqa: E402
from experiments.quantum1b.hardware_readiness.frozen_controls import (  # noqa: E402
    FROZEN_CONTROL_SET,
    SIMULATOR_BASELINE_ALPHA,
    SIMULATOR_BASELINE_MAXITER,
    SIMULATOR_BASELINE_P,
    SIMULATOR_BASELINE_SEED,
)
from experiments.quantum1b.noise_validation import (  # noqa: E402
    fake_kingston_backend,
    ideal_backend,
    run_noise_validation_cell,
)
from experiments.quantum1b.qubo_encoding import build_qubo  # noqa: E402

OUTPUT_PATH = Path(__file__).parents[1] / "results" / "quantum_lab_candidate_trade_e2e.json"

# Same convention run_track_a.py uses for classical-only results: no real
# quantum backend exists for this stage, so a SIMULATOR-kind placeholder
# with num_qubits=0 is used rather than making `backend` optional (the
# schema requires it).
_CLASSICAL_BACKEND = BackendDescriptor(name="classical_highs", kind=BackendKind.SIMULATOR, num_qubits=0)

# EXP-D1's own noise-validation convention (p=1, maxiter=15, shots=1024,
# seed=0) -- reused verbatim, not re-chosen, so this prototype's noisy
# stage is directly comparable to the existing Track D record.
NOISE_VALIDATION_P = 1
NOISE_VALIDATION_MAXITER = 15
NOISE_VALIDATION_SHOTS = 1024
NOISE_VALIDATION_SEED = 0


@dataclass(frozen=True)
class CandidateTradeE2EReport:
    generated_at: str
    instance_family: str
    instance_n_candidates: int
    instance_seed: int
    instance_fingerprint: str
    n_qubits: int

    classical_exact_optimum: float
    classical_evidence_level: str

    ideal_expectation_qaoa_best_feasible_objective: float | None
    ideal_expectation_qaoa_evidence_level: str
    ideal_cvar_qaoa_best_feasible_objective: float | None
    ideal_cvar_qaoa_evidence_level: str

    noisy_expectation_qaoa_best_feasible_objective: float | None
    noisy_expectation_qaoa_depth: int
    noisy_expectation_qaoa_two_qubit_gates: int
    noisy_expectation_qaoa_depth_inflation_ratio: float | None
    noisy_expectation_qaoa_evidence_level: str

    classical_vs_ideal_expectation_gap: float | None
    classical_vs_ideal_cvar_gap: float | None
    classical_vs_noisy_gap: float | None

    financial_edge_claimed: bool
    quantum_advantage_claimed: bool
    conclusion: str
    limitations: str


def run(*, n_candidates: int = 4, family: str = "independent", seed: int = 0) -> CandidateTradeE2EReport:
    """Runs the full pipeline once, on a small (default: `FROZEN_CONTROLS[0]`-
    matching) instance. Cheap enough to run on every call -- no caching, no
    hidden state -- see `FROZEN_CONTROL_SET` for why these particular
    defaults were chosen (the same instance Hardware Day would submit
    first)."""
    assert (family, n_candidates, seed) in {
        (c.family, c.n_candidates, c.seed) for c in FROZEN_CONTROL_SET
    }, "candidate_trade_prototype's default instance must stay aligned with FROZEN_CONTROL_SET"

    # 1. Classical -> QUBO.
    instance = generate_instance(family, n_candidates, seed=seed)
    qubo = build_qubo(instance)

    # 2. Classical exact benchmark (Track A).
    exact = solve_exact_milp(instance)
    classical_er = ExperimentResult(
        metadata=RunMetadata(algorithm="exact_milp", seed=seed),
        config=ExperimentConfig(
            name="quantum-lab-e2e-classical", algorithm="exact_milp",
            application="quantum1b_quantum_lab", backend=_CLASSICAL_BACKEND, shots=None, seed=seed,
        ),
        metrics={"objective": exact.objective},
        financial_edge_claimed=False,
        notes="Quantum Lab end-to-end prototype's classical benchmark stage (Track A's solve_exact_milp).",
    )

    # 3. Ideal QAOA / CVaR-QAOA (Track B's own EXP-B1 convention).
    exp_result = run_expectation_qaoa(
        qubo, p=SIMULATOR_BASELINE_P, seed=SIMULATOR_BASELINE_SEED, maxiter=SIMULATOR_BASELINE_MAXITER
    )
    exp_metrics = exact_distribution_metrics(exp_result, qubo, instance)
    cvar_result = run_cvar_qaoa(
        qubo, p=SIMULATOR_BASELINE_P, alpha=SIMULATOR_BASELINE_ALPHA,
        seed=SIMULATOR_BASELINE_SEED, maxiter=SIMULATOR_BASELINE_MAXITER,
    )
    cvar_metrics = exact_distribution_metrics(cvar_result, qubo, instance, cvar_alpha=SIMULATOR_BASELINE_ALPHA)

    ideal_exp_er = ExperimentResult(
        metadata=RunMetadata(algorithm="expectation_qaoa", seed=SIMULATOR_BASELINE_SEED),
        config=ExperimentConfig(
            name="quantum-lab-e2e-ideal-expectation-qaoa", algorithm="expectation_qaoa",
            application="quantum1b_quantum_lab",
            backend=BackendDescriptor(name="numpy_statevector_ideal", kind=BackendKind.SIMULATOR, num_qubits=qubo.n_qubits()),
            shots=None, seed=SIMULATOR_BASELINE_SEED,
        ),
        metrics={"best_feasible_objective": exp_metrics.best_feasible_objective or float("nan")},
        financial_edge_claimed=False,
        notes="Quantum Lab end-to-end prototype's ideal expectation-QAOA stage (Track B's cvar_qaoa.run_expectation_qaoa).",
    )
    ideal_cvar_er = ExperimentResult(
        metadata=RunMetadata(algorithm="cvar_qaoa", seed=SIMULATOR_BASELINE_SEED),
        config=ExperimentConfig(
            name="quantum-lab-e2e-ideal-cvar-qaoa", algorithm="cvar_qaoa",
            application="quantum1b_quantum_lab",
            backend=BackendDescriptor(name="numpy_statevector_ideal", kind=BackendKind.SIMULATOR, num_qubits=qubo.n_qubits()),
            shots=None, seed=SIMULATOR_BASELINE_SEED,
        ),
        metrics={"best_feasible_objective": cvar_metrics.best_feasible_objective or float("nan")},
        financial_edge_claimed=False,
        notes="Quantum Lab end-to-end prototype's ideal CVaR-QAOA stage (Track B's cvar_qaoa.run_cvar_qaoa).",
    )

    # 4. Noise-validated (Track D's own EXP-D1 convention) -- demonstrates
    # the pipeline runs end-to-end on a real-device-calibrated noise model,
    # not just an ideal simulator.
    ideal_backend_result = run_noise_validation_cell(
        instance, qubo, backend_label="ideal", backend=ideal_backend(),
        p=NOISE_VALIDATION_P, maxiter=NOISE_VALIDATION_MAXITER, shots=NOISE_VALIDATION_SHOTS,
        seed=NOISE_VALIDATION_SEED,
    )
    noisy_result = run_noise_validation_cell(
        instance, qubo, backend_label="fake_kingston", backend=fake_kingston_backend(),
        p=NOISE_VALIDATION_P, maxiter=NOISE_VALIDATION_MAXITER, shots=NOISE_VALIDATION_SHOTS,
        seed=NOISE_VALIDATION_SEED,
    )
    noisy_er = ExperimentResult(
        metadata=RunMetadata(algorithm="qaoa_noise_validation", seed=NOISE_VALIDATION_SEED),
        config=ExperimentConfig(
            name="quantum-lab-e2e-noisy-expectation-qaoa", algorithm="qaoa_noise_validation",
            application="quantum1b_quantum_lab",
            backend=BackendDescriptor(
                name="FakeKingston", kind=BackendKind.FAKE, num_qubits=156, calibration_source="fake_backend:FakeKingston"
            ),
            shots=NOISE_VALIDATION_SHOTS, seed=NOISE_VALIDATION_SEED,
        ),
        metrics={
            "best_feasible_objective": noisy_result.best_feasible_objective or float("nan"),
            "circuit_depth": float(noisy_result.depth),
            "two_qubit_gates": float(noisy_result.two_qubit_gates),
        },
        financial_edge_claimed=False,
        notes="Quantum Lab end-to-end prototype's FakeKingston noise-validated stage (Track D's noise_validation.run_noise_validation_cell).",
    )

    def _gap(quantum_val: float | None) -> float | None:
        if quantum_val is None or exact.objective == 0:
            return None
        return (exact.objective - quantum_val) / abs(exact.objective)

    depth_ratio = (
        noisy_result.depth / ideal_backend_result.depth if ideal_backend_result.depth else None
    )

    return CandidateTradeE2EReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        instance_family=family,
        instance_n_candidates=n_candidates,
        instance_seed=seed,
        instance_fingerprint=instance.fingerprint(),
        n_qubits=qubo.n_qubits(),
        classical_exact_optimum=exact.objective,
        classical_evidence_level=compute_evidence_level(classical_er).value,
        ideal_expectation_qaoa_best_feasible_objective=exp_metrics.best_feasible_objective,
        ideal_expectation_qaoa_evidence_level=compute_evidence_level(ideal_exp_er).value,
        ideal_cvar_qaoa_best_feasible_objective=cvar_metrics.best_feasible_objective,
        ideal_cvar_qaoa_evidence_level=compute_evidence_level(ideal_cvar_er).value,
        noisy_expectation_qaoa_best_feasible_objective=noisy_result.best_feasible_objective,
        noisy_expectation_qaoa_depth=noisy_result.depth,
        noisy_expectation_qaoa_two_qubit_gates=noisy_result.two_qubit_gates,
        noisy_expectation_qaoa_depth_inflation_ratio=depth_ratio,
        noisy_expectation_qaoa_evidence_level=compute_evidence_level(noisy_er).value,
        classical_vs_ideal_expectation_gap=_gap(exp_metrics.best_feasible_objective),
        classical_vs_ideal_cvar_gap=_gap(cvar_metrics.best_feasible_objective),
        classical_vs_noisy_gap=_gap(noisy_result.best_feasible_objective),
        financial_edge_claimed=False,
        quantum_advantage_claimed=False,
        conclusion=(
            "Demonstrates the full classical->QUBO->QAOA/CVaR-QAOA->noise-validated-simulator pipeline runs "
            "end-to-end on one frozen instance, with the classical exact solver as the benchmark at every stage. "
            "This is an engineering/pipeline demonstration, consistent with and reusing Track A/B/D's own prior "
            "findings on this exact instance family -- it is not a new scientific result and makes no new claim "
            "about CVaR-QAOA's performance beyond what EXP-B1/EXP-D1 already established."
        ),
        limitations=(
            "Single tiny instance (4 candidates, 9 qubits), single seed, no statistical replication -- this is a "
            "pipeline demonstration, not a sized research sweep (see EXP-B1/EXP-D1 for the actual sized studies "
            "this reuses). The noisy stage uses its own independent COBYLA run (seed=0, maxiter=15, matching "
            "EXP-D1's convention) rather than reusing EXP-D1's own cached result, so small numeric differences "
            "from EXP-D1's own published numbers are expected and not a discrepancy."
        ),
    )


def main() -> None:
    report = run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"classical={report.classical_exact_optimum} "
        f"ideal_exp_qaoa={report.ideal_expectation_qaoa_best_feasible_objective} "
        f"ideal_cvar_qaoa={report.ideal_cvar_qaoa_best_feasible_objective} "
        f"noisy_qaoa={report.noisy_expectation_qaoa_best_feasible_objective}"
    )


if __name__ == "__main__":
    main()


__all__ = ["CandidateTradeE2EReport", "run", "main", "OUTPUT_PATH"]
