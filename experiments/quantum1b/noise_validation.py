"""Track D (hardware-readiness), EXP-D1 -- FakeBackend noise-aware
CVaR-QAOA-shaped circuit validation.

**Why this exists and why it is separate from Track B.** QUANTUM1B_PLAN.md
section 3 deliberately kept Track B (EXP-B1/B2/B3) at evidence ceiling B/C
("No FakeBackend/noise-model tier in 1B ... explicitly QUANTUM-1C-scale
work per the roadmap sequencing") to protect that phase's compute budget.
This module is genuinely separate, later work: the overnight hardware-
readiness pass explicitly asked for FakeBackend/Aer noise-aware validation
as part of assessing Hardware Day readiness -- an ENGINEERING question
("does the real-circuit-plus-real-device-noise pipeline work end to end,
and what does it look like") distinct from Track B's SCIENTIFIC question
("does CVaR-QAOA beat expectation-QAOA"). This module answers the former,
on one small instance, not a full preregistered sweep -- see
`docs/quantum/QUANTUM1B_HARDWARE_READINESS.md` for why this is
deliberately narrow in scope.

**What it reuses (reconcile, not rebuild):**
- `algorithms.qaoa_qubo.solvers._build_qiskit_qaoa_circuit` (via
  `hardware_readiness.lineage.build_qaoa_circuit_for_lineage`) for the
  actual circuit.
- `algorithms.qae_iqae.backends.kingston_backend`/`kingston_sim` for the
  FakeKingston device-calibration snapshot and its Aer noise-model-plus-
  coupling-map simulator -- the SAME backend already used elsewhere in
  this estate (QAE's FakeKingston validation).
- `hardware_readiness.lineage.capture_transpilation_metadata` (itself a
  thin wrapper on `algorithms.qae_iqae.backends.transpile_for`) for
  depth/gate-count capture.
- `experiments.quantum1b.classical_exact.solve_exact_enumeration` (Track A)
  as the ground truth.

**What is genuinely new here:** the COBYLA-driven optimization loop under
a noisy, device-connectivity-constrained simulator (QiskitQAOASolver.solve()
accepts a bare `noise_model`, not a full connectivity-aware backend --
using `kingston_sim()` directly, as this module does, is necessary to
capture real SWAP-routing depth growth, which is exactly the kind of
Hardware-Day-relevant fact a bare noise_model-only simulator would miss).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from algorithms.qae_iqae.backends import kingston_backend
from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising
from core.schemas import BackendDescriptor, BackendKind
from experiments.quantum1b.classical_exact import solve_exact_enumeration
from experiments.quantum1b.instance import OptionSelectionInstance, evaluate_selection
from experiments.quantum1b.qubo_encoding import QUBOEncoding, selection_from_bits
from experiments.quantum1b.hardware_readiness.lineage import (
    RunLineageRecord,
    build_lineage_record,
    build_qaoa_circuit_for_lineage,
    capture_transpilation_metadata,
)


@dataclass
class NoiseValidationRunResult:
    backend_label: str  # "ideal" | "fake_kingston"
    depth: int
    gate_counts: dict
    two_qubit_gates: int
    best_params: np.ndarray
    n_evaluations: int
    optimizer_status: str
    runtime_s: float
    feasible_fraction: float
    best_feasible_objective: float | None
    expected_objective: float
    exact_optimum: float
    objective_gap: float | None  # None if best_feasible_objective is None (no feasible sample)
    shots: int
    lineage_record: RunLineageRecord
    """**Added 2026-10-01, remediating independent-review finding P2-6**
    ('build_lineage_record() is dead code -- never invoked by any real or
    dry-run execution path'). This is the single real, exercised code path
    in this estate that already builds BOTH a genuine transpiled circuit
    (via `capture_transpilation_metadata`) AND calls
    `hardware_readiness.lineage.build_lineage_record()` with it -- Track
    D's EXP-D1 and the Quantum Lab's candidate-trade prototype both
    construct one of these per cell."""


def _decode_counts(
    counts: dict, qubo: QUBOEncoding, instance: OptionSelectionInstance
) -> tuple[float, float | None, float]:
    """Returns (feasible_fraction, best_feasible_objective_or_None,
    expected_objective) from a Qiskit `.get_counts()` dict, decoding each
    bitstring through the TRUE constrained objective
    (`instance.evaluate_selection`), never the QUBO's own claimed cost --
    same discipline as `cvar_qaoa.exact_distribution_metrics`."""
    total = sum(counts.values())
    n = qubo.n_qubits()
    feasible_shots = 0
    best_feasible: float | None = None
    expected = 0.0
    for bitstring_qiskit, c in counts.items():
        bits = [int(b) for b in bitstring_qiskit[::-1]]  # qiskit little-endian -> our bit order
        bits = bits[:n]
        sel = selection_from_bits(qubo, instance, bits)
        ev = evaluate_selection(instance, sel)
        weight = c / total
        expected += weight * ev.objective
        if ev.feasible:
            feasible_shots += c
            if best_feasible is None or ev.objective > best_feasible:
                best_feasible = ev.objective
    return feasible_shots / total, best_feasible, expected


def run_noise_validation_cell(
    instance: OptionSelectionInstance,
    qubo: QUBOEncoding,
    *,
    backend_label: str,
    backend,
    p: int = 1,
    maxiter: int = 15,
    shots: int = 1024,
    seed: int = 0,
    optimization_level: int = 1,
) -> NoiseValidationRunResult:
    """Runs one COBYLA-optimized QAOA circuit against `backend` (an Aer
    simulator -- either bare/ideal or FakeKingston-derived/noisy), decoding
    every shot through the true constrained objective. `backend` is
    injected so the ideal and noisy cells share identical circuit-
    construction and decoding code -- the ONLY thing that differs between
    cells is the backend object itself.

    `optimization_level` defaults to 1 (qiskit's own default). **An
    earlier version of this docstring recommended `optimization_level=0`
    for n_qubits>=14, based on a claimed transpile-time wall at
    `optimization_level=1`; that claim did not reproduce under
    independent review (finding P1-2) or this session's own follow-up --
    `optimization_level=1` completes in ~3s at the one cell directly
    measured (n_qubits=14, Track B's QAOA circuit) and produces a
    SHALLOWER circuit than `optimization_level=0` there. The recommendation
    is withdrawn; use the default (1) unless a specific, re-verified
    reason says otherwise** — see
    `experiments/quantum1b/calibrate_transpile_optimization_level.py`
    (committed, re-runnable) and `docs/quantum/QUANTUM1B_HARDWARE_READINESS.md`'s
    corrected EXP-D1 section."""
    from qiskit import transpile as qiskit_transpile

    t0 = time.perf_counter()
    J, h, _offset = qubo_to_ising(qubo.Q)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi / 2, 2 * p)
    eval_count = [0]

    def objective(params: np.ndarray) -> float:
        eval_count[0] += 1
        gamma, beta = params[:p], params[p:]
        qc = build_qaoa_circuit_for_lineage(qubo.Q, p=p, gamma=gamma, beta=beta)
        tqc = qiskit_transpile(qc, backend, optimization_level=optimization_level, seed_transpiler=seed)
        job = backend.run(tqc, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()
        _feas_frac, _best_feas, expected = _decode_counts(counts, qubo, instance)
        return -expected  # COBYLA minimizes; we want to MAXIMIZE expected true objective

    result = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
    best_params = np.asarray(result.x, dtype=float)
    gamma, beta = best_params[:p], best_params[p:]

    final_qc = build_qaoa_circuit_for_lineage(qubo.Q, p=p, gamma=gamma, beta=beta)
    transpiled = capture_transpilation_metadata(final_qc, backend, optimization_level=optimization_level, seed=seed)
    final_job = backend.run(transpiled.circuit, shots=shots, seed_simulator=seed)
    final_counts = final_job.result().get_counts()
    feasible_fraction, best_feasible_objective, expected_objective = _decode_counts(
        final_counts, qubo, instance
    )

    exact = solve_exact_enumeration(instance)
    gap = (
        (exact.objective - best_feasible_objective) / abs(exact.objective)
        if best_feasible_objective is not None and exact.objective != 0
        else None
    )

    runtime = time.perf_counter() - t0
    status = getattr(result, "message", None) or str(getattr(result, "status", "unknown"))

    backend_kind = BackendKind.FAKE if backend_label == "fake_kingston" else BackendKind.SIMULATOR
    lineage_record = build_lineage_record(
        instance_fingerprint=instance.fingerprint(),
        transpiled=transpiled,
        qubo_n_qubits=qubo.n_qubits(),
        backend=BackendDescriptor(
            name=getattr(backend, "name", backend_label), kind=backend_kind, num_qubits=qubo.n_qubits()
        ),
        shots=shots,
        seed=seed,
    )

    return NoiseValidationRunResult(
        backend_label=backend_label,
        depth=transpiled.depth,
        gate_counts=transpiled.gate_counts,
        two_qubit_gates=transpiled.twoq_gates,
        best_params=best_params,
        n_evaluations=eval_count[0],
        optimizer_status=str(status),
        runtime_s=runtime,
        feasible_fraction=feasible_fraction,
        best_feasible_objective=best_feasible_objective,
        expected_objective=expected_objective,
        exact_optimum=exact.objective,
        objective_gap=gap,
        shots=shots,
        lineage_record=lineage_record,
    )


def ideal_backend():
    from qiskit_aer import AerSimulator

    return AerSimulator()


def fake_kingston_backend():
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(kingston_backend())


__all__ = [
    "NoiseValidationRunResult",
    "run_noise_validation_cell",
    "ideal_backend",
    "fake_kingston_backend",
]
