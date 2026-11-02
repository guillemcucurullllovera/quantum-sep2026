"""SolverChoice.TRAINING_PIPELINE_TQA: routes a PortfolioSpec's QUBO through
IBM's QAOA training-pipeline methodology
(`algorithms_extended.qaoa_training_pipeline_asymmetrica`) instead of this
platform's own hand-rolled optimizer loops. Uses the real
`TQATrainer` (Trotterized-Quantum-Annealing parameter schedule, arXiv
Quantum 5, 491 (2021)) driven by the real `StatevectorEvaluator` (an
Aer-Estimator-backed energy evaluator) to produce QAOA angles, then samples
the resulting standard-mixer QAOA circuit on Aer to extract a candidate
bitstring -- a genuinely different training methodology from every other
generation's classical-optimizer-over-a-black-box-energy-function loop:
TQA produces a problem-size-scaling schedule from a single scalar (dt)
rather than optimizing 2p independent angles directly.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from algorithms.qaoa_qubo.qubo_builder import PortfolioQUBOBuilder, qubo_to_ising
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.evaluation.statevector_evaluator import (
    StatevectorEvaluator,
)
from algorithms_extended.qaoa_training_pipeline_asymmetrica.upstream.training.tqa_trainer import TQATrainer
from applications.portfolio.domain import PortfolioSpec
from applications.portfolio.exceptions import SolverFailureError
from core.errors import ValidationError


def _zstring(n: int, indices: set[int]) -> str:
    chars = ["I"] * n
    for i in indices:
        chars[i] = "Z"
    return "".join(reversed(chars))  # Qiskit little-endian: qubit 0 is the rightmost char


def _cost_operator(J: np.ndarray, h: np.ndarray, n: int) -> SparsePauliOp:
    terms: list[tuple[str, float]] = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            terms.append((_zstring(n, {i}), float(h[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                terms.append((_zstring(n, {i, j}), float(J[i, j])))
    if not terms:
        terms = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(terms)


def solve_training_pipeline_tqa(
    spec: PortfolioSpec,
    seed: int = 42,
    *,
    p: int = 2,
    shots: int = 2048,
) -> tuple[np.ndarray, float, float, dict]:
    """Returns (bitstring, qubo_cost, wall_time_s, extra_diagnostics)."""
    spec.validate_shape()
    n = spec.n_assets()
    if n > 16:
        raise ValidationError(
            f"TRAINING_PIPELINE_TQA: n={n} assets -- the statevector evaluator is impractical "
            "beyond ~16 qubits; use a different solver generation for larger universes."
        )

    from applications.portfolio.multi_solver import _to_portfolio_constraints

    constraints = _to_portfolio_constraints(spec)
    Q = PortfolioQUBOBuilder(spec.expected_returns(), spec.cov(), risk_aversion=spec.risk_aversion).build(constraints)
    J, h, _offset = qubo_to_ising(Q)
    # TQA's default dt=0.75 schedule implicitly assumes an O(1)-scaled
    # Hamiltonian (same reason algorithms/qaoa_qubo/gen1's QAOAOptimizer
    # computes its own gamma_max from the cost spread rather than using a
    # fixed constant) -- a cardinality-penalty-dominated Ising Hamiltonian
    # from this platform's QUBO builder is typically O(10) rather than O(1),
    # which left TQA unable to find a good schedule at all (verified: raw,
    # unnormalized angles landed on the all-zero/all-one bitstrings
    # regardless of p). Normalizing by the largest |coefficient| restores
    # the scale TQA's schedule was designed for; the reported `cost` is
    # still computed from the ORIGINAL, unscaled Q, so this normalization
    # is purely an internal training detail.
    scale = max(float(np.abs(h).max(initial=0.0)), float(np.abs(J).max(initial=0.0)), 1e-9)
    cost_op = _cost_operator(J / scale, h / scale, n)

    t0 = time.perf_counter()
    try:
        # TQATrainer's documented default convention is to MAXIMIZE the
        # evaluator's returned energy (see its docstring); this platform's
        # QUBO/Ising convention is the opposite -- x^T Q x (and its Ising
        # transform) is meant to be MINIMIZED, so energy_minimization=True
        # is required here or TQA drives toward the worst bitstring instead
        # of the best one.
        trainer = TQATrainer(evaluator=StatevectorEvaluator(), energy_minimization=True)
        param_result = trainer.train(cost_op=cost_op, reps=p)
    except Exception as exc:
        raise SolverFailureError(f"TRAINING_PIPELINE_TQA TQATrainer.train failed: {exc}") from exc
    angles = param_result.data["optimized_qaoa_angles"]

    circuit: QuantumCircuit = qaoa_ansatz(cost_op, reps=p)
    param_binding = dict(zip(circuit.parameters, angles))
    bound = circuit.assign_parameters(param_binding)
    bound.measure_all()

    backend = AerSimulator()
    job = backend.run(bound, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    dt = time.perf_counter() - t0

    best_bitstring_qiskit = max(counts, key=counts.get)
    little_endian = best_bitstring_qiskit[::-1]  # qubit i = little_endian[i]
    x = np.array([int(b) for b in little_endian], dtype=float)
    cost = float(x @ Q @ x)

    extra = {
        "tqa_dt": param_result.data["optimized_params"],
        "tqa_energy": param_result.data["energy"],
        "shots": shots,
        "best_state_frequency": counts[best_bitstring_qiskit] / shots,
    }
    return x, cost, dt, extra
