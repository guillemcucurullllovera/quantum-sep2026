"""Integration test: RotosolveOptimizer (brute substep) wired into
``algorithms.qaoa_qubo.solvers.QiskitQAOASolver`` as a real alternative to
its default COBYLA optimizer.

Kept deliberately small (n=3, p=1, few shots, one Rotosolve sweep) so it
runs in reasonable CI time -- ``QiskitQAOASolver`` transpiles+runs a real
``AerSimulator`` circuit per cost evaluation, and Rotosolve's brute substep
needs many more evaluations per parameter than COBYLA needs per iterate (see
the docstring in ``RotosolveOptimizer`` for why: scipy's ``brute`` also runs
its default Nelder-Mead polish after each grid search).
"""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.qaoa_qubo.solvers import QiskitQAOASolver
from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising


def _random_qubo(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.uniform(-1, 1, (n, n))
    return (Q + Q.T) / 2


class TestRotosolveWiredIntoQAOASolver:
    def test_rotosolve_path_runs_and_returns_valid_result(self):
        Q = _random_qubo(3, seed=9)
        result = QiskitQAOASolver(
            p=1, shots=128, seed=1, optimizer="rotosolve",
            rotosolve_num_steps=1, rotosolve_Ns=4, rotosolve_n_sweeps=1,
        ).solve(Q)

        assert result.engine == "qiskit_aer"
        assert len(result.best_bitstring) == 3
        assert set(result.best_bitstring) <= {"0", "1"}
        assert len(result.best_params) == 2  # 2*p = 2
        assert np.isfinite(result.best_cost)
        assert result.n_evaluations > 0

    def test_default_optimizer_is_still_cobyla(self):
        # Backward compatibility: existing callers that don't pass
        # `optimizer=` must keep getting COBYLA behavior unchanged.
        solver = QiskitQAOASolver(p=1, shots=128, maxiter=10, seed=1)
        assert solver.optimizer == "cobyla"

    def test_cobyla_and_rotosolve_both_beat_a_uniform_random_baseline(self):
        # Real benchmark: both optimizers should do better, on average, than
        # evaluating the QAOA circuit at random angles -- a sanity floor
        # that doesn't require either optimizer to beat the other (their
        # per-evaluation cost differs too much for a fair apples-to-apples
        # "best cost at fixed evaluations" comparison here; that trade-off
        # is documented in THIRD_PARTY_NOTICES.md's benchmark table).
        Q = _random_qubo(3, seed=4)
        J, h, _ = qubo_to_ising(Q)
        n = Q.shape[0]

        from algorithms.qaoa_qubo.solvers import _cost_diagonal_from_ising, _build_qiskit_qaoa_circuit
        from qiskit.quantum_info import Statevector

        cost_diag = _cost_diagonal_from_ising(J, h, n)
        rng = np.random.default_rng(0)
        random_costs = []
        for _ in range(30):
            gamma = rng.uniform(0, np.pi / 2, 1)
            beta = rng.uniform(0, np.pi / 2, 1)
            qc = _build_qiskit_qaoa_circuit(J, h, 1, gamma, beta)
            qc.remove_final_measurements()
            probs = Statevector(qc).probabilities()
            random_costs.append(float(np.dot(probs, cost_diag)))
        random_baseline = float(np.mean(random_costs))

        result_cobyla = QiskitQAOASolver(p=1, shots=256, maxiter=15, seed=2, optimizer="cobyla").solve(Q)
        result_roto = QiskitQAOASolver(
            p=1, shots=256, seed=2, optimizer="rotosolve",
            rotosolve_num_steps=1, rotosolve_Ns=5, rotosolve_n_sweeps=1,
        ).solve(Q)

        assert result_cobyla.best_cost < random_baseline
        assert result_roto.best_cost < random_baseline
