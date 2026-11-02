"""Unified QAOA solver interface: a fast NumPy engine (Gen2, default) and a
real Qiskit circuit + backend execution path (hardware-aware compilation,
noisy simulation, and — once `hardware.ibm_runtime` exists — real QPU
execution).

Scoping note: the Qiskit path currently implements the standard RX mixer
(|+>^n initial state), not the cardinality-preserving XY-mixer used by the
NumPy engines. Building an exact hardware circuit for XY-mixer Dicke-state
preparation is real, non-trivial engineering (see e.g. Qiskit's
`excitation_preserving`-family circuits) and is intentionally left for a
later tranche rather than approximated poorly here — see algorithms/qaoa_qubo/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from core.errors import CredentialError, ValidationError
from core.schemas import BackendDescriptor, BackendKind
from algorithms.qaoa_qubo.gen2 import QAOAv2, QAOAConfig
from algorithms.qaoa_qubo.qubo_builder import qubo_to_ising
from algorithms_extended.pennylane_asymmetrica.optimize.rotosolve import RotosolveOptimizer


def _cost_diagonal_from_ising(J: np.ndarray, h: np.ndarray, n: int) -> np.ndarray:
    dim = 2 ** n
    diag = np.zeros(dim)
    for idx in range(dim):
        bits = [(idx >> b) & 1 for b in range(n)]
        z = [1 - 2 * b for b in bits]
        e = sum(h[i] * z[i] for i in range(n))
        e += sum(J[i, j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n))
        diag[idx] = e
    return diag


@dataclass
class QAOASolverResult:
    best_bitstring: str
    best_cost: float
    best_params: np.ndarray
    engine: str
    n_evaluations: int
    approximation_ratio: float | None = None


def _build_qiskit_qaoa_circuit(J: np.ndarray, h: np.ndarray, p: int, gamma: np.ndarray, beta: np.ndarray) -> QuantumCircuit:
    """Standard-mixer QAOA circuit: |+>^n -> {exp(-i*gamma*H_cost), RX(2*beta)}^p -> measure.
    H_cost built from Ising (J, h) via real RZ/RZZ gates (Qiskit's native rzz)."""
    n = len(h)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for layer in range(p):
        g, b = gamma[layer], beta[layer]
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * g * J[i, j], i, j)
        for i in range(n):
            qc.rx(2 * b, i)
    qc.measure(range(n), range(n))
    return qc


class QiskitQAOASolver:
    """Hardware-aware QAOA execution: builds a real Qiskit circuit, transpiles
    it against a target backend, and evaluates the cost expectation from
    measured shot counts (correct for our diagonal cost Hamiltonian)."""

    def __init__(
        self,
        p: int = 2,
        shots: int = 4096,
        maxiter: int = 80,
        seed: int = 42,
        optimizer: Literal["cobyla", "rotosolve"] = "cobyla",
        rotosolve_num_steps: int = 4,
        rotosolve_Ns: int = 11,
        rotosolve_n_sweeps: int = 3,
    ) -> None:
        """
        Args:
            optimizer: "cobyla" (default, scipy) or "rotosolve" -- a
                gradient-free coordinate-descent optimizer adapted from
                PennyLane's RotosolveOptimizer
                (algorithms_extended/pennylane_asymmetrica/optimize/rotosolve.py).
                Uses the general "brute" substep (iterative 1-D grid search,
                not the closed-form single-frequency case) because gamma/beta
                here each multiply several different Ising coefficients at
                once within a layer, so the restricted cost is generally a
                sum of several different frequencies in that one angle, not
                a single sinusoid.
            rotosolve_num_steps, rotosolve_Ns: only used for optimizer="rotosolve" --
                number of coarse-to-fine refinement rounds and grid points per
                round of the per-parameter 1-D brute-force substep (scipy's
                `brute` also runs its default Nelder-Mead polish after each
                grid search, so actual evaluation counts run higher than
                `rotosolve_num_steps * rotosolve_Ns` alone would suggest).
            rotosolve_n_sweeps: number of full coordinate-descent sweeps
                over all 2*p parameters. `maxiter` is COBYLA-specific (it
                counts scalar cost evaluations, scipy's own unit for that
                method) and is not reused here -- one Rotosolve sweep costs
                roughly `2*p*rotosolve_num_steps*rotosolve_Ns` evaluations by
                itself, an incomparable unit, so the sweep count is set
                directly instead of derived from `maxiter`.
        """
        self.p = p
        self.shots = shots
        self.maxiter = maxiter
        self.seed = seed
        self.optimizer = optimizer
        self.rotosolve_num_steps = rotosolve_num_steps
        self.rotosolve_Ns = rotosolve_Ns
        self.rotosolve_n_sweeps = rotosolve_n_sweeps

    def solve(
        self,
        Q: np.ndarray,
        *,
        backend_descriptor: BackendDescriptor | None = None,
        noise_model=None,
    ) -> QAOASolverResult:
        n = Q.shape[0]
        if n > 14:
            raise ValidationError(
                f"QiskitQAOASolver.solve: n={n} qubits — statevector-backed AerSimulator "
                "shot execution here is only practical up to ~14 qubits without a real "
                "QPU/noise-aware sampler backend; use the NumPy engine for larger problems."
            )
        J, h, _offset = qubo_to_ising(Q)
        cost_diag = _cost_diagonal_from_ising(J, h, n)

        backend = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()
        rng = np.random.default_rng(self.seed)
        x0 = rng.uniform(0, np.pi / 2, 2 * self.p)
        eval_count = [0]

        def objective(params: np.ndarray) -> float:
            eval_count[0] += 1
            gamma, beta = params[: self.p], params[self.p :]
            qc = _build_qiskit_qaoa_circuit(J, h, self.p, gamma, beta)
            transpiled = transpile(qc, backend, optimization_level=1, seed_transpiler=self.seed)
            job = backend.run(transpiled, shots=self.shots, seed_simulator=self.seed)
            counts = job.result().get_counts()
            total = sum(counts.values())
            exp = 0.0
            for bitstring, c in counts.items():
                idx = int(bitstring[::-1], 2)  # qiskit little-endian bit order
                exp += (c / total) * cost_diag[idx]
            return exp

        if self.optimizer == "rotosolve":
            rotosolver = RotosolveOptimizer(
                substep="brute",
                brute_num_steps=self.rotosolve_num_steps,
                brute_Ns=self.rotosolve_Ns,
                bounds=(0.0, np.pi),
            )
            best_x, _cost_history = rotosolver.minimize(
                objective, x0, n_steps=self.rotosolve_n_sweeps
            )
        else:
            result = minimize(objective, x0, method="COBYLA", options={"maxiter": self.maxiter})
            best_x = result.x

        gamma, beta = best_x[: self.p], best_x[self.p :]
        qc = _build_qiskit_qaoa_circuit(J, h, self.p, gamma, beta)
        transpiled = transpile(qc, backend, optimization_level=1, seed_transpiler=self.seed)
        job = backend.run(transpiled, shots=self.shots, seed_simulator=self.seed)
        counts = job.result().get_counts()
        best_bitstring_qiskit = max(counts, key=counts.get)
        best_bitstring = best_bitstring_qiskit[::-1]
        total = sum(counts.values())
        best_cost = sum(
            (c / total) * cost_diag[int(bitstring[::-1], 2)] for bitstring, c in counts.items()
        )

        return QAOASolverResult(
            best_bitstring=best_bitstring,
            best_cost=float(best_cost),
            best_params=best_x,
            engine="qiskit_aer",
            n_evaluations=eval_count[0],
        )

    def run_on_ibm_runtime(self, *args, **kwargs):
        """Real-QPU execution path. Deliberately not implemented here: this
        requires `hardware.ibm_runtime` session/job-lifecycle infrastructure
        (built in a sibling workstream) and `core.config.get_settings().ibm`
        credentials — never faked or stubbed to "succeed" silently."""
        from core.config import get_settings

        get_settings().ibm.require()  # raises CredentialError if not configured, even before the NotImplementedError
        raise NotImplementedError(
            "QPU execution is wired through hardware.ibm_runtime, not solvers.py directly. "
            "This method exists so the interface is discoverable; integrate once "
            "hardware.ibm_runtime.session management lands."
        )


class NumpyQAOASolver:
    """Wraps the Gen2 NumPy engine (correlation-XY mixer, SK warm-start) —
    fast enough for unit tests and for validating constraint satisfaction."""

    def __init__(self, p: int = 2, n_iter: int = 200, seed: int = 42) -> None:
        self.p = p
        self.n_iter = n_iter
        self.seed = seed

    def solve(
        self, Q: np.ndarray, *, cardinality_k: int | None = None, corr_matrix: np.ndarray | None = None
    ) -> QAOASolverResult:
        J, h, _offset = qubo_to_ising(Q)
        # Gen2's build_cost_hamiltonian expects diagonal = h_i, off-diag = J_ij
        J_full = J + J.T
        np.fill_diagonal(J_full, h)

        cfg = QAOAConfig(
            p=self.p, n_iter=self.n_iter, seed=self.seed,
            use_dicke_init=cardinality_k is not None, target_k=cardinality_k,
        )
        engine = QAOAv2(cfg)
        engine.fit(J_full, corr_matrix=corr_matrix, warmstart=True)
        portfolio = engine.sample_portfolio()
        bitstring = "".join(str(int(b)) for b in portfolio)
        cost = float(portfolio @ Q @ portfolio)

        approx_ratio = None
        if Q.shape[0] <= 16:
            bf_energy, _ = engine.brute_force_optimum(k=cardinality_k)
            if abs(bf_energy) > 1e-10:
                approx_ratio = engine.best_energy_ / bf_energy

        return QAOASolverResult(
            best_bitstring=bitstring,
            best_cost=cost,
            best_params=engine.best_angles_,
            engine="numpy_gen2",
            n_evaluations=self.n_iter,
            approximation_ratio=approx_ratio,
        )


def solve_qubo(
    Q: np.ndarray,
    *,
    engine: Literal["numpy", "qiskit"] = "numpy",
    cardinality_k: int | None = None,
    **kwargs,
) -> QAOASolverResult:
    """Convenience dispatcher used by applications/portfolio."""
    if engine == "numpy":
        return NumpyQAOASolver(**kwargs).solve(Q, cardinality_k=cardinality_k)
    elif engine == "qiskit":
        return QiskitQAOASolver(**kwargs).solve(Q)
    raise ValidationError(f"Unknown QAOA engine '{engine}', expected 'numpy' or 'qiskit'")
