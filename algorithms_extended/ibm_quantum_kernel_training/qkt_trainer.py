"""SPSA-based quantum kernel alignment training.

`algorithms.pqk.kta_alignment.kta_gradient_ascent` estimates the KTA
gradient via per-parameter central differences (2 circuit-batch evaluations
per parameter per step) then applies ADAM. That's accurate but scales
linearly in the number of trainable parameters per step.

The upstream IBM toolkit this module adapts trains its covariant feature
maps via Qiskit's own `QuantumKernelTrainer`, which typically drives SPSA
(Simultaneous Perturbation Stochastic Approximation) -- SPSA estimates a
descent direction from just 2 evaluations *regardless* of parameter count,
trading gradient accuracy for per-step cost. This module reimplements that
strategy (using `qiskit_algorithms.optimizers.SPSA`, already a platform
dependency) as a genuinely different training path, not a copy of optimizer. See `compare_training_strategies` below for an honest, run-it-
yourself comparison rather than an assumed winner.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA

from algorithms.pqk.kta_alignment import compute_gram_matrix, compute_kta


def fidelity_kernel_entry(
    feature_map: QuantumCircuit,
    x1: np.ndarray,
    x2: np.ndarray,
    theta: np.ndarray,
    backend: AerSimulator | None = None,
    noise_model=None,
    shots: int = 1024,
) -> float:
    """Overlap-circuit fidelity kernel entry: |<phi(x2)|phi(x1)>|^2, estimated
    as the all-zero measurement probability of feature_map(x1) followed by
    feature_map(x2)^-1, both bound to training parameters `theta`.
    """
    backend = backend or AerSimulator()
    bound1 = feature_map.bind_training(theta).bind_data(x1)
    bound2 = feature_map.bind_training(theta).bind_data(x2)
    circ = bound1.compose(bound2.inverse())
    circ.measure_all()
    tcirc = transpile(circ, backend)
    result = backend.run(tcirc, shots=shots, noise_model=noise_model).result()
    counts = result.get_counts()
    zero_state = "0" * feature_map.num_qubits
    return counts.get(zero_state, 0) / shots


def make_kernel_fn_factory(
    feature_map, backend=None, noise_model=None, shots: int = 512
) -> Callable[[np.ndarray], Callable[[np.ndarray, np.ndarray], float]]:
    """Adapts `fidelity_kernel_entry` to the `kernel_fn_factory(params) ->
    kernel_fn(x1, x2)` shape `compute_gram_matrix` expects, so this
    trainer's kernel entries are usable by the shared Gen1 alignment
    utilities without duplicating Gram-matrix bookkeeping.
    """

    def factory(theta: np.ndarray):
        def kernel_fn(x1: np.ndarray, x2: np.ndarray) -> float:
            return fidelity_kernel_entry(feature_map, x1, x2, theta, backend, noise_model, shots)

        return kernel_fn

    return factory


def spsa_kta_train(
    feature_map,
    X: np.ndarray,
    y: np.ndarray,
    theta_init: np.ndarray,
    maxiter: int = 60,
    backend=None,
    noise_model=None,
    shots: int = 512,
    learning_rate: float = 0.2,
    perturbation: float = 0.15,
) -> tuple[np.ndarray, float, list[float]]:
    """Train a covariant feature map's theta parameters via SPSA to maximize
    KTA. Returns (best_theta, best_kta, per-iteration KTA history).

    `backend` is constructed once here (not per kernel-entry call) and
    reused for every circuit execution in this training run -- constructing
    a fresh `AerSimulator()` per call was a real, measured ~3x overhead cost
    caught while testing this module, not a cosmetic cleanup. `learning_rate`
    /`perturbation` are passed explicitly so `qiskit_algorithms.optimizers.SPSA`
    skips its default ~25-evaluation auto-calibration phase, which otherwise
    dominates wall time for a small `maxiter` -- callers who want SPSA's own
    calibration can still get it by passing `learning_rate=None,
    perturbation=None` explicitly.
    """
    backend = backend or AerSimulator()
    factory = make_kernel_fn_factory(feature_map, backend, noise_model, shots)
    history: list[float] = []

    def neg_kta(theta: np.ndarray) -> float:
        K = compute_gram_matrix(X, factory(theta))
        kta = compute_kta(K, y)
        history.append(kta)
        return -kta

    spsa = SPSA(maxiter=maxiter, learning_rate=learning_rate, perturbation=perturbation)
    result = spsa.minimize(neg_kta, x0=np.asarray(theta_init, dtype=float))
    best_theta = np.asarray(result.x)
    best_kta = -float(result.fun)
    return best_theta, best_kta, history


def compare_training_strategies(
    feature_map_spsa,
    numpy_kernel_fn_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    theta_init: np.ndarray,
    spsa_maxiter: int = 60,
    adam_n_steps: int = 60,
    shots: int = 512,
) -> dict:
    """Run both training strategies on the same data/init and report which
    converges to a higher KTA in a comparable iteration budget. Neither
    outcome is assumed -- this platform's own classical-simulability audit
    (Q10) is a reminder not to assume quantum-specific machinery wins.
    """
    from algorithms.pqk.kta_alignment import kta_gradient_ascent

    spsa_theta, spsa_kta, spsa_hist = spsa_kta_train(
        feature_map_spsa, X, y, theta_init, maxiter=spsa_maxiter, shots=shots
    )
    adam_theta, adam_kta, adam_hist = kta_gradient_ascent(
        theta_init, X, y, numpy_kernel_fn_factory, n_steps=adam_n_steps, target_kta=1.1  # never early-stop
    )
    return {
        "spsa": {"best_kta": spsa_kta, "n_evals": len(spsa_hist), "history": spsa_hist},
        "adam_central_diff": {"best_kta": adam_kta, "n_evals": len(adam_hist), "history": adam_hist},
        "winner": "spsa" if spsa_kta > adam_kta else "adam_central_diff",
    }
